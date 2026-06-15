"""NewsAPI 新闻情绪采集器。

数据来源：NewsAPI.org — 免费 100 次/天，全球新闻聚合
采集逻辑：搜索 gold 相关新闻 → 关键词情感打分 → SQLite 入库
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import NewsSentiment

NEWSAPI_URL = "https://newsapi.org/v2/everything"

# 黄金利多/利空关键词
BULLISH = [
    "上涨", "飙升", "创新高", "避险", "降息", "宽松", "地缘", "冲突",
    "央行购金", "通胀攀升", "衰退", "违约", "金价.*涨", "黄金.*涨",
    "rally", "surge", "record high", "safe haven", "rate cut",
    "geopolitical", "inflation hedge", "recession", "central bank gold",
    "bullion", "bullish", "outperform", "outlook positive",
]
BEARISH = [
    "下跌", "暴跌", "加息", "鹰派", "美元走强", "风险偏好", "获利了结",
    "金价.*跌", "黄金.*跌", "taper", "tightening",
    "sell-off", "strong dollar", "risk-on", "hawkish",
    "bearish", "underperform", "decline", "plunge", "outlook negative",
]


def _text_sentiment(title: str, description: str = "") -> float:
    """基于关键词的情感评分。综合标题和描述。正值=利多，负值=利空。"""
    text = (title + " " + description).lower()
    score = 0.0
    for kw in BULLISH:
        if re.search(kw, text, re.IGNORECASE):
            score += 1.2
    for kw in BEARISH:
        if re.search(kw, text, re.IGNORECASE):
            score -= 1.2
    return max(-5.0, min(5.0, score))


def collect_news_sentiment(
    db: Session,
    query: str = "gold price OR gold market",
    days_back: int = 3,
    max_records: int = 50,
) -> int:
    """从 NewsAPI 采集黄金相关新闻情绪并入库。"""
    settings = get_settings()
    if not settings.newsapi_key:
        raise ValueError("NEWSAPI_KEY not configured in .env")

    from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    requested_records = max(1, min(max_records, settings.newsapi_daily_limit, 100))

    params = {
        "q": query or "gold price OR gold market OR central bank gold OR gold investment",
        "from": from_date,
        "sortBy": "publishedAt",
        "pageSize": requested_records,
        "language": "en",
        "apiKey": settings.newsapi_key,
    }

    try:
        resp = requests.get(
            NEWSAPI_URL, params=params,
            headers={"User-Agent": "gold-fredictor/1.0"},
            timeout=settings.newsapi_timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            return 0

        articles = data.get("articles", [])
        if not articles:
            return 0

        now = datetime.now(timezone.utc)
        count = 0
        seen_urls: set[str] = set()
        candidate_urls = [
            (article.get("url") or "").strip()[:500]
            for article in articles
            if (article.get("url") or "").strip()
        ]
        existing_urls = (
            set(
                db.scalars(
                    select(NewsSentiment.source_url).where(
                        NewsSentiment.source_url.in_(candidate_urls)
                    )
                ).all()
            )
            if candidate_urls
            else set()
        )

        for article in articles:
            title = (article.get("title") or "").strip()
            url = (article.get("url") or "").strip()
            description = (article.get("description") or "").strip()

            url = url[:500]
            if not title or not url or url in seen_urls or url in existing_urls:
                continue
            seen_urls.add(url)

            sent = _text_sentiment(title, description)

            summary = description[:300] if description else None
            stmt = sqlite_insert(NewsSentiment).values(
                timestamp=now,
                source_url=url,
                title=title[:500],
                sentiment_score=sent,
                relevance=None,
                summary=summary,
                source="NEWSAPI",
                updated_at=now,
            )
            db.execute(stmt)
            count += 1

        return count
    except requests.RequestException as e:
        raise RuntimeError(f"NewsAPI request failed: {e}") from e


def get_recent_sentiment(db: Session, days: int = 7) -> float | None:
    """获取最近 N 天平均新闻情绪评分。"""
    from sqlalchemy import func, select

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = db.scalar(
        select(func.avg(NewsSentiment.sentiment_score)).where(
            NewsSentiment.timestamp >= cutoff,
            NewsSentiment.source.in_(["NEWSAPI", "GDELT"]),
        )
    )
    return round(float(result), 4) if result else None


def load_sample_sentiment(db: Session, days: int = 30) -> int:
    """测试兼容入口：生产环境不再加载样例新闻。"""
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    count = 0
    for i in range(days):
        stmt = sqlite_insert(NewsSentiment).values(
            timestamp=now - timedelta(days=days - i),
            source_url=f"https://example.com/test-gold-news/{i}",
            title=f"Test gold news item {i}",
            sentiment_score=0.0,
            relevance=None,
            summary="测试新闻情绪数据。",
            source="TEST",
            updated_at=now,
        )
        db.execute(stmt)
        count += 1
    db.commit()
    return count
