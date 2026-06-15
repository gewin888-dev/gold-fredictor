"""NewsAPI 新闻情绪采集器（VADER 情感分析版）。

数据来源：NewsAPI.org — 免费 100 次/天，全球新闻聚合
采集逻辑：搜索 gold 相关新闻 → 关键词情感打分 → SQLite 入库
"""

from __future__ import annotations

import re
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from sqlalchemy import case, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.config import get_settings
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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


_vader: SentimentIntensityAnalyzer | None = None

def _get_vader() -> SentimentIntensityAnalyzer:
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    return _vader

def _vader_sentiment(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    scores = _get_vader().polarity_scores(text)
    return round(scores["compound"] * 5.0, 3)

def _gold_keyword_direction(text: str) -> int:
    text_lower = text.lower()
    bull = sum(1 for kw in BULLISH if re.search(kw, text_lower, re.IGNORECASE))
    bear = sum(1 for kw in BEARISH if re.search(kw, text_lower, re.IGNORECASE))
    if bull > bear: return 1
    elif bear > bull: return -1
    return 0

def _gold_sentiment(title: str, description: str = "") -> float:
    text = (title + " " + description[:300]).strip()
    if not text: return 0.0
    vader_s = _vader_sentiment(text)
    kw_d = _gold_keyword_direction(text)
    if kw_d == 0:
        return round(vader_s * 0.5, 3)
    magnitude = max(0.3, abs(vader_s))
    return round(kw_d * magnitude, 3)

def _text_sentiment(title: str, description: str = "") -> float:
    """兼容旧调用"""
    return _gold_sentiment(title, description)


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

            spread_seconds = count * 23
            article_ts = now - timedelta(seconds=spread_seconds)
            sent = _gold_sentiment(title, description)

            summary = description[:300] if description else None
            stmt = sqlite_insert(NewsSentiment).values(
                timestamp=article_ts,
                source_url=url,
                title=title[:500],
                sentiment_score=sent,
                relevance=None,
                summary=summary,
                source="NEWSAPI",
                updated_at=article_ts,
            )
            db.execute(stmt)
            count += 1

        return count
    except requests.RequestException as e:
        raise RuntimeError(f"NewsAPI request failed: {e}") from e


def get_recent_sentiment(db: Session, days: int = 7) -> float | None:
    """获取最近 N 天平均新闻情绪评分。"""
    from sqlalchemy import case, func, select

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = db.scalar(
        select(func.avg(NewsSentiment.sentiment_score)).where(
            NewsSentiment.timestamp >= cutoff,
            NewsSentiment.source.in_(["NEWSAPI", "GDELT"]),
        )
    )
    return round(float(result), 4) if result else None


def get_daily_sentiment_trend(db: Session, days: int = 30) -> list[dict[str, Any]]:
    """获取每日情绪聚合趋势，用于仪表盘折线图。"""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = db.execute(
        select(
            func.date(NewsSentiment.timestamp).label("day"),
            func.avg(NewsSentiment.sentiment_score).label("avg_score"),
            func.count().label("count"),
            func.sum(
                case((NewsSentiment.sentiment_score > 0, 1), else_=0)
            ).label("bullish_count"),
        )
        .where(
            NewsSentiment.timestamp >= cutoff,
            NewsSentiment.source.in_(["NEWSAPI", "GDELT"]),
        )
        .group_by("day")
        .order_by("day")
    ).all()
    return [
        {
            "date": str(day),
            "avg_score": round(float(avg), 3) if avg else 0.0,
            "count": int(cnt),
            "bullish_pct": round(float(bull) / int(cnt), 3) if cnt else 0.0,
        }
        for day, avg, cnt, bull in rows
    ]


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
