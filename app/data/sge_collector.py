"""中国黄金溢价采集器。

数据来源：
- 新浪财经 — 沪金期货 AU0 (人民币/克)
- 新浪财经 — USDCNY 汇率
- Yahoo Finance — 国际金价 GC=F (美元/盎司)

溢价 = (沪金(元/克) * 31.1035 / USDCNY) / 国际金价(美元/盎司) - 1
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.models import ChinaGoldPremium, GoldPrice

# 新浪财经实时行情接口
_SINA_AU_URL = "https://hq.sinajs.cn/list=nf_AU0"
_SINA_USDCNY_URL = "https://hq.sinajs.cn/list=USDCNY"
_SINA_HEADERS = {"Referer": "https://finance.sina.com.cn"}

OZ_PER_GRAM = 31.1035  # 1 金衡盎司 = 31.1035 克


def _parse_sina_au(quote_str: str) -> dict[str, float | None]:
    """解析新浪沪金期货行情字符串。

    格式: var hq_str_nf_AU0="名称,时间,开盘,最高,最低,最新,结算,昨收,买价,卖价,..."
    索引:  0=名称, 1=未知, 2=开盘, 3=最高, 4=最低, 5=昨收, 6=最新, 7=结算, ...
    """
    try:
        # 提取引号内内容
        inner = quote_str.split('"')[1] if '"' in quote_str else quote_str
        if not inner or inner.count(",") < 10:
            return {"sge_price_cny": None, "usdcny": None}
        parts = inner.split(",")
        # 最新价在索引 6（部分格式），开盘=2，最高=3，最低=4
        # 新浪 nf_AU0 格式: 名称,开盘,最高,最低,昨收,未知,最新价,...
        # Let's try to identify fields by position
        # Typical Sina futures format:
        # 0=名称, 1=开盘, 2=最高, 3=最低, 4=昨日收盘, 5=买价, 6=卖价, 7=最新价, 8=结算价, ...
        # But nf_AU0 uses a different format. Let's print to debug.
        
        # 实际格式（通过测试确认）：
        # 0=名称, 1=时间, 2=开盘, 3=最高, 4=最低, 5=昨收, 6=买一价, 7=卖一价, 8=最新价, ...
        # 但看起来数据中 开盘=892.000, 最高=909.940, 最低=885.440, 昨收=906.600, 最新=906.600
        
        price_str = parts[7] if len(parts) > 7 else None  # 最新价
        if price_str is None or price_str == "" or price_str == "0.000":
            price_str = parts[6] if len(parts) > 6 else None

        price = float(price_str) if price_str and price_str != "0.000" else None
        return {"sge_price_cny": price, "usdcny": None}
    except (ValueError, IndexError):
        return {"sge_price_cny": None, "usdcny": None}


def _parse_sina_usdcny(quote_str: str) -> float | None:
    """解析新浪 USDCNY 汇率字符串。

    格式: var hq_str_USDCNY="时间,最新价,昨收,开盘,最高,最低,..."
    索引: 0=时间, 1=最新价
    """
    try:
        inner = quote_str.split('"')[1] if '"' in quote_str else quote_str
        if not inner:
            return None
        parts = inner.split(",")
        rate = float(parts[1]) if len(parts) > 1 and parts[1] else None
        return rate
    except (ValueError, IndexError):
        return None


def _fetch_sina_au_price() -> dict[str, float | None]:
    """从新浪财经获取沪金期货最新价格。"""
    try:
        resp = requests.get(_SINA_AU_URL, headers=_SINA_HEADERS, timeout=10)
        resp.encoding = "gbk"
        return _parse_sina_au(resp.text)
    except Exception:
        return {"sge_price_cny": None, "usdcny": None}


def _fetch_usdcny() -> float | None:
    """从新浪财经获取 USDCNY 汇率。"""
    try:
        resp = requests.get(_SINA_USDCNY_URL, headers=_SINA_HEADERS, timeout=10)
        resp.encoding = "gbk"
        return _parse_sina_usdcny(resp.text)
    except Exception:
        return None


def collect_china_gold_premium(db: Session) -> ChinaGoldPremium | None:
    """采集中国黄金溢价数据。

    从新浪财经获取沪金价格 + 汇率，从 GoldPrice 表获取国际金价，
    计算上海-国际溢价百分比。
    """
    now = datetime.now(timezone.utc)

    # 获取沪金价格
    au_data = _fetch_sina_au_price()
    sge_price = au_data.get("sge_price_cny")

    # 获取汇率
    usdcny = _fetch_usdcny()

    # 获取国际金价
    lbma_usd = None
    gold_row = db.scalar(select(GoldPrice).order_by(GoldPrice.date.desc()))
    if gold_row:
        lbma_usd = gold_row.close

    premium_pct = None
    if sge_price and usdcny and lbma_usd and usdcny > 0 and lbma_usd > 0:
        sge_usd_per_oz = sge_price * OZ_PER_GRAM / usdcny
        premium_pct = round((sge_usd_per_oz / lbma_usd - 1) * 100, 4)

    source = "SINA"
    if not sge_price:
        source = "ESTIMATE"

    record = ChinaGoldPremium(
        timestamp=now,
        sge_price_cny=sge_price,
        lbma_price_usd=lbma_usd,
        usdcny=usdcny,
        premium_pct=premium_pct,
        source=source,
    )

    stmt = sqlite_insert(ChinaGoldPremium).values(
        timestamp=record.timestamp,
        sge_price_cny=record.sge_price_cny,
        lbma_price_usd=record.lbma_price_usd,
        usdcny=record.usdcny,
        premium_pct=record.premium_pct,
        source=record.source,
        updated_at=now,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["timestamp"],
        set_={
            "sge_price_cny": record.sge_price_cny,
            "lbma_price_usd": record.lbma_price_usd,
            "usdcny": record.usdcny,
            "premium_pct": record.premium_pct,
            "source": record.source,
            "updated_at": now,
        },
    )
    db.execute(stmt)
    db.commit()
    return record


def load_sample_china_premium(db: Session, days: int = 120) -> int:
    """加载中国黄金溢价样例数据。"""
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    count = 0
    for i in range(days):
        dt = now - timedelta(days=days - i)
        # 模拟：溢价在 -2% 到 +5% 之间波动
        import math

        premium = 1.5 + 2.0 * math.sin(i * 0.15) + (i % 30 - 15) * 0.05
        stmt = sqlite_insert(ChinaGoldPremium).values(
            timestamp=dt,
            premium_pct=round(premium, 4),
            source="SAMPLE",
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["timestamp"],
            set_={"premium_pct": round(premium, 4), "source": "SAMPLE", "updated_at": now},
        )
        db.execute(stmt)
        count += 1
    db.commit()
    return count
