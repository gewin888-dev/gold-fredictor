"""数据获取工具函数。"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import GoldPrice


def gold_price_frame(db: Session) -> pd.DataFrame:
    """从 GoldPrice 表读取金价日线，返回 DataFrame。"""
    rows = db.scalars(select(GoldPrice).order_by(GoldPrice.date.asc())).all()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "gold_price"])
    return pd.DataFrame(
        [{"timestamp": row.date, "gold_price": row.close} for row in rows]
    ).sort_values("timestamp")
