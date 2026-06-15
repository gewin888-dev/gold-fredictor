"""央行黄金储备 / 购金数据采集器。

数据来源：data/cb_gold_monthly.json — 按月维护，每月更新。
更新方式：编辑 JSON 文件后调用 POST /collect/cb_gold 即可刷新数据库。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.models import CentralBankGold

# JSON 数据文件路径
_DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "cb_gold_monthly.json"


def _load_json() -> dict:
    """读取月度央行购金 JSON 数据文件。"""
    if not _DATA_FILE.exists():
        raise FileNotFoundError(f"数据文件不存在: {_DATA_FILE}")
    with open(_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_central_bank_gold(db: Session) -> int:
    """从 JSON 数据文件加载央行购金月度数据到数据库（覆盖式 upsert）。

    返回写入的记录数。
    """
    j = _load_json()
    countries_map: dict[str, str] = j["countries"]
    reserves: dict[str, float] = j.get("reserves", {})
    data: list[dict] = j["data"]
    now = datetime.now(timezone.utc)
    count = 0

    for row in data:
        period = row["period"]  # "2025-06"
        # 用当月第一天作为 timestamp
        y, m = period.split("-")
        ts = datetime(int(y), int(m), 1, tzinfo=timezone.utc)

        # 计算 Global 合计
        global_net = sum(row.get(code, 0) or 0 for code in countries_map)

        # Global 汇总
        count += _upsert(db, "Global", period, ts, global_net, "WGC", now)

        # 各国家明细
        for code in countries_map:
            net = row.get(code, 0) or 0
            count += _upsert(
                db, code, period, ts, net, "WGC", now, reserves.get(code)
            )

    return count


def _upsert(
    db: Session,
    country: str,
    period: str,
    timestamp: datetime,
    net_change: float,
    source: str,
    now: datetime,
    reserves_val: float | None = None,
) -> int:
    stmt = sqlite_insert(CentralBankGold).values(
        country=country,
        period=period,
        timestamp=timestamp,
        reserves_tonnes=reserves_val,
        net_change_tonnes=net_change,
        source=source,
        updated_at=now,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["country", "period"],
        set_={
            "net_change_tonnes": net_change,
            "reserves_tonnes": reserves_val,
            "source": source,
            "updated_at": now,
        },
    )
    db.execute(stmt)
    return 1


def load_sample_cb_gold(db: Session, quarters: int = 8) -> int:
    """向后兼容。实际数据从 JSON 加载。"""
    return collect_central_bank_gold(db)
