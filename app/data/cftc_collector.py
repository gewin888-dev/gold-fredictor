from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.data.cftc_client import CftcClient, CftcPositionRecord
from app.models import CftcPosition


def upsert_cftc_position(db: Session, record: CftcPositionRecord, source: str = "CFTC") -> None:
    now = datetime.now(timezone.utc)
    stmt = sqlite_insert(CftcPosition).values(
        market_name=record.market_name,
        contract_market_code=record.contract_market_code,
        exchange_code=record.exchange_code,
        timestamp=record.timestamp,
        open_interest=record.open_interest,
        noncommercial_long=record.noncommercial_long,
        noncommercial_short=record.noncommercial_short,
        noncommercial_spreading=record.noncommercial_spreading,
        commercial_long=record.commercial_long,
        commercial_short=record.commercial_short,
        noncommercial_net=record.noncommercial_net,
        source=source,
        updated_at=now,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["contract_market_code", "timestamp"],
        set_={
            "market_name": record.market_name,
            "exchange_code": record.exchange_code,
            "open_interest": record.open_interest,
            "noncommercial_long": record.noncommercial_long,
            "noncommercial_short": record.noncommercial_short,
            "noncommercial_spreading": record.noncommercial_spreading,
            "commercial_long": record.commercial_long,
            "commercial_short": record.commercial_short,
            "noncommercial_net": record.noncommercial_net,
            "source": source,
            "updated_at": now,
        },
    )
    db.execute(stmt)


def collect_cftc_gold_position(db: Session) -> CftcPositionRecord:
    record = CftcClient().fetch_current_gold_position()
    upsert_cftc_position(db, record)
    db.commit()
    return record
