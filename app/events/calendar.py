from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.models import MacroEvent


@dataclass(frozen=True)
class MacroEventInput:
    event_id: str
    timestamp: datetime
    name: str
    country: str
    importance: str
    description: str
    source: str = "MANUAL"


def upsert_macro_event(db: Session, event: MacroEventInput) -> None:
    stmt = sqlite_insert(MacroEvent).values(
        event_id=event.event_id,
        timestamp=event.timestamp,
        name=event.name,
        country=event.country,
        importance=event.importance,
        description=event.description,
        source=event.source,
        updated_at=datetime.now(timezone.utc),
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["event_id"],
        set_={
            "timestamp": event.timestamp,
            "name": event.name,
            "country": event.country,
            "importance": event.importance,
            "description": event.description,
            "source": event.source,
            "updated_at": datetime.now(timezone.utc),
        },
    )
    db.execute(stmt)


def list_macro_events(db: Session, days_ahead: int = 30) -> list[MacroEvent]:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    end = now + timedelta(days=days_ahead)
    return db.scalars(
        select(MacroEvent)
        .where(MacroEvent.timestamp >= now, MacroEvent.timestamp <= end)
        .order_by(MacroEvent.timestamp.asc())
    ).all()


def load_sample_macro_events(db: Session, base_time: datetime | None = None) -> int:
    base = base_time or datetime.now(timezone.utc)
    now = datetime.now(timezone.utc)
    if base < now:
        base = now
    base = base.replace(hour=12, minute=30, second=0, microsecond=0)
    events = [
        MacroEventInput(
            event_id=f"sample-cpi-{base.date()}",
            timestamp=base + timedelta(days=5),
            name="美国 CPI 数据",
            country="US",
            importance="high",
            description="通胀数据可能影响实际利率、美元和黄金波动。",
            source="SAMPLE",
        ),
        MacroEventInput(
            event_id=f"sample-fomc-{base.date()}",
            timestamp=base + timedelta(days=14),
            name="FOMC 利率决议",
            country="US",
            importance="high",
            description="美联储政策路径会影响实际利率和黄金定价。",
            source="SAMPLE",
        ),
        MacroEventInput(
            event_id=f"sample-pce-{base.date()}",
            timestamp=base + timedelta(days=24),
            name="美国 PCE 通胀",
            country="US",
            importance="medium",
            description="PCE 是美联储关注的通胀指标。",
            source="SAMPLE",
        ),
        MacroEventInput(
            event_id=f"sample-nfp-{base.date()}",
            timestamp=base + timedelta(days=32),
            name="美国非农就业",
            country="US",
            importance="high",
            description="就业数据可能影响降息预期、美元和避险情绪。",
            source="SAMPLE",
        ),
    ]
    for event in events:
        upsert_macro_event(db, event)
    db.commit()
    return len(events)
