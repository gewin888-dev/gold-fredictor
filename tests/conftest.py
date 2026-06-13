from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import CftcPosition, GoldPrice, MacroObservation


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def insert_required_macro_observations(db, days: int = 80, end_at: Optional[datetime] = None) -> None:
    if end_at is None:
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    else:
        start = end_at - timedelta(days=days - 1)
    series_values = {
        "GOLDAMGBD228NLBM": lambda i: 2400 + i * 2.0,
        "DGS10": lambda i: 4.2 - i * 0.002,
        "DFII10": lambda i: 2.2 - i * 0.005,
        "T10YIE": lambda i: 2.1 + i * 0.002,
        "FEDFUNDS": lambda i: 5.3 - i * 0.001,
        "VIXCLS": lambda i: 14 + i * 0.04,
        "DTWEXBGS": lambda i: 120 - i * 0.05,
    }
    for series_id, value_fn in series_values.items():
        for index in range(days):
            db.add(
                MacroObservation(
                    series_id=series_id,
                    timestamp=start + timedelta(days=index),
                    value=float(value_fn(index)),
                    source="TEST",
                )
            )
    db.commit()


def insert_gold_prices(db, days: int = 80, end_at: Optional[datetime] = None) -> None:
    if end_at is None:
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    else:
        start = end_at - timedelta(days=days - 1)
    for index in range(days):
        close = 2400 + index * 2.0
        db.add(
            GoldPrice(
                date=start + timedelta(days=index),
                open=close - 3,
                high=close + 8,
                low=close - 8,
                close=float(close),
                source="TEST",
            )
        )
    db.commit()


def insert_cftc_position(db, timestamp: Optional[datetime] = None) -> None:
    report_time = timestamp or datetime(2025, 3, 21, tzinfo=timezone.utc)
    db.add(
        CftcPosition(
            market_name="GOLD - COMMODITY EXCHANGE INC.",
            contract_market_code="088691",
            exchange_code="CMX",
            timestamp=report_time,
            open_interest=300000,
            noncommercial_long=180000,
            noncommercial_short=50000,
            noncommercial_spreading=15000,
            commercial_long=60000,
            commercial_short=210000,
            noncommercial_net=130000,
            source="TEST",
        )
    )
    db.commit()
