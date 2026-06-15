from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.data.fred_client import FRED_SERIES, FredClient, FredSeriesConfig
from app.models import MacroObservation, MacroSeries

logger = logging.getLogger(__name__)


def upsert_series(db: Session, series: FredSeriesConfig) -> MacroSeries:
    existing = db.scalar(select(MacroSeries).where(MacroSeries.series_id == series.series_id))
    if existing:
        existing.name = series.name
        existing.frequency = series.frequency
        existing.unit = series.unit
        existing.source = "FRED"
        existing.updated_at = datetime.now(timezone.utc)
        return existing

    record = MacroSeries(
        series_id=series.series_id,
        name=series.name,
        frequency=series.frequency,
        unit=series.unit,
        source="FRED",
    )
    db.add(record)
    return record


def upsert_observation(db: Session, series_id: str, timestamp, value: float) -> None:
    timestamp_value = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
    stmt = sqlite_insert(MacroObservation).values(
        series_id=series_id,
        timestamp=timestamp_value,
        value=float(value),
        source="FRED",
        updated_at=datetime.now(timezone.utc),
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["series_id", "timestamp"],
        set_={
            "value": float(value),
            "source": "FRED",
            "updated_at": datetime.now(timezone.utc),
        },
    )
    db.execute(stmt)


def collect_fred_data(db: Session, observation_start: str | None = None) -> dict[str, int]:
    """采集 FRED 数据。并行请求所有系列，单点失败不阻塞其他。

    默认增量拉取：每个系列从上次入库日期之后开始，避免重复拉历史数据。
    传 observation_start 可覆盖（用于首次全量回填）。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime as dt, timezone as tz, timedelta
    from sqlalchemy import func
    import time as _t

    _t0 = _t.time()

    # 计算每个系列的增量起始日期
    if observation_start:
        per_series_start = {s.series_id: observation_start for s in FRED_SERIES}
    else:
        # 一次性查询所有系列的最新时间戳
        thirty_days_ago = (dt.now(tz.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        latest_map: dict[str, str] = {}
        rows = db.execute(
            select(
                MacroObservation.series_id,
                func.max(MacroObservation.timestamp),
            ).where(
                MacroObservation.series_id.in_([s.series_id for s in FRED_SERIES]),
            ).group_by(MacroObservation.series_id)
        ).all()
        for sid, max_ts in rows:
            if max_ts:
                # 从最后一条数据的前一天开始（留 1 天缓冲区，FRED 偶尔修正历史值）
                buffered = (max_ts.replace(tzinfo=None) - timedelta(days=1)).strftime("%Y-%m-%d")
                latest_map[sid] = buffered
        per_series_start = {
            s.series_id: latest_map.get(s.series_id, thirty_days_ago)
            for s in FRED_SERIES
        }

    # Phase 1: 并行拉取所有系列数据（纯 HTTP，无 DB）
    results: dict[str, tuple[str, list[tuple]]] = {}  # series_id -> (status, [(ts, val), ...])
    from app.data.fred_client import FredSeriesConfig as _FSC

    def _fetch_one(series: _FSC, start_date: str) -> tuple[str, str, list[tuple]]:
        try:
            client = FredClient(timeout=5)
            df = client.get_observations(series.series_id, observation_start=start_date)
            rows = [(row.timestamp, row.value) for row in df.itertuples(index=False)]
            return (series.series_id, "ok", rows)
        except Exception as e:
            logger.warning("FRED series %s failed: %s", series.series_id, str(e)[:100])
            return (series.series_id, str(e)[:100], [])

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch_one, s, per_series_start[s.series_id]): s for s in FRED_SERIES}
        for future in as_completed(futures):
            sid, status, rows = future.result()
            results[sid] = (status, rows)

    logger.debug("FRED phase1 HTTP: %.1fs", _t.time() - _t0)

    # Phase 2: 批量写入 DB（使用原始 sqlite3 连接，绕过 SQLA 开销）
    _p2_start = _t.time()
    counts: dict[str, int] = {}
    failed = 0
    now_utc = datetime.now(timezone.utc)

    for series in FRED_SERIES:
        status, rows = results.get(series.series_id, ("missing", []))
        if status != "ok":
            failed += 1
            counts[series.series_id] = 0
            continue
        upsert_series(db, series)
        counts[series.series_id] = len(rows)

    db.flush()  # flush upsert_series changes first

    # 批量 upsert：gather all rows, single executemany
    all_rows = []
    for series in FRED_SERIES:
        status, rows = results.get(series.series_id, ("missing", []))
        if status != "ok" or not rows:
            continue
        for ts, val in rows:
            ts_val = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            all_rows.append((series.series_id, ts_val, float(val), "FRED", now_utc))

    if all_rows:
        raw_conn = db.connection().connection  # DBAPI connection
        raw_conn.executemany(
            "INSERT INTO macro_observations (series_id, timestamp, value, source, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (series_id, timestamp) DO UPDATE SET "
            "value = excluded.value, source = excluded.source, updated_at = excluded.updated_at",
            all_rows,
        )

    logger.debug("FRED phase2 DB: %.1fs (total: %.1fs)", _t.time() - _p2_start, _t.time() - _t0)

    if failed > 0:
        logger.warning(
            "FRED collection: %d/%d succeeded, %d failed",
            len(FRED_SERIES) - failed, len(FRED_SERIES), failed,
        )

    return counts
