"""实时金价采集器。

数据来源（优先顺序）：
1. 新浪财经 (hf_GC) — COMEX 黄金期货实时报价，约 15 分钟延迟，免费
2. 本地 GoldPrice 表 — 新浪失败时兜底

含内存缓存：同一次进程内 60 秒内不重复请求。
Yahoo Finance 已不可用（中国大陆 IP 被封锁）。
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

GOLD_TICKER = "GC=F"
SINA_SYMBOL = "hf_GC"
CACHE_TTL = 60  # 缓存秒数
INTRADAY_MAX_AGE_HOURS = 24  # 日内快照保留时长

_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()
_cache_time: float = 0.0


def _safe_float(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return round(float(val), 2)


def fetch_gold_history(days: int = 200, ticker: str = GOLD_TICKER) -> list[dict[str, Any]]:
    """下载历史金价日线数据，存入 GoldPrice 表。

    优先尝试 Sina 日线 API（中国大陆可能受限），
    兜底：用 Sina 实时报价更新当日记录。

    返回: 采集到的记录数。
    """
    from datetime import datetime as dt

    # 1) 尝试 Sina 日线 API
    try:
        url = f"https://stock2.finance.sina.com.cn/futures/api/jsonp.php//InnerFuturesNewService.getDailyKLine?symbol={SINA_SYMBOL}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "https://finance.sina.com.cn/",
        }
        s = requests.Session()
        s.headers.update(headers)
        s.get("https://finance.sina.com.cn/futures/quotes/GC.shtml", timeout=10)
        resp = s.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        text = resp.text

        import json
        json_match = re.search(r"\((\[.*\])\)", text, re.DOTALL)
        if json_match:
            raw_data = json.loads(json_match.group(1))
            if raw_data:
                return _store_daily_bars(raw_data, days)
    except Exception:
        pass  # Sina 日线 API 不可用，走兜底

    # 2) 兜底：用实时报价更新当日记录
    return _update_today_from_sina()


def _fetch_from_db() -> dict[str, Any]:
    """从本地 GoldPrice 表获取最新金价（兜底）。"""
    try:
        from app.database import SessionLocal
        from sqlalchemy import select
        from app.models import GoldPrice

        db = SessionLocal()
        try:
            row = db.scalar(
                select(GoldPrice).order_by(GoldPrice.date.desc())
            )
            if row:
                return {
                    "ok": True,
                    "ticker": "DB/GoldPrice",
                    "price": row.close,
                    "previous_close": None,
                    "change": None,
                    "change_pct": None,
                    "day_high": row.high,
                    "day_low": row.low,
                    "volume": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "history_7d": [],
                    "history_24h": [],
                    "source": "database (日频)",
                    "delay": ">1 day",
                    "freshness": "stale",
                }
        finally:
            db.close()
    except Exception:
        pass
    return {"ok": False, "error": "No data available", "ticker": "N/A"}


def _store_daily_bars(raw_data: list[dict], days: int) -> list[dict[str, Any]]:
    """将 Sina 日线数据写入 GoldPrice 表。"""
    from datetime import datetime as dt
    from app.database import SessionLocal
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    from app.models import GoldPrice

    requested_days = max(1, int(days))
    cutoff = dt.now(timezone.utc) - timedelta(days=requested_days)

    db = SessionLocal()
    count = 0
    try:
        for item in raw_data:
            date_str = item.get("d", "")
            try:
                date_val = dt.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if date_val < cutoff:
                continue

            close_val = _safe_float(item.get("c"))
            if close_val is None:
                continue

            stmt = sqlite_insert(GoldPrice).values(
                date=date_val,
                open=_safe_float(item.get("o")),
                high=_safe_float(item.get("h")),
                low=_safe_float(item.get("l")),
                close=close_val,
                source="SINA",
                updated_at=dt.now(timezone.utc),
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["date"],
                set_={
                    "open": _safe_float(item.get("o")),
                    "high": _safe_float(item.get("h")),
                    "low": _safe_float(item.get("l")),
                    "close": close_val,
                    "source": "SINA",
                    "updated_at": dt.now(timezone.utc),
                },
            )
            db.execute(stmt)
            count += 1
        db.commit()
    finally:
        db.close()
    return [{"ok": True, "count": count, "source": "SINA", "days": requested_days}]


def _update_today_from_sina() -> list[dict[str, Any]]:
    """用 Sina 实时报价更新当日 GoldPrice 记录（兜底）。"""
    from datetime import datetime as dt
    from app.database import SessionLocal
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    from app.models import GoldPrice

    result = _fetch_from_sina()
    if not result.get("ok"):
        return [result]

    price = result.get("price")
    if price is None:
        return [{"ok": False, "error": "No price from Sina"}]

    today = dt.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    db = SessionLocal()
    try:
        stmt = sqlite_insert(GoldPrice).values(
            date=today,
            open=result.get("day_open"),
            high=result.get("day_high"),
            low=result.get("day_low"),
            close=price,
            source="SINA",
            updated_at=dt.now(timezone.utc),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["date"],
            set_={
                "open": result.get("day_open"),
                "high": result.get("day_high"),
                "low": result.get("day_low"),
                "close": price,
                "source": "SINA",
                "updated_at": dt.now(timezone.utc),
            },
        )
        db.execute(stmt)
        db.commit()
        return [{"ok": True, "count": 1, "source": "SINA", "days": 1}]
    except Exception as e:
        db.rollback()
        return [{"ok": False, "error": str(e)}]
    finally:
        db.close()


def fetch_gold_intraday(ticker: str = GOLD_TICKER, interval_minutes: int = 5) -> dict[str, Any]:
    """获取日内金价，用于 24 小时走势图。

    从本地 IntradaySnapshot 表读取快照，按间隔聚合为 OHLC。
    不再依赖 Yahoo Finance（中国大陆无法访问）。
    """
    try:
        from datetime import datetime as dt
        from app.database import SessionLocal
        from sqlalchemy import select
        from app.models import IntradaySnapshot

        cutoff = dt.now(timezone.utc) - timedelta(hours=INTRADAY_MAX_AGE_HOURS)
        db = SessionLocal()
        try:
            rows = db.scalars(
                select(IntradaySnapshot)
                .where(IntradaySnapshot.timestamp >= cutoff)
                .order_by(IntradaySnapshot.timestamp.asc())
            ).all()

            if not rows:
                return {"ok": False, "error": "暂无日内快照数据，请等待后台采集", "ticker": ticker}

            df = pd.DataFrame([{
                "timestamp": r.timestamp,
                "close": r.price,
                "high": r.high or r.price,
                "low": r.low or r.price,
                "open": r.price,
            } for r in rows])

            df = df.set_index("timestamp")
            freq = f"{interval_minutes}min"
            agg = df.resample(freq).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }).dropna()

            if agg.empty:
                return {"ok": False, "error": "日内数据不足以生成图表", "ticker": ticker}

            current_price = float(df["close"].iloc[-1])
            points = [
                {
                    "time": idx.strftime("%H:%M"),
                    "open": round(row["open"], 2),
                    "high": round(row["high"], 2),
                    "low": round(row["low"], 2),
                    "close": round(row["close"], 2),
                }
                for idx, row in agg.iterrows()
            ]

            return {
                "ok": True,
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "interval_minutes": interval_minutes,
                "points": points,
                "freshness": "delayed",
            }
        finally:
            db.close()
    except Exception as e:
        return {"ok": False, "error": str(e), "ticker": ticker}


def fetch_gold_price(ticker: str = GOLD_TICKER, use_cache: bool = True) -> dict[str, Any]:
    """获取实时金价，带缓存和兜底。

    优先新浪财经 hf_GC，失败 → 本地数据库兜底。
    """
    global _cache, _cache_time

    # 读缓存
    if use_cache:
        with _cache_lock:
            if _cache and time.time() - _cache_time < CACHE_TTL:
                return _cache

    # 尝试新浪
    result = _fetch_from_sina()
    if not result.get("ok"):
        # 新浪失败，用本地数据库兜底
        result = _fetch_from_db()

    # 缓存时间记录在写入之前，用于 freshness 判断
    age = time.time() - _cache_time
    with _cache_lock:
        _cache = result
        _cache_time = time.time()

    # 打上缓存标记
    if age > CACHE_TTL:
        result["freshness"] = "stale"

    return result


def _fetch_from_sina() -> dict[str, Any]:
    """从新浪财经 hf_GC 接口获取 COMEX 黄金期货实时报价。

    返回字段（hf_GC）：
        0: 开盘价  1: (空)  2: 最新价  3: 当前价
        4: 最高价  5: 最低价  6: 时间  7: 昨收价
        8: 买价    9: 卖价  10: 成交量  11: 持仓量
        12: 日期   13: 名称
    """
    from datetime import datetime as dt

    try:
        url = f"https://hq.sinajs.cn/list={SINA_SYMBOL}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "https://finance.sina.com.cn/",
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        resp.encoding = "gbk"
        text = resp.text.strip()

        if not text or '=""' in text:
            return {"ok": False, "error": "Sina returned empty", "ticker": SINA_SYMBOL}

        # 解析 var hq_str_hf_GC="...";
        parts = text.split('"')
        if len(parts) < 2:
            return {"ok": False, "error": "Sina format error", "ticker": SINA_SYMBOL}

        fields = parts[1].split(",")
        if len(fields) < 8:
            return {"ok": False, "error": "Sina fields insufficient", "ticker": SINA_SYMBOL}

        open_price = _safe_float(fields[0])
        current_price = _safe_float(fields[3]) or _safe_float(fields[2])
        day_high = _safe_float(fields[4])
        day_low = _safe_float(fields[5])
        prev_close = _safe_float(fields[7]) or _safe_float(fields[1])

        change = None
        change_pct = None
        if current_price is not None and prev_close is not None and prev_close > 0:
            change = round(current_price - prev_close, 2)
            change_pct = round(change / prev_close * 100, 2)

        # 近 7 天日线从数据库取
        history_7d: list[dict] = []
        try:
            from app.database import SessionLocal
            from app.models import GoldPrice
            from sqlalchemy import select

            local_db = SessionLocal()
            try:
                rows = local_db.scalars(
                    select(GoldPrice).order_by(GoldPrice.date.desc()).limit(7)
                ).all()
                for r in reversed(rows):
                    history_7d.append({
                        "date": r.date.strftime("%Y-%m-%d") if hasattr(r.date, "strftime") else str(r.date),
                        "open": r.open,
                        "high": r.high,
                        "low": r.low,
                        "close": r.close,
                    })
            finally:
                local_db.close()
        except Exception:
            pass

        return {
            "ok": True,
            "ticker": SINA_SYMBOL,
            "price": current_price,
            "previous_close": prev_close,
            "change": change,
            "change_pct": change_pct,
            "day_open": open_price,
            "day_high": day_high,
            "day_low": day_low,
            "volume": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "history_7d": history_7d,
            "history_24h": [],
            "source": "Sina Finance",
            "delay": "~15 min",
            "freshness": "delayed",
        }
    except Exception as e:
        return {
            "ok": False,
            "ticker": SINA_SYMBOL,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def record_intraday_snapshot() -> dict[str, Any]:
    """记录一次日内金价快照到 IntradaySnapshot 表。

    从新浪获取当前价格，写入快照表。清理超过 24 小时的旧快照。
    应被后台线程周期性调用。
    """
    try:
        result = _fetch_from_sina()
        if not result.get("ok"):
            return result

        price = result.get("price")
        if price is None:
            return {"ok": False, "error": "No price from Sina"}

        high = result.get("day_high")
        low = result.get("day_low")

        from app.database import SessionLocal
        from app.models import IntradaySnapshot

        db = SessionLocal()
        try:
            now = datetime.now(timezone.utc)
            snap = IntradaySnapshot(
                timestamp=now,
                price=float(price),
                high=float(price),  # 只用当前价，聚合时会算出真实 high
                low=float(price),   # 同上
                source="SINA",
                updated_at=now,
            )
            db.add(snap)

            # 清理超过 24 小时的旧快照
            cutoff = now - timedelta(hours=INTRADAY_MAX_AGE_HOURS)
            from sqlalchemy import delete
            db.execute(delete(IntradaySnapshot).where(IntradaySnapshot.timestamp < cutoff))

            db.commit()
            return {"ok": True, "recorded_at": now.isoformat(), "price": price}
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── 后台日内记录线程 ──────────────────────────────────────────────

_recorder_thread: threading.Thread | None = None
_recorder_stop = threading.Event()
RECORD_INTERVAL_SECONDS = 60  # 每 60 秒记录一次


def _intraday_recorder_loop():
    """后台线程：周期性记录金价快照。"""
    while not _recorder_stop.is_set():
        try:
            # 仅在美国期货交易时段活跃（避免非交易时段记录无用数据）
            # COMEX: 周日 18:00 - 周五 17:00 ET = UTC-4/5
            now = datetime.now(timezone.utc)
            weekday = now.weekday()  # 0=Mon, 6=Sun
            hour = now.hour

            # COMEX 活跃时段：周日 22:00 UTC - 周五 21:00 UTC
            # 简化：只在周六全天跳过，周日 22:00 前也跳过
            if weekday == 5:  # 周六 → 跳过
                pass
            else:
                record_intraday_snapshot()
        except Exception:
            pass  # 静默处理，不中断线程
        _recorder_stop.wait(RECORD_INTERVAL_SECONDS)


def start_intraday_recorder():
    """启动后台日内快照记录线程。"""
    global _recorder_thread, _recorder_stop
    if _recorder_thread is not None and _recorder_thread.is_alive():
        return
    _recorder_stop.clear()
    _recorder_thread = threading.Thread(target=_intraday_recorder_loop, daemon=True)
    _recorder_thread.start()


def stop_intraday_recorder():
    """停止后台日内快照记录线程。"""
    global _recorder_thread, _recorder_stop
    _recorder_stop.set()
    if _recorder_thread is not None:
        _recorder_thread.join(timeout=5)
        _recorder_thread = None
