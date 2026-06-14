"""黄金走势实时监控仪表盘"""

from __future__ import annotations

import json
import html
import sys
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import SessionLocal, init_db
from app.config import get_settings
from app.backtesting.score_backtest import run_score_backtest
from app.events.calendar import list_macro_events
from app.models import CftcPosition, ExternalMarketIndicator, GoldScoreSnapshot, MacroObservation, MacroSeries
from app.monitoring.health import get_data_health
from app.scoring.gold_score import compute_and_store_gold_score
from app.scoring.factor_registry import factor_groups as registry_factor_groups
from app.scoring.factor_registry import factor_help as registry_factor_help
from app.scoring.factor_registry import inactive_factor_reasons as registry_inactive_reasons

import httpx

st.set_page_config(page_title="黄金走势监控", layout="wide", initial_sidebar_state="collapsed")
SETTINGS = get_settings()
LOW_CONFIDENCE_SOURCES = {"SAMPLE", "ESTIMATE", "MANUAL", "JSON"}

st.markdown("""
<style>
@keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.35 } }
:root {
  --gold: #c9972b;
  --ink: #18212f;
  --body: #334155;
  --muted: #64748b;
  --line: #e5e7eb;
  --panel: #f8fafc;
  --paper: #ffffff;
}
.stApp,
div[data-testid="stAppViewContainer"] {
  background: #f6f8fb;
  color: var(--body);
}
div[data-testid="stHeader"] {
  background: rgba(246, 248, 251, 0.86);
}
div[data-testid="stSidebar"] {
  background: #ffffff;
  border-right: 1px solid var(--line);
}
.stMarkdown,
.stMarkdown p,
.stCaption,
label,
span,
div {
  color: inherit;
}
p,
li,
td,
th {
  color: var(--body);
}
.block-container {
  padding-top: 1.4rem;
  padding-bottom: 2.5rem;
  max-width: 1380px;
}
h1, h2, h3 {
  letter-spacing: 0;
  color: var(--ink);
}
div[data-testid="stMetric"] {
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 14px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
div[data-testid="stMetric"] > div[data-testid="stMetricDelta"] {
  font-size: 0.85rem;
}
div[data-testid="stMetricLabel"] {
  color: var(--muted);
  font-size: 0.78rem;
}
div[data-testid="stMetricValue"] {
  font-size: 1.15rem;
  color: var(--ink);
}
.hero {
  padding: 18px 20px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: linear-gradient(135deg, #fffdf7 0%, #ffffff 48%, #f8fafc 100%);
  margin-bottom: 14px;
}
.hero-title {
  font-size: 1.55rem;
  font-weight: 700;
  color: var(--ink);
  margin-bottom: 4px;
}
.hero-subtitle {
  color: var(--muted);
  font-size: 0.95rem;
}
.status-pill {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.82rem;
  font-weight: 600;
  border: 1px solid var(--line);
  background: #ffffff;
}
.status-ok { color: #047857; border-color: #bbf7d0; background: #f0fdf4; }
.status-warn { color: #b45309; border-color: #fde68a; background: #fffbeb; }
.status-error { color: #b91c1c; border-color: #fecaca; background: #fef2f2; }
.live-dot {
  display:inline-block;
  width:8px;
  height:8px;
  border-radius:50%;
  background:#10b981;
  animation:pulse 2s infinite;
  margin-right:6px;
}
.section-label {
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin: 18px 0 6px 0;
}
.time-strip {
  display: grid;
  grid-template-columns: repeat(6, auto);
  gap: 8px;
  align-items: stretch;
  margin: -2px 0 16px 0;
  justify-content: start;
}
.time-box {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px 12px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.time-box-compact {
  padding: 8px 10px;
}
.time-box-muted {
  background: #ffffff;
  border: 1px solid var(--line);
}
.time-label {
  color: var(--muted);
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0;
  margin-bottom: 4px;
}
.time-value {
  color: var(--ink);
  font-size: 1.18rem;
  font-weight: 750;
  font-family: "SF Mono", "Menlo", "Consolas", monospace;
  font-variant-numeric: tabular-nums;
}
.time-date {
  color: var(--muted);
  font-size: 0.78rem;
  margin-top: 2px;
}
.time-box-muted .time-label { color: var(--muted); }
.time-box-muted .time-value { color: var(--ink); }
.time-box-muted .time-date { color: var(--muted); }
.muted-data {
  color: #94a3b8;
  filter: grayscale(0.75);
  opacity: 0.55;
}
.production-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid #bbf7d0;
  background: #f0fdf4;
  color: #047857;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.78rem;
  font-weight: 700;
}
.low-reliability {
  opacity: 0.48;
  filter: grayscale(0.7);
}
.factor-card {
  position: relative;
  min-height: 66px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 9px 10px;
  background: #ffffff;
  color: var(--ink);
  display: flex;
  flex-direction: column;
  justify-content: center;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
  cursor: help;
}
.factor-card:hover {
  border-color: #cbd5e1;
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.10);
}
.factor-card::after {
  content: attr(data-tooltip);
  display: none;
  position: absolute;
  left: 0;
  top: calc(100% + 8px);
  z-index: 30;
  width: min(320px, 70vw);
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid #d7dde6;
  background: #ffffff;
  color: #334155;
  font-size: 0.76rem;
  line-height: 1.38;
  font-weight: 500;
  white-space: normal;
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.16);
}
.factor-card:hover::after {
  display: block;
}
.factor-card-title {
  font-size: 0.76rem;
  font-weight: 700;
  color: #64748b;
  margin-bottom: 5px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.factor-card-value {
  font-size: 1.02rem;
  font-weight: 750;
  color: var(--ink);
  margin-bottom: 0;
}
.factor-card-positive .factor-card-value { color: #047857; }
.factor-card-negative .factor-card-value { color: #b91c1c; }
.factor-card-neutral .factor-card-value { color: #64748b; }
.factor-card-inactive {
  background: #f8fafc;
  filter: grayscale(0.7);
  opacity: 0.72;
}
.factor-card-inactive .factor-card-value {
  color: #94a3b8;
}
div[data-testid="stExpander"] {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 8px;
}
div[data-testid="stExpander"] summary p {
  color: var(--ink);
  font-weight: 650;
}
button[kind="secondary"],
div[data-testid="stButton"] button {
  background: #ffffff;
  color: var(--ink);
  border: 1px solid #d7dde6;
}
button[kind="secondary"]:hover,
div[data-testid="stButton"] button:hover {
  border-color: var(--gold);
  color: #8a6517;
}
div[data-testid="stDataFrame"],
div[data-testid="stTable"] {
  background: #ffffff;
}
@media (max-width: 760px) {
  .time-strip { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)

# ═══ 缓存 ═══

UTC_TZ = dt.timezone.utc
BEIJING_TZ = dt.timezone(dt.timedelta(hours=8))

PLOTLY_LIGHT_LAYOUT = {
    "plot_bgcolor": "#ffffff",
    "paper_bgcolor": "#ffffff",
    "font": {"color": "#334155", "size": 12},
    "hoverlabel": {
        "bgcolor": "#ffffff",
        "bordercolor": "#cbd5e1",
        "font": {"color": "#18212f", "size": 12},
    },
}


def _safe_float(val: str) -> float | None:
    try:
        return round(float(val), 2)
    except (ValueError, TypeError):
        return None


def _ago(ts_str: str) -> str:
    """返回相对时间：刚刚 / 3分钟前 / 1小时前"""
    if not ts_str:
        return "—"
    try:
        ts = dt.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC_TZ)
        secs = (dt.datetime.now(UTC_TZ) - ts.astimezone(UTC_TZ)).total_seconds()
        if secs < 60: return ts.astimezone(BEIJING_TZ).strftime("%H:%M:%S")
        if secs < 3600: return f"{int(secs/60)}分钟前"
        if secs < 86400: return f"{int(secs/3600)}小时前"
        return f"{int(secs/86400)}天前"
    except Exception:
        return ts_str[:16]


def api(path: str, method: str = "get", **kw) -> dict:
    """调用 FastAPI 端点"""
    try:
        with httpx.Client(timeout=httpx.Timeout(15)) as c:
            fn = getattr(c, method.lower())
            base = "http://localhost:8000"
            r = fn(f"{base}{path}", **kw)
            return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


@st.cache_data(ttl=25)
def get_gold() -> dict:
    try:
        with httpx.Client(timeout=httpx.Timeout(10)) as c:
            r = c.get("http://localhost:8000/gold/price")
            return r.json() if r.status_code == 200 else {}
    except Exception:
        db = SessionLocal()
        try:
            from app.models import GoldPrice as GP
            row = db.scalar(select(GP).order_by(GP.date.desc()))
            if not row:
                return {}
            return {
                "ok": True, "price": row.close,
                "previous_close": None, "change": None, "change_pct": None,
                "day_high": row.high, "day_low": row.low,
                "timestamp": row.date.isoformat() if row.date else "",
                "source": row.source, "freshness": "stale",
            }
        finally:
            db.close()


@st.cache_data(ttl=25)
def get_shanghai_gold() -> dict:
    """获取上海黄金期货（沪金连续）实时价格。nf_AU0 字段：
       0=名称 1=成交量 2=开盘 3=最高 4=最低 5=最新价
    """
    import requests as req
    try:
        url = "https://hq.sinajs.cn/list=nf_AU0"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}
        r = req.get(url, headers=headers, timeout=8)
        r.encoding = "gbk"
        text = r.text.strip()
        parts = text.split('"')
        if len(parts) < 2: return {}
        fields = parts[1].split(",")
        if len(fields) < 8: return {}
        price = _safe_float(fields[5])  # 最新价（停盘时为0）
        high = _safe_float(fields[3])   # 最高价
        low = _safe_float(fields[4])    # 最低价
        # 停盘时用结算价兜底
        if price == 0 or price is None:
            settle = _safe_float(fields[6])
            if settle and settle > 0:
                price = settle
        if price is None: return {}
        return {"ok": True, "name": fields[0], "price": price, "high": high, "low": low}
    except Exception:
        return {}


@st.cache_data(ttl=300)
def get_shanghai_daily() -> pd.DataFrame:
    """获取沪金连续日线K线数据（新浪财经期货）。字段：d=日期 o=开 h=高 l=低 c=收。"""
    import json
    import requests as req
    try:
        url = "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_AU0=/InnerFuturesNewService.getDailyKLine?symbol=AU0"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}
        r = req.get(url, headers=headers, timeout=15)
        text = r.text
        start = text.find("(") + 1
        end = text.rfind(")")
        if start > 0 and end > start:
            data = json.loads(text[start:end])
            return pd.DataFrame(data)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120)
def get_shanghai_intraday() -> pd.DataFrame:
    """获取沪金连续日内5分钟K线（新浪财经，仅当日）。"""
    import json
    import requests as req
    try:
        url = "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_AU0=/InnerFuturesNewService.getFewMinLine?symbol=AU0&type=5"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}
        r = req.get(url, headers=headers, timeout=15)
        text = r.text
        start = text.find("(") + 1
        end = text.rfind(")")
        if start > 0 and end > start:
            data = json.loads(text[start:end])
            df = pd.DataFrame(data)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["d"])
                cutoff = pd.Timestamp.now() - pd.Timedelta(hours=36)
                df = df[df["timestamp"] >= cutoff]
                df = df[df["timestamp"] <= pd.Timestamp.now()]
                df["close"] = df["c"].astype(float)
                return df.sort_values("timestamp")
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=55)
def get_scores() -> pd.DataFrame:
    db = SessionLocal()
    try:
        rows = db.scalars(select(GoldScoreSnapshot).order_by(GoldScoreSnapshot.timestamp.asc())).all()
        return pd.DataFrame([{
            "时间": r.timestamp, "评分": r.total_score, "方向": r.direction,
            "因子": r.factor_scores, "风险": r.risk_flags, "说明": r.summary,
        } for r in rows])
    finally:
        db.close()


@st.cache_data(ttl=55)
def get_macro() -> pd.DataFrame:
    db = SessionLocal()
    try:
        rows = db.scalars(select(MacroObservation).order_by(MacroObservation.timestamp.asc())).all()
        meta = db.scalars(select(MacroSeries)).all()
        meta_map = {m.series_id: m.name for m in meta}
        return pd.DataFrame([{
            "时间": r.timestamp, "指标": meta_map.get(r.series_id, r.series_id),
            "值": r.value, "series_id": r.series_id,
        } for r in rows])
    finally:
        db.close()


@st.cache_data(ttl=55)
def get_external_indicators() -> pd.DataFrame:
    db = SessionLocal()
    try:
        rows = db.scalars(
            select(ExternalMarketIndicator)
            .order_by(ExternalMarketIndicator.timestamp.desc(), ExternalMarketIndicator.indicator_id.asc())
            .limit(200)
        ).all()
        return pd.DataFrame([{
            "时间": r.timestamp,
            "指标ID": r.indicator_id,
            "指标": r.name,
            "类别": r.category,
            "值": r.value,
            "单位": r.unit,
            "来源": r.source,
            "备注": r.note,
        } for r in rows])
    finally:
        db.close()


@st.cache_data(ttl=60)
def get_cftc() -> pd.DataFrame:
    db = SessionLocal()
    try:
        rows = db.scalars(select(CftcPosition).order_by(CftcPosition.timestamp.asc())).all()
        return pd.DataFrame([{
            "时间": r.timestamp, "净多": r.noncommercial_net,
            "多": r.noncommercial_long, "空": r.noncommercial_short,
        } for r in rows])
    finally:
        db.close()


@st.cache_data(ttl=40)
def get_health() -> dict:
    db = SessionLocal()
    try:
        return get_data_health(db)
    finally:
        db.close()


@st.cache_data(ttl=60)
def get_bt(h: int = 20) -> dict:
    db = SessionLocal()
    try:
        return run_score_backtest(db, horizon_days=h)
    finally:
        db.close()


@st.cache_data(ttl=90)
def get_events(days_ahead: int = 60) -> pd.DataFrame:
    db = SessionLocal()
    try:
        events = list_macro_events(db, days_ahead=days_ahead)
        if not events:
            return pd.DataFrame()
        return pd.DataFrame([{
            "时间": e.timestamp, "事件": e.name,
            "国家": e.country, "重要性": e.importance, "描述": e.description,
        } for e in events])
    finally:
        db.close()


@st.cache_data(ttl=55)
def get_premium() -> pd.DataFrame:
    db = SessionLocal()
    try:
        from app.models import ChinaGoldPremium
        rows = db.scalars(select(ChinaGoldPremium).order_by(ChinaGoldPremium.timestamp.desc()).limit(30)).all()
        return pd.DataFrame([{
            "时间": r.timestamp, "溢价": r.premium_pct,
            "SGE价格": r.sge_price_cny, "国际价": r.lbma_price_usd,
            "来源": r.source,
        } for r in rows])
    finally:
        db.close()


@st.cache_data(ttl=60)
def get_intraday() -> pd.DataFrame:
    """获取COMEX日内金价完整时间戳（直接查数据库，保留日期）。"""
    from datetime import datetime, timedelta, timezone as tz
    from app.models import IntradaySnapshot
    db = SessionLocal()
    try:
        cutoff = datetime.now(tz.utc) - timedelta(hours=25)
        rows = db.scalars(
            select(IntradaySnapshot)
            .where(IntradaySnapshot.timestamp >= cutoff)
            .order_by(IntradaySnapshot.timestamp.asc())
        ).all()
        return pd.DataFrame([{
            "timestamp": r.timestamp,
            "close": r.price,
        } for r in rows])
    finally:
        db.close()


@st.cache_data(ttl=55)
def get_cb_gold() -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (Global月度汇总, 国别月度明细)"""
    db = SessionLocal()
    try:
        from app.models import CentralBankGold
        global_rows = db.scalars(
            select(CentralBankGold)
            .where(CentralBankGold.country == "Global")
            .order_by(CentralBankGold.timestamp.asc())
        ).all()
        global_df = pd.DataFrame([{
            "月份": row.period, "净购金(吨)": row.net_change_tonnes,
            "来源": row.source,
        } for row in global_rows])
        country_rows = db.scalars(
            select(CentralBankGold)
            .where(CentralBankGold.country != "Global")
            .order_by(CentralBankGold.timestamp.desc())
        ).all()
        country_df = pd.DataFrame([{
            "国家": row.country, "月份": row.period, "购金(吨)": row.net_change_tonnes,
            "来源": row.source,
        } for row in country_rows])
        return global_df, country_df
    finally:
        db.close()


@st.cache_data(ttl=55)
def get_sentiment() -> tuple[float | None, pd.DataFrame]:
    db = SessionLocal()
    try:
        from app.models import NewsSentiment
        rows = db.scalars(
            select(NewsSentiment)
            .where(NewsSentiment.source.in_(["GDELT", "NEWSAPI"]))
            .order_by(NewsSentiment.timestamp.desc())
            .limit(50)
        ).all()
        df = pd.DataFrame([{
            "时间": r.timestamp, "情绪": r.sentiment_score, "标题": r.title or "",
            "来源": r.source_url or "",
            "数据源": r.source,
        } for r in rows])
        latest_score = float(rows[0].sentiment_score) if rows else None
        return latest_score, df
    finally:
        db.close()


# ═══ 页面渲染 ═══

# 自动刷新频率固定15秒
if "_rf_interval" not in st.session_state:
    st.session_state["_rf_interval"] = 15

def _now_utc():
    return dt.datetime.now(UTC_TZ)

def _now_beijing():
    return dt.datetime.now(BEIJING_TZ)

def _is_low_confidence_source(source: object) -> bool:
    return str(source or "").upper() in LOW_CONFIDENCE_SOURCES


def _finite_chart_frame(df: pd.DataFrame, required_cols: list[str], numeric_cols: list[str]) -> pd.DataFrame:
    """Return rows that Altair can safely render without infinite extents."""
    if df.empty:
        return df.copy()
    out = df.dropna(subset=required_cols).copy()
    if "时间" in out.columns:
        out["时间"] = pd.to_datetime(out["时间"], errors="coerce", utc=True).dt.tz_convert(None)
        out = out.dropna(subset=["时间"])
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out[np.isfinite(out[col])]
    return out


def _intraday_coverage_label(df: pd.DataFrame) -> tuple[str, list[str]]:
    if df.empty or "timestamp" not in df.columns:
        return "暂无日内数据", ["后台尚未记录到可绘制的日内金价快照。"]
    clean = df.dropna(subset=["timestamp", "close"]).copy()
    clean["close"] = pd.to_numeric(clean["close"], errors="coerce")
    clean = clean[np.isfinite(clean["close"])]
    if clean.empty:
        return "暂无有效日内数据", ["日内快照存在，但价格字段不可绘制。"]
    start = pd.to_datetime(clean["timestamp"].min())
    end = pd.to_datetime(clean["timestamp"].max())
    coverage_minutes = max(0.0, (end - start).total_seconds() / 60)
    if coverage_minutes >= 23 * 60:
        label = "24小时金价走势"
    elif coverage_minutes >= 60:
        label = f"近 {coverage_minutes / 60:.1f} 小时金价走势"
    else:
        label = f"近 {coverage_minutes:.0f} 分钟金价走势"
    notes: list[str] = []
    if coverage_minutes < 60:
        notes.append("日内快照覆盖不足 1 小时，暂不能代表 24 小时走势。")
    elif coverage_minutes < 23 * 60:
        notes.append("日内快照尚未覆盖完整 24 小时，图表按当前可用区间展示。")
    if clean["close"].nunique() <= 1:
        notes.append("COMEX 报价在当前快照区间内没有变化，可能处于休市或报价源冻结。")
    return label, notes

st.markdown(
    """<div class="hero">
      <div class="hero-title">黄金走势实时监控与预测系统</div>
      <div class="hero-subtitle">宏观利率、美元、持仓、事件和情绪的综合监控面板。仅用于数据分析和风险提示，不构成投资建议。</div>
    </div>""",
    unsafe_allow_html=True,
)

# ═══ 侧边栏 ═══

with st.sidebar:
    st.markdown("### ⚙️ 自动优化")
    auto_config = api("/settings/auto-optimize")
    auto_settings = auto_config.get("settings", {})
    new_score = st.toggle("评分参数自动搜索", value=auto_settings.get("AUTO_OPTIMIZE_SCORE_PARAMS", False))
    new_activate = st.toggle("达标自动激活", value=auto_settings.get("AUTO_ACTIVATE_OPTIMIZED_PARAMS", False))
    new_pred = st.toggle("预测模型自动搜索", value=auto_settings.get("AUTO_OPTIMIZE_PREDICTION_MODEL", False))
    new_pred_activate = st.toggle("预测模型自动激活", value=auto_settings.get("AUTO_ACTIVATE_PREDICTION_MODEL", False))
    if st.button("💾 保存", use_container_width=True):
        r = api("/settings/auto-optimize", "post", json={
            "AUTO_OPTIMIZE_SCORE_PARAMS": new_score,
            "AUTO_ACTIVATE_OPTIMIZED_PARAMS": new_activate,
            "AUTO_OPTIMIZE_PREDICTION_MODEL": new_pred,
            "AUTO_ACTIVATE_PREDICTION_MODEL": new_pred_activate,
        })
        if r.get("ok"):
            st.success("已保存")
            st.rerun()
    st.divider()
    saved = st.session_state.get("_auto_saved", auto_settings)
    st.caption("⚠️ 达标自动激活已开启" if saved.get("AUTO_ACTIVATE_OPTIMIZED_PARAMS") else "仅生成候选，需手动激活")


with st.expander("模型健康", expanded=False):
    health_payload = auto_config.get("health", {}) if isinstance(auto_config, dict) else {}
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("评分参数版本", health_payload.get("score_params_version", "default"))
    h2.metric("预测模型版本", health_payload.get("prediction_model_version") or "—")
    h3.metric("已评估预测", health_payload.get("prediction_evaluated_count", 0))
    h4.metric("到期未评估", health_payload.get("prediction_due_pending_count", 0))

    h5, h6, h7, h8 = st.columns(4)
    h5.metric("评分样本条件", "满足" if health_payload.get("score_sample_ready") else "不足")
    h6.metric("预测样本条件", "满足" if health_payload.get("prediction_sample_ready") else "不足")
    h7.metric("自动搜索", "开" if (
        auto_settings.get("AUTO_OPTIMIZE_SCORE_PARAMS") or auto_settings.get("AUTO_OPTIMIZE_PREDICTION_MODEL")
    ) else "关")
    h8.metric("自动激活", "开" if (
        auto_settings.get("AUTO_ACTIVATE_OPTIMIZED_PARAMS") or auto_settings.get("AUTO_ACTIVATE_PREDICTION_MODEL")
    ) else "关")

    latest_score_candidate = health_payload.get("latest_score_candidate") or {}
    latest_prediction_candidate = health_payload.get("latest_prediction_candidate") or {}
    c_left, c_right = st.columns(2)
    with c_left:
        st.caption(
            "最近评分候选："
            f"{latest_score_candidate.get('version', '—')} · "
            f"命中率 {latest_score_candidate.get('hit_rate', '—')} · "
            f"样本 {latest_score_candidate.get('sample_count', '—')}"
        )
    with c_right:
        st.caption(
            "最近预测候选："
            f"{latest_prediction_candidate.get('version', '—')} · "
            f"方向准确率 {latest_prediction_candidate.get('direction_accuracy', '—')} · "
            f"MAPE {latest_prediction_candidate.get('mape_price_pct', '—')}"
        )
    for reason in health_payload.get("reasons", []):
        st.caption(f"• {reason}")

# ═══════════════════════════════════════════
# 时间条 + 金价卡片
# ═══════════════════════════════════════════

health = get_health()
health_status = health.get("status", "unknown")
health_label = {"ok": "正常", "warn": "延迟", "error": "异常"}.get(health_status, health_status)
health_color = {"ok": "#047857", "warn": "#b45309", "error": "#b91c1c"}.get(health_status, "#64748b")

gold = get_gold()

# 时间条（时钟 + 数据源/更新/状态）—— 用 Streamlit 列对齐下面卡片
utc_now = _now_utc()
beijing_now = _now_beijing()
tc1, tc2, tc3, tc4, tc5, tc6 = st.columns(6)
with tc1:
    st.markdown(
        f'<div class="time-box time-box-muted time-box-compact">'
        f'<div class="time-value"><span style="font-size:0.65rem;color:#94a3b8;">北京 </span>{beijing_now.strftime("%H:%M")}</div>'
        f'<div style="color:#94a3b8;font-size:0.74rem;">{beijing_now.strftime("%m月%d日")}</div></div>',
        unsafe_allow_html=True)
with tc2:
    st.markdown(
        f'<div class="time-box time-box-muted time-box-compact">'
        f'<div class="time-value"><span style="font-size:0.65rem;color:#94a3b8;">UTC </span>{utc_now.strftime("%H:%M")}</div>'
        f'<div style="color:#94a3b8;font-size:0.74rem;">{utc_now.strftime("%m月%d日")}</div></div>',
        unsafe_allow_html=True)
with tc3:
    st.markdown(
        f'<div class="time-box time-box-muted time-box-compact">'
        f'<div class="time-label">自动刷新</div>'
        f'<div class="time-value">{st.session_state["_rf_interval"]}s</div></div>',
        unsafe_allow_html=True)
with tc4:
    src = gold.get("source", "—") if gold.get("ok") else "—"
    st.markdown(
        f'<div class="time-box time-box-muted time-box-compact">'
        f'<div class="time-label">数据源</div>'
        f'<div class="time-value">{src}</div></div>',
        unsafe_allow_html=True)
with tc5:
    upd = _ago(gold.get("timestamp", "")) if gold.get("ok") else "—"
    st.markdown(
        f'<div class="time-box time-box-muted time-box-compact">'
        f'<div class="time-label">更新</div>'
        f'<div class="time-value">{upd}</div></div>',
        unsafe_allow_html=True)
with tc6:
    st.markdown(
        f'<div class="time-box time-box-muted time-box-compact">'
        f'<div class="time-label">状态</div>'
        f'<div class="time-value" style="color:{health_color};font-weight:600;">{health_label}</div></div>',
        unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

if gold.get("ok") and gold.get("price"):
    fr = gold.get("freshness", "stale")
    fr_icon = {"live": "🟢", "delayed": "🟡", "stale": "🔴"}.get(fr, "")
    chg = gold.get("change")
    chg_pct = gold.get("change_pct")
    delta = f"{chg:+.2f} ({chg_pct:+.2f}%)" if chg is not None else None

    g1, g2, g3, g4, g5, g6 = st.columns(6)
    g1.metric(f"{fr_icon} COMEX 金价", f"${gold['price']:,.2f}", delta=delta)
    g2.metric("今日最高", f"${gold.get('day_high'):,.0f}" if gold.get('day_high') else "—")
    g3.metric("今日最低", f"${gold.get('day_low'):,.0f}" if gold.get('day_low') else "—")

    # 沪金连续（带交易状态，红停绿交）
    sh = get_shanghai_gold()
    if sh.get("ok"):
        bj_now = _now_beijing()
        h, m = bj_now.hour, bj_now.minute
        t = h * 60 + m
        in_day = (540 <= t <= 690) or (810 <= t <= 900)
        in_night = t >= 1260 or t <= 150
        if in_day or in_night:
            trading = True
            status_zh = "交易中"
        else:
            trading = False
            status_zh = "停盘中"
        status_color_hex = "#047857" if trading else "#b91c1c"
        status_icon = "🟢" if trading else "🔴"
        g4.metric(
            "沪金连续",
            f"¥{sh['price']:,.2f}/g",
            delta=f"{'🟢' if trading else '🔴'} {status_zh}  ·  高{sh['high']:,.0f} 低{sh['low']:,.0f}",
            delta_color="off",
        )
    else:
        g4.metric("沪金连续", "—")

    prem_df_for_metric = get_premium()
    premium_value = "—"
    premium_delta = None
    premium_src = prem_df_for_metric.iloc[0].get("来源", "") if not prem_df_for_metric.empty else ""
    if (
        not prem_df_for_metric.empty
        and pd.notna(prem_df_for_metric.iloc[0]["溢价"])
        and not (SETTINGS.production_mode and _is_low_confidence_source(prem_df_for_metric.iloc[0].get("来源")))
    ):
        premium_value = f"{prem_df_for_metric.iloc[0]['溢价']:+.2f}%"
        if premium_src.upper() == "SINA":
            premium_delta = "展示用，不参与评分"
    elif SETTINGS.production_mode and not prem_df_for_metric.empty:
        premium_value = "待接入"
        premium_delta = "需 SGE/LBMA 源"
    g5.metric("中国溢价", premium_value, delta=premium_delta, delta_color="off")

    # 第6列聚合：来源 + 更新
    source_info = gold.get("source", "—")
    g6.metric(f"{fr_icon} 数据", f"{source_info}", delta=_ago(gold.get("timestamp", "")))

    # 金价走势图 — 多周期可选
    st.caption("金价走势")
    range_tab = st.radio("周期", ["7天", "30天", "360天"], horizontal=True, index=0, label_visibility="collapsed")
    range_days = {"7天": 7, "30天": 30, "360天": 360}[range_tab]

    import plotly.graph_objects as go
    db = SessionLocal()
    try:
        from app.models import GoldPrice as GP
        rows = db.scalars(select(GP).order_by(GP.date.desc()).limit(range_days)).all()
        if rows:
            df = pd.DataFrame([{"日期": r.date, "收盘": r.close} for r in reversed(rows)])
            # ── 沪金日线 ──
            sh_daily = get_shanghai_daily()
            sh_line = None
            if not sh_daily.empty:
                sh_daily["日期"] = pd.to_datetime(sh_daily["d"])
                sh_daily["收盘_cny"] = sh_daily["c"].astype(float)
                start_d = df["日期"].min()
                sh_line = sh_daily[(sh_daily["日期"] >= start_d - pd.Timedelta(days=2)) & (sh_daily["日期"] <= df["日期"].max() + pd.Timedelta(days=2))]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["日期"], y=df["收盘"], mode="lines+markers",
                name="COMEX",
                line=dict(color="#f0b90b", width=1.5), marker=dict(size=3),
                hovertemplate="%{x|%Y-%m-%d}<br>COMEX $%{y:,.0f}<extra></extra>",
            ))
            # 沪金第二trace
            sh_trace = None
            if sh_line is not None and not sh_line.empty:
                sh_trace = go.Scatter(
                    x=sh_line["日期"], y=sh_line["收盘_cny"], mode="lines+markers",
                    name="沪金",
                    line=dict(color="#dc2626", width=1.2), marker=dict(size=2),
                    yaxis="y2",
                    hovertemplate="%{x|%Y-%m-%d}<br>沪金 ¥%{y:,.1f}/g<extra></extra>",
                )
                fig.add_trace(sh_trace)
            # 右上角图例标注
            fig.add_annotation(
                x=0.005, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top",
                text="<span style='color:#c9972b'>● COMEX</span>  <span style='color:#dc2626'>● 沪金</span>",
                showarrow=False, font=dict(size=11),
                bgcolor="rgba(255,255,255,0.82)", borderpad=4,
            )
            fig.update_layout(
                **PLOTLY_LIGHT_LAYOUT,
                height=300, margin=dict(l=0,r=0,t=0,b=30),
                xaxis=dict(tickformat="%m/%d", tickangle=-45, showgrid=False),
                yaxis=dict(title=None, showgrid=True, gridcolor="#f1f5f9"),
                yaxis2=dict(
                    title=None, overlaying="y", side="right",
                    showgrid=False, color="#dc2626",
                ),
                hovermode="x unified", showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            latest_daily = pd.to_datetime(df["日期"].iloc[-1]).strftime("%Y-%m-%d %H:%M:%S")
            st.caption(f"金价日线：COMEX（黄） + 沪金连续（红，¥/g）。最新日线：{latest_daily}。COMEX 实时报价见上方卡片。")
    finally:
        db.close()

    # 24 小时金价走势趋势图
    comex_raw = get_intraday()
    if not comex_raw.empty:
        coverage_label, intraday_notes = _intraday_coverage_label(comex_raw)
        st.divider()
        col_i1, col_i2 = st.columns([4, 1])
        with col_i1:
            st.caption(coverage_label)
        with col_i2:
            st.markdown(f'<span class="live-dot"></span> <span style="color:#94a3b8;font-size:0.72rem;">实时监控中 · {time.strftime("%H:%M:%S")}</span>', unsafe_allow_html=True)

        # COMEX: resample 1-min raw to 5-min OHLC
        comex_raw["timestamp"] = pd.to_datetime(comex_raw["timestamp"], errors="coerce")
        comex_raw["close"] = pd.to_numeric(comex_raw["close"], errors="coerce")
        comex_raw = comex_raw.dropna(subset=["timestamp", "close"])
        comex_raw = comex_raw.set_index("timestamp").sort_index()
        comex_5m = comex_raw.resample("5min").agg({"close": "last"}).dropna()
        idf = comex_5m.reset_index().rename(columns={"index": "timestamp"})

        # Shanghai: keep original Sina timestamps (now with full date)
        sh_intra = get_shanghai_intraday()
        sh_price_text = ""

        fig_i = go.Figure()
        if not idf.empty:
            # ── COMEX 金价（黄色） ──
            fig_i.add_trace(go.Scatter(
                x=idf["timestamp"], y=idf["close"], mode="lines+markers",
                name="COMEX",
                line=dict(color="#f0b90b", width=1.5), marker=dict(size=2),
                hovertemplate="%{x|%m/%d %H:%M}<br>COMEX $%{y:,.2f}<extra></extra>",
            ))
        # ── 沪金日内（红色） ──
        if not sh_intra.empty:
            sh_price_text = f"沪金 ¥{sh_intra['close'].iloc[-1]:,.1f}/g · "
            fig_i.add_trace(go.Scatter(
                x=sh_intra["timestamp"], y=sh_intra["close"], mode="lines+markers",
                name="沪金",
                line=dict(color="#dc2626", width=1.2), marker=dict(size=2),
                yaxis="y2",
                hovertemplate="%{x|%m/%d %H:%M}<br>沪金 ¥%{y:,.1f}/g<extra></extra>",
            ))
        else:
            _sh = get_shanghai_gold()
            if _sh.get("ok"):
                sh_price_text = f"沪金 ¥{_sh['price']:,.1f}/g · "
        # 右上角图例标注
        fig_i.add_annotation(
            x=0.005, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top",
            text="<span style='color:#c9972b'>● COMEX</span>  <span style='color:#dc2626'>● 沪金</span>",
            showarrow=False, font=dict(size=11),
            bgcolor="rgba(255,255,255,0.82)", borderpad=4,
        )
        fig_i.update_layout(
            **PLOTLY_LIGHT_LAYOUT,
            height=300, margin=dict(l=0,r=0,t=0,b=20),
            xaxis=dict(tickformat="%m/%d %H:%M", showgrid=False),
            yaxis=dict(title=None, showgrid=True, gridcolor="#f1f5f9"),
            yaxis2=dict(
                title=None, overlaying="y", side="right",
                showgrid=False, color="#dc2626",
            ),
            hovermode="x unified", showlegend=False
        )
        st.plotly_chart(fig_i, use_container_width=True)
        caption_parts = [f"{coverage_label} · {sh_price_text}5分钟聚合，来源：新浪财经"]
        caption_parts.extend(note.rstrip("。") for note in intraday_notes)
        st.caption("；".join(caption_parts) + "。")
    else:
        st.caption("日内金价快照不足，后台记录器启动并采集到数据后会显示走势。")
else:
    st.warning("实时金价暂时无法获取。")

st.divider()

# ═══════════════════════════════════════════
# 评分
# ═══════════════════════════════════════════

scores = get_scores()

if scores.empty:
    st.warning("暂无评分数据。")
else:
    latest = scores.iloc[-1]
    raw_factors = json.loads(latest["因子"])
    # v2 格式: {"scores": {...}, "details": {...}}
    if isinstance(raw_factors, dict) and "scores" in raw_factors:
        factors = raw_factors["scores"]
        factor_details = raw_factors.get("details", {})
    else:
        factors = raw_factors  # v1 兼容
        factor_details = {}
    risks = json.loads(latest["风险"])

    sh_col1, sh_col2 = st.columns([4, 1])
    with sh_col1:
        st.subheader("黄金多空评分")
    with sh_col2:
        if st.button("🔄 采集+评分", use_container_width=True, type="primary"):
            api("/score/compute", "post")
            st.cache_data.clear()

    s1, s2, s3, s4 = st.columns(4)
    score_val = latest["评分"]
    direction_icon = "🟢" if score_val >= 30 else ("🔴" if score_val <= -30 else "🟡")
    # 综合评分突出显示
    score_color = "#047857" if score_val >= 30 else ("#b91c1c" if score_val <= -30 else "#b45309")
    score_bg = "#f0fdf4" if score_val >= 30 else ("#fef2f2" if score_val <= -30 else "#fffbeb")
    dir_label = "偏多" if score_val >= 30 else ("偏空" if score_val <= -30 else "中性")
    s1.markdown(
        f'<div style="text-align:center;padding:12px 8px;border-radius:8px;'
        f'background:{score_bg};border:2px solid {score_color};min-width:120px">'
        f'<div style="font-size:0.72rem;color:#64748b;margin-bottom:2px">综合评分 · {dir_label}</div>'
        f'<div style="font-size:1.55rem;font-weight:800;color:{score_color}">{direction_icon} {score_val:+.1f}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    bull = sum(v for v in factors.values() if v > 0)
    bear = sum(abs(v) for v in factors.values() if v < 0)
    net = bull - bear
    s2.metric("因子总数", len(factors))
    s3.metric("利多合计", f"+{bull:.0f}分")
    s4.metric("利空合计", f"-{bear:.0f}分")
    # ── 评分参考说明 ──
    st.markdown(
        '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px 14px;margin:4px 0 8px 0">'
        '<span style="font-size:0.78rem;color:#64748b">📖 '
        '<b>评分解读</b>：+30 以上偏多，−30 以下偏空，中间为中性。'
        f'当前评分基于 {len(factors)} 项已入库因子（利率、美元、流动性、持仓、情绪、央行等）加权计算，'
        '正分=利多黄金，负分=利空黄金。仅供参考，不构成投资建议。</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    factor_help = registry_factor_help()
    factor_groups = registry_factor_groups()
    inactive_factor_reasons = registry_inactive_reasons()

    def render_factor_card(column, name: str, value: float | None, tooltip: str) -> None:
        safe_name = html.escape(name)
        safe_tooltip = html.escape(tooltip or "暂无说明。", quote=True)
        if value is None:
            class_name = "factor-card factor-card-inactive"
            value_text = "未评分"
        else:
            icon = "🟢" if value > 0 else ("🔴" if value < 0 else "⚪")
            class_name = "factor-card factor-card-positive" if value > 0 else (
                "factor-card factor-card-negative" if value < 0 else "factor-card factor-card-neutral"
            )
            value_text = f"{icon} {value:+.1f}"
        column.markdown(
            f"""
            <div class="{class_name}" data-tooltip="{safe_tooltip}">
              <div class="factor-card-title">{safe_name}</div>
              <div class="factor-card-value">{html.escape(value_text)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    shown: set[str] = set()
    for group_name, names in factor_groups.items():
        group_items = [(name, factors.get(name)) for name in names if name in factors or name in inactive_factor_reasons]
        if not group_items:
            continue
        st.markdown(f'<div class="section-label">{group_name}</div>', unsafe_allow_html=True)
        per_row = 10
        for i in range(0, len(group_items), per_row):
            row_items = group_items[i:i + per_row]
            cols = st.columns(len(row_items))
            for j, (name, val) in enumerate(row_items):
                if val is None:
                    reason = inactive_factor_reasons.get(name, "暂无可信数据入库；后续可接入数据源或手动录入。")
                    render_factor_card(cols[j], name, None, reason)
                else:
                    render_factor_card(cols[j], name, val, factor_help.get(name, ""))
                    shown.add(name)

    other_factors = sorted(
        [(name, val) for name, val in factors.items() if name not in shown],
        key=lambda x: x[1],
        reverse=True,
    )
    if other_factors:
        st.markdown('<div class="section-label">其他已入库因子</div>', unsafe_allow_html=True)
        for i in range(0, len(other_factors), 10):
            row_items = other_factors[i:i + 10]
            cols = st.columns(len(row_items))
            for j, (name, val) in enumerate(row_items):
                render_factor_card(cols[j], name, val, factor_help.get(name, ""))

    # 计算详情
    if factor_details:
        with st.expander("📐 计算详情", expanded=False):
            st.dataframe(
                pd.DataFrame([{"因子": n, **{k: f"{v:+.3f}" for k,v in d.items()}} for n,d in factor_details.items()]),
                use_container_width=True, hide_index=True,
            )

    # 评分走势图 — Plotly 竖线+悬浮
    chart_scores = scores[["时间", "评分"]].copy()
    chart_scores["时间"] = pd.to_datetime(chart_scores["时间"])

    col_scr1, col_scr2 = st.columns([3, 1])
    with col_scr1:
        days_back = st.select_slider("范围", options=[7,14,30,90,180,360], value=30, label_visibility="collapsed")
    with col_scr2:
        st.caption(f"近 {days_back} 天")

    cutoff = pd.Timestamp.now(tz=chart_scores["时间"].iloc[-1].tz) - pd.Timedelta(days=days_back)
    chart_scores = chart_scores[chart_scores["时间"] >= cutoff].set_index("时间").sort_index()

    import plotly.graph_objects as go
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=chart_scores.index, y=chart_scores["评分"], mode="lines",
        line=dict(color="#c9972b", width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>评分: %{y:+.1f}<extra></extra>",
    ))
    fig_s.add_hline(y=0, line=dict(color="#94a3b8", dash="dash", width=0.5))
    # 默认放大到最近 2/3 区域
    default_start = chart_scores.index[max(0, len(chart_scores) - max(3, len(chart_scores) * 2 // 3))]
    default_end = chart_scores.index[-1]
    fig_s.update_layout(
        **PLOTLY_LIGHT_LAYOUT,
        height=280, margin=dict(l=0,r=0,t=0,b=20),
        yaxis=dict(title=None, showgrid=True, gridcolor="#f1f5f9"),
        hovermode="x unified", showlegend=False,
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False),
                   range=[default_start, default_end]),
        dragmode="pan",
    )
    st.plotly_chart(fig_s, use_container_width=True, config={
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "displaylogo": False,
        "scrollZoom": True,
    })
    st.caption(f"最新评分时间：{pd.to_datetime(latest['时间']).strftime('%Y-%m-%d %H:%M:%S')}。评分曲线为日频/快照数据，不代表逐秒行情。")

    # 风险提示 — 可折叠
    with st.expander(f"💡 风险提示（{len(risks)} 条）", expanded=len(risks) <= 3):
        for i, r in enumerate(risks):
            color = "#f0fdf4" if i % 2 == 0 else "#fffbeb"
            icon = "📌"
            st.markdown(
                f'<div style="background:{color};border:1px solid #e2e8f0;border-radius:6px;'
                f'padding:8px 14px;margin:4px 0;font-size:0.88rem;color:#475569;">'
                f'{icon} {r}</div>',
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════
# 预测
# ═══════════════════════════════════════════

st.divider()
st.subheader("金价预测")

@st.cache_data(ttl=90)
def get_prediction() -> tuple[dict, str]:
    """返回 (data_dict, error_msg)。error_msg 为空时表示成功。"""
    try:
        with httpx.Client(timeout=httpx.Timeout(45)) as c:
            r = c.get("http://localhost:8000/predict/gold")
            if r.status_code != 200:
                return {}, f"API 返回状态 {r.status_code}"
            data = r.json()
            if not data.get("ok"):
                return data, data.get("reason", "未知错误")
            return data, ""
    except Exception as e:
        return {}, f"连接后端失败：{e}"


@st.cache_data(ttl=60)
def get_prediction_evaluation() -> dict:
    return api("/predict/evaluation")


@st.cache_data(ttl=60)
def get_prediction_models() -> dict:
    return api("/predict/models")


@st.cache_data(ttl=45)
def get_prediction_due_status() -> dict:
    return api("/predict/due-status")

pred, pred_error = get_prediction()
if pred.get("ok"):
    current = pred.get("current_price")
    due_status = get_prediction_due_status()
    evaluated_count = int(due_status.get("evaluated_count") or 0)
    due_pending_count = int(due_status.get("due_pending_count") or 0)
    future_pending_count = int(due_status.get("future_pending_count") or 0)
    st.caption(
        f"v2 多信号集成：模型 {pred.get('model_version', '—')}，训练源 {', '.join(pred.get('training_sources', []))}。"
        "短期动量+评分回归，长期宏观基准+调整。仅供参考，不构成投资建议。"
    )
    st.caption(
        f"预测闭环状态：已评估 {evaluated_count} 条，到期待评估 {due_pending_count} 条，"
        f"待到期 {future_pending_count} 条。{due_status.get('message', '')}"
    )

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1.2, 3.8])
    with ctrl1:
        if st.button("保存本次预测", use_container_width=True):
            api("/predict/gold/snapshot", "post")
            st.cache_data.clear()
            st.rerun()
    with ctrl2:
        if st.button(f"补评估到期预测({due_pending_count})", use_container_width=True):
            api("/predict/evaluate", "post")
            st.cache_data.clear()
            st.rerun()
    with ctrl3:
        candidate_help = "评估样本少于 120 条，不建议生成候选模型。" if evaluated_count < 120 else "样本条件基本满足，可生成候选模型。"
        st.caption(candidate_help)
        if st.button("生成候选模型", use_container_width=True):
            r = api("/predict/models/optimize", "post", params={"n_iter": 40, "top_k": 5, "save_best": True})
            st.session_state["prediction_optimize_result"] = r
            st.cache_data.clear()
            st.rerun()
    opt_result = st.session_state.get("prediction_optimize_result")
    if opt_result:
        if opt_result.get("ok"):
            best = opt_result.get("best") or {}
            activation = opt_result.get("activation") or {}
            mode = "已自动激活" if activation.get("activated") else "等待人工激活"
            st.success(
                f"已生成候选模型 {opt_result.get('saved_version')}："
                f"综合分 {best.get('optimization_score')}，"
                f"MAPE {best.get('weighted_mape_price_pct')}%，"
                f"方向准确率 {best.get('weighted_direction_accuracy')}。"
                f"{mode}。"
            )
            if activation.get("reasons"):
                st.caption("自动激活判断：" + "；".join(activation.get("reasons", [])))
        else:
            st.warning(f"候选模型生成失败：{opt_result.get('reason', '未知原因')}")

    preds = pred.get("predictions", [])
    if preds:
        # Hover tooltip 样式
        st.markdown("""<style>
        .pred-wrap { position:relative; display:inline-block; width:100%; }
        .pred-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
            padding:8px 10px; cursor:pointer; transition:box-shadow 0.15s; }
        .pred-card:hover { box-shadow:0 2px 8px rgba(0,0,0,0.12); }
        .pred-card h4 { margin:0 0 2px 0; font-size:0.8rem; color:#64748b; }
        .pred-card .price { font-size:1.15rem; font-weight:700; color:#1e293b; }
        .pred-card .delta { font-size:0.82rem; margin-left:6px; }
        .pred-card .range { font-size:0.72rem; color:#94a3b8; margin-top:2px; }
        .pred-card .confidence { font-size:0.68rem; color:#64748b; margin-top:1px;
            background:#f1f5f9; display:inline-block; padding:1px 6px; border-radius:3px; }
        .pred-tip { visibility:hidden; opacity:0; transition:opacity 0.2s;
            position:absolute; z-index:999; bottom:110%; left:-10px;
            background:#ffffff; border:1px solid #d7dde6; border-radius:8px; padding:14px 18px;
            font-size:0.78rem; box-shadow:0 10px 28px rgba(15,23,42,0.16);
            min-width:460px; pointer-events:none; white-space:normal; }
        .pred-wrap:last-child .pred-tip { left:auto; right:-10px; }
        .pred-wrap:first-child .pred-tip { left:0; }
        .pred-tip-right { left:auto !important; right:0 !important; }
        .pred-tip-left { left:0 !important; }
        .pred-card:hover + .pred-tip,
        .pred-card:active + .pred-tip,
        .pred-card:focus + .pred-tip { visibility:visible; opacity:1; }
        .pred-tip td { padding:4px 12px; vertical-align:top; color:#334155; line-height:1.55; }
        .pred-tip td:first-child { border-right:1px solid #e2e8f0; width:180px; }
        .pred-tip .th { color:#64748b; font-size:0.7rem; text-transform:uppercase; }
        .pred-tip .h1 { color:#18212f; font-size:0.95rem; font-weight:700; }
        .pred-tip .h2 { color:#18212f; font-size:0.82rem; }
        .pred-tip .h3 { color:#475569; font-size:0.72rem; }
        </style>""", unsafe_allow_html=True)

        cols = st.columns(len(preds))
        for i, p in enumerate(preds):
            icon = "🟢" if p.get("return_pct", 0) > 0 else "🔴"
            rel = p.get("reliability", 0.5)
            note = p.get("note", "")
            low = p.get("low", 0)
            high = p.get("high", 0)
            color = "#16a34a" if p.get("return_pct", 0) > 0 else "#dc2626"

            note_html = note.replace("\n", "<br>") if note else "暂无详情"

            with cols[i]:
                tip_class = ""
                if i == 0:
                    tip_class = " pred-tip-left"
                elif i == len(preds) - 1:
                    tip_class = " pred-tip-right"

                st.markdown(f"""
                <div class="pred-wrap">
                  <div class="pred-card" tabindex="0">
                    <h4>{p.get('horizon', '?')}</h4>
                    <span class="price">${p.get('predicted', 0):,.0f}</span>
                    <span class="delta" style="color:{color}">{icon} {p.get('return_pct', 0):+.1f}%</span>
                    <div class="range">${low:,.0f} — ${high:,.0f}</div>
                    <div class="confidence">置信度 {rel:.0%}</div>
                  </div>
                  <div class="pred-tip{tip_class}">
                    <table><tr>
                    <td>
                      <div class="th">预测价格</div><div class="h1">${p.get('predicted', 0):,.0f}</div>
                      <div class="th">预期收益 · 置信度</div><div class="h2" style="color:{color}">{icon} {p.get('return_pct', 0):+.1f}% · {rel:.0%}</div>
                      <div class="th">波动区间</div><div class="h3">${low:,.0f} — ${high:,.0f}</div>
                      <div class="th">样本量</div><div class="h3">{p.get('samples', 0)} 条</div>
                    </td>
                    <td>
                      <div class="th">预测理由</div>
                      <div class="h3">{note_html}</div>
                    </td>
                    </tr></table>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div style='height:54px'></div>", unsafe_allow_html=True)

    # 预测曲线图
    if current and preds:
        import plotly.graph_objects as go
        horizons = [p["horizon"] for p in preds]
        pred_prices = [p["predicted"] for p in preds]
        lows = [p["low"] for p in preds]
        highs = [p["high"] for p in preds]

        x_labels = ["当前"] + horizons
        y_vals = [current] + pred_prices
        y_low = [current] + lows
        y_high = [current] + highs

        fig_p = go.Figure()
        # 置信区间填充
        fig_p.add_trace(go.Scatter(
            x=x_labels + x_labels[::-1],
            y=y_high + y_low[::-1],
            fill="toself", fillcolor="rgba(240,185,11,0.15)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        # 预测线
        fig_p.add_trace(go.Scatter(
            x=x_labels, y=y_vals, mode="lines+markers",
            line=dict(color="#f0b90b", width=1.5), marker=dict(size=6),
            hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
        ))
        fig_p.update_layout(
            **PLOTLY_LIGHT_LAYOUT,
            height=300, margin=dict(l=0,r=0,t=0,b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(title=None, showgrid=True, gridcolor="#f1f5f9"),
            hovermode="x unified", showlegend=False
        )
        st.plotly_chart(fig_p, use_container_width=True)

        with st.expander("预测理论与理由", expanded=False):
            st.caption("当前点为真实价格；未来点为模型估计。以下内容自动随 /predict/gold 重新计算。")
            for p in preds:
                note = p.get("note") or "暂无预测理由。"
                err = p.get("error_metrics") or {}
                err_text = ""
                if err.get("ok"):
                    err_text = (
                        f"历史误差：MAE ${err.get('mae_price', 0):,.0f}，"
                        f"MAPE {err.get('mape_price_pct', 0):.1f}%，"
                        f"方向准确率 {err.get('direction_accuracy', 0):.0%}。"
                    )
                st.markdown(
                    f"**{p.get('horizon', '?')} · 可靠性{p.get('reliability_label', '低')}**  \n"
                    f"预测价 `${p.get('predicted', 0):,.0f}`，"
                    f"预期收益 `{p.get('return_pct', 0):+.1f}%`，"
                    f"区间 `${p.get('low', 0):,.0f} - ${p.get('high', 0):,.0f}`。"
                )
                if err_text:
                    st.caption(err_text)
                st.markdown(note.replace("\n", "  \n"))

        eval_data = get_prediction_evaluation()
        model_data = get_prediction_models()
        if eval_data.get("ok"):
            summary = eval_data.get("summary", {})
            st.markdown("#### 预测验证闭环")
            e1, e2, e3, e4, e5 = st.columns(5)
            e1.metric("已验证", f"{summary.get('evaluated_count', 0)} 条")
            e2.metric("待到期", f"{summary.get('future_pending_count', 0)} 条")
            e3.metric("到期待评估", f"{summary.get('due_pending_count', 0)} 条")
            mae = summary.get("mae_price")
            mape = summary.get("mape_price_pct")
            acc = summary.get("direction_accuracy")
            e4.metric("MAE", f"${mae:,.0f}" if mae is not None else "—")
            e5.metric("方向准确率", f"{acc:.0%}" if acc is not None else "—", delta=f"MAPE {mape:.1f}%" if mape is not None else None)

            by_h = eval_data.get("by_horizon", [])
            if by_h:
                hdf = pd.DataFrame(by_h)
                hdf = hdf.rename(columns={
                    "horizon_days": "期限",
                    "count": "样本",
                    "mae_price": "MAE($)",
                    "mape_price_pct": "MAPE(%)",
                    "direction_accuracy": "方向准确率",
                })
                hdf["方向准确率"] = hdf["方向准确率"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
                st.dataframe(hdf, use_container_width=True, hide_index=True)
            else:
                st.caption("还没有到期预测。保存快照后，等对应 horizon 到期并采集到真实金价，系统会自动比对。")

        if model_data.get("ok") and model_data.get("data"):
            with st.expander("预测模型版本", expanded=False):
                mdf = pd.DataFrame(model_data["data"])
                show_cols = [
                    "version", "method", "is_active", "evaluated_count",
                    "mae_price", "mape_price_pct", "direction_accuracy", "notes",
                ]
                existing_cols = [c for c in show_cols if c in mdf.columns]
                st.dataframe(mdf[existing_cols], use_container_width=True, hide_index=True)
                st.caption("候选模型默认不自动激活。确认表现后，可调用 API：POST /predict/models/{version}/activate。")
else:
    if pred_error:
        st.warning(f"预测数据加载失败：{pred_error}")
        if st.button("重试加载预测", key="retry_pred"):
            st.cache_data.clear()
            st.rerun()
    else:
        st.caption(
            "暂无预测数据（需要足够的评分历史或已配置的同版本评分源）。"
            " 请确认已采集 FRED/CFTC 等核心数据并至少执行过一次评分计算。"
        )

# ═══════════════════════════════════════════
# 宏观指标
# ═══════════════════════════════════════════

st.divider()
st.subheader("宏观指标")

macro = get_macro()
if not macro.empty:
    import plotly.express as px
    series_ids = macro["series_id"].unique()
    preferred_series = [
        "DFII5", "DFII10", "DFII30", "THREEFYTP10",
        "WALCL", "WDTGAL", "RRPONTSYD", "WRESBAL",
        "GFDEGDQ188S", "FYFSD",
    ]
    default_series = [sid for sid in preferred_series if sid in series_ids][:5] or list(series_ids[:4])
    selected = st.multiselect("选择指标", series_ids, default=default_series)
    if selected:
        mdf = _finite_chart_frame(
            macro[macro["series_id"].isin(selected)],
            required_cols=["时间", "值"],
            numeric_cols=["值"],
        )
        if not mdf.empty:
            macro_chart = px.line(mdf, x="时间", y="值", color="指标")
            macro_chart.update_layout(
                **PLOTLY_LIGHT_LAYOUT,
                height=220,
                margin=dict(l=0, r=0, t=8, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                yaxis_title=None,
                xaxis_title=None,
            )
            st.plotly_chart(macro_chart, use_container_width=True)
        else:
            st.caption("所选宏观指标暂无可绘制数值。")

external_df = get_external_indicators()
if not external_df.empty:
    with st.expander("ETF / COMEX / 期权 / 地缘 / 实物需求外部指标"):
        st.dataframe(external_df, use_container_width=True, hide_index=True)

with st.expander("手动录入外部指标", expanded=False):
    catalog_payload = api("/external/indicators/catalog")
    catalog_rows = catalog_payload.get("data", []) if catalog_payload.get("ok") else []
    manual_candidates = [
        row for row in catalog_rows
        if row.get("indicator_id") in {
            "COMEX_REGISTERED_GOLD_OZ",
            "COMEX_GOLD_FRONT_SPREAD_PCT",
            "GEO_RISK_INTENSITY",
            "INDIA_CHINA_PHYSICAL_DEMAND",
            "GLD_FLOW_TONNES",
            "GOLD_OPTION_IV_30D",
            "GOLD_OPTION_SKEW_25D",
        }
    ]
    if manual_candidates:
        labels = [f"{row.get('name')} ({row.get('indicator_id')})" for row in manual_candidates]
        selected_label = st.selectbox("指标", labels, key="manual_indicator_select")
        selected_meta = manual_candidates[labels.index(selected_label)]
        m1, m2, m3 = st.columns([1, 1, 1])
        with m1:
            manual_value = st.number_input(f"数值（{selected_meta.get('unit') or ''}）", value=0.0, key="manual_indicator_value")
        with m2:
            manual_date = st.date_input("日期", value=_now_beijing().date(), key="manual_indicator_date")
        with m3:
            manual_source = st.text_input("来源", value="MANUAL", key="manual_indicator_source")
        manual_note = st.text_input("备注", value=str(selected_meta.get("reason") or ""), key="manual_indicator_note")
        st.caption(str(selected_meta.get("reason") or ""))
        if st.button("保存外部指标", use_container_width=True):
            ts = dt.datetime.combine(manual_date, dt.time.min, tzinfo=UTC_TZ).isoformat()
            result = api(
                "/external/indicators",
                "post",
                json={
                    "indicator_id": selected_meta.get("indicator_id"),
                    "timestamp": ts,
                    "value": manual_value,
                    "source": manual_source or "MANUAL",
                    "name": selected_meta.get("name"),
                    "category": selected_meta.get("category"),
                    "unit": selected_meta.get("unit"),
                    "note": manual_note,
                },
            )
            if result.get("ok"):
                st.success("已保存外部指标")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(result.get("reason", "保存失败"))
    else:
        st.caption("暂无可手动录入的外部指标目录。")

# ═══════════════════════════════════════════
# CFTC
# ═══════════════════════════════════════════

st.divider()
st.subheader("CFTC 黄金期货持仓")
cftc = get_cftc()
if not cftc.empty:
    import plotly.express as px
    cftc_melt = cftc.melt(id_vars=["时间"], value_vars=["多", "空", "净多"],
                          var_name="类型", value_name="合约数")
    cftc_melt = _finite_chart_frame(
        cftc_melt,
        required_cols=["时间", "合约数"],
        numeric_cols=["合约数"],
    )
    if not cftc_melt.empty:
        cftc_chart = px.line(
            cftc_melt,
            x="时间",
            y="合约数",
            color="类型",
            color_discrete_map={"多": "#10b981", "空": "#ef4444", "净多": "#6366f1"},
            markers=len(cftc) <= 5,
        )
        cftc_chart.update_layout(
            **PLOTLY_LIGHT_LAYOUT,
            height=200,
            margin=dict(l=0, r=0, t=8, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            yaxis_title="合约数",
            xaxis_title=None,
        )
        st.plotly_chart(cftc_chart, use_container_width=True)
    else:
        st.caption("CFTC 数据暂无可绘制合约数。")
else:
    st.caption("暂无 CFTC 数据。")

# ═══════════════════════════════════════════
# 央行购金
# ═══════════════════════════════════════════

st.divider()
st.subheader("央行购金")
global_cb, country_cb = get_cb_gold()
if (
    SETTINGS.production_mode
    and not global_cb.empty
    and global_cb["来源"].astype(str).str.upper().isin(LOW_CONFIDENCE_SOURCES).all()
    and not SETTINGS.show_low_confidence_data
):
    st.caption("央行购金数据来源：WGC/IMF IFS，基于季度报告按月估算。")
elif not global_cb.empty:
    import plotly.express as px
    cb_plot = _finite_chart_frame(
        global_cb,
        required_cols=["月份", "净购金(吨)"],
        numeric_cols=["净购金(吨)"],
    )
    if not cb_plot.empty:
        cb_chart = px.bar(cb_plot, x="月份", y="净购金(吨)", hover_data=["来源"])
        cb_chart.update_traces(marker_color="#c9972b")
        cb_chart.update_layout(
            **PLOTLY_LIGHT_LAYOUT,
            height=190,
            margin=dict(l=0, r=0, t=8, b=20),
            yaxis_title="吨",
            xaxis_title=None,
            showlegend=False,
        )
        st.plotly_chart(cb_chart, use_container_width=True)
    else:
        st.caption("央行购金暂无可绘制数值。")
else:
    st.caption("央行购金可验证来源待接入。")

# ═══════════════════════════════════════════
# 宏观事件
# ═══════════════════════════════════════════

st.divider()
st.subheader("宏观事件")
events = get_events()
if not events.empty:
    evt_df = events[["时间", "事件", "国家", "重要性"]].copy()
    evt_df["时间"] = evt_df["时间"].apply(lambda t: t.strftime("%m-%d %H:%M") if hasattr(t, "strftime") else str(t)[:16])
    st.dataframe(evt_df, use_container_width=True, hide_index=True)
else:
    st.caption("未来60天暂无宏观事件。")

# ═══════════════════════════════════════════
# 新闻情绪
# ═══════════════════════════════════════════

st.divider()
st.subheader("新闻情绪")
sent_score, sent_df = get_sentiment()
st.caption(f"生产新闻源：NewsAPI（每日限额 {SETTINGS.newsapi_daily_limit} 次）；兼容展示 GDELT。密钥来自 .env，不在页面暴露。")
if sent_score is not None:
    sent_icon = "🟢" if sent_score > 0 else "🔴"
    st.metric("最新情绪评分", f"{sent_icon} {sent_score:+.2f}",
              help=">0 偏利多黄金，<0 偏利空黄金。基于新闻标题关键词情感分析。")
    if not sent_df.empty:
        import plotly.graph_objects as go
        recent = sent_df.tail(30).copy()
        colors = ["#10b981" if v > 0 else "#ef4444" for v in recent["情绪"]]
        fig_sent = go.Figure()
        fig_sent.add_trace(go.Bar(
            x=recent["时间"], y=recent["情绪"],
            marker_color=colors,
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>情绪: %{y:+.2f}<br>%{customdata}<extra></extra>",
            customdata=recent["标题"].fillna(""),
        ))
        fig_sent.update_layout(
            **PLOTLY_LIGHT_LAYOUT,
            height=140, margin=dict(l=0,r=0,t=0,b=20),
            xaxis=dict(tickformat="%H:%M", dtick=600000, showgrid=False),
            yaxis=dict(title=None, showgrid=True, gridcolor="#f1f5f9"),
            hovermode="x unified", showlegend=False
        )
        st.plotly_chart(fig_sent, use_container_width=True)

        # 可折叠的新闻列表
        with st.expander("新闻列表", expanded=False):
            show = recent.sort_values("时间", ascending=False).copy()
            show["时间"] = show["时间"].apply(
                lambda t: t.strftime("%Y-%m-%d %H:%M") if hasattr(t, "strftime") else str(t)[:16]
            )
            # 标题做成可点击链接
            def _make_link(row):
                url = row.get("来源", "")
                title = row["标题"]
                if url and str(url).startswith("http"):
                    return f'<a href="{url}" target="_blank">{title}</a>'
                return title
            show["标题"] = show.apply(_make_link, axis=1)
            show["情绪"] = show["情绪"].apply(lambda x: f"{x:+.2f}")
            st.write(
                show[["时间", "标题", "情绪", "数据源"]].to_html(
                    index=False, escape=False,
                ),
                unsafe_allow_html=True,
            )
else:
    st.caption("暂无 NewsAPI/GDELT 新闻情绪数据。")

# ═══════════════════════════════════════════
# 评分模型自我进化
# ═══════════════════════════════════════════

st.divider()
with st.expander("评分模型自我进化", expanded=False):
    st.caption(
        "通过随机搜索 + 滚动回测自动寻找最优因子权重。"
        "每次优化保存一个参数版本，可追溯、可回滚。"
    )

    col_opt1, col_opt2, col_opt3 = st.columns([2, 1, 1])
    with col_opt1:
        n_iter = st.slider("搜索迭代次数", 20, 200, 50, 10, key="opt_n_iter")
    with col_opt2:
        horizon_days = st.selectbox("回测展望期（天）", [10, 20, 30, 60], index=1, key="opt_horizon")
    with col_opt3:
        do_opt = st.button("🚀 开始优化", use_container_width=True, type="primary")

    if do_opt:
        with st.spinner(f"随机搜索 {n_iter} 次，评估 {horizon_days} 天命中率（约需 {n_iter*5//60} 分钟）..."):
            import httpx
            try:
                resp = httpx.post(
                    "http://localhost:8000/score/optimize",
                    params={"n_iter": n_iter, "horizon_days": horizon_days},
                    timeout=httpx.Timeout(600),
                )
                r = resp.json() if resp.status_code == 200 else {}
            except Exception:
                r = {}
        if r.get("ok"):
            best = r.get("best", {})
            baseline = r.get("baseline") or {}
            best_hr = best.get("hit_rate")
            base_hr = baseline.get("hit_rate")
            st.success(f"优化完成！版本 {r.get('version', '?')}")
            c1, c2 = st.columns(2)
            c1.metric("最优命中率", f"{best_hr*100:.1f}%" if best_hr else "—")
            c2.metric("默认命中率", f"{base_hr*100:.1f}%" if base_hr else "—")
        else:
            st.error(r.get("reason", "优化失败"))

    # 显示历史版本
    @st.cache_data(ttl=120)
    def get_param_versions() -> pd.DataFrame:
        payload = api("/score/params")
        rows = payload.get("data", []) if payload.get("ok") else []
        return pd.DataFrame([{
            "版本": r.get("version"),
            "命中率": f"{r.get('hit_rate')*100:.1f}%" if r.get("hit_rate") else "—",
            "样本数": r.get("sample_count") or "—",
            "激活": "✅" if r.get("is_active") else "",
            "创建时间": str(r.get("created_at") or "")[:16].replace("T", " "),
            "备注": r.get("notes") or "",
        } for r in rows])

    @st.cache_data(ttl=300)
    def get_param_compare(version: str) -> dict:
        return api(f"/score/params/{version}/compare")

    versions = get_param_versions()
    if not versions.empty:
        st.caption("参数版本历史")
        st.dataframe(versions, use_container_width=True, hide_index=True)

        # 激活/回滚
        col_a1, col_a2, col_a3 = st.columns([2, 1, 1])
        with col_a1:
            active_ver = st.selectbox("选择版本", versions["版本"].tolist(), key="activate_ver")
        with col_a2:
            if st.button("✅ 激活", use_container_width=True):
                rr = api(
                    f"/score/params/{active_ver}/activate",
                    "post",
                    json={"operator": "dashboard", "reason": "人工审核后在仪表盘激活"},
                )
                if rr.get("ok"):
                    st.success(f"已激活 {active_ver}")
                    risk = rr.get("overfit_risk") or {}
                    if risk.get("not_recommended_for_direct_activation"):
                        st.warning("该版本存在过拟合风险：" + "；".join(risk.get("warnings", [])))
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(rr.get("reason", "失败"))
        with col_a3:
            if st.button("🔄 恢复默认", use_container_width=True):
                rr = api(
                    "/score/params/deactivate",
                    "post",
                    json={"operator": "dashboard", "reason": "人工恢复默认评分规则"},
                )
                if rr.get("ok"):
                    st.success("已恢复默认规则 v2")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(rr.get("reason", "失败"))

        compare = get_param_compare(active_ver)
        if compare.get("ok"):
            detail = compare.get("data", {})
            candidate = detail.get("candidate") or {}
            baseline = detail.get("baseline") or {}
            risk = detail.get("overfit_risk", {})
            st.markdown("#### 候选对比详情")
            metric_rows = [
                ("保存命中率", None, candidate.get("stored_hit_rate")),
                ("命中率", baseline.get("hit_rate"), candidate.get("hit_rate")),
                ("信号覆盖率", baseline.get("signal_ratio"), candidate.get("signal_ratio")),
                ("方向样本数", baseline.get("signal_count"), candidate.get("signal_count")),
                ("多头样本", baseline.get("long_signal_count"), candidate.get("long_signal_count")),
                ("空头样本", baseline.get("short_signal_count"), candidate.get("short_signal_count")),
                ("尾部收益", baseline.get("worst_decile_return"), candidate.get("worst_decile_return")),
                ("近期窗口命中率", baseline.get("recent_hit_rate"), candidate.get("recent_hit_rate")),
                ("相对baseline提升", 0, candidate.get("baseline_lift")),
            ]
            st.dataframe(
                pd.DataFrame([
                    {"指标": name, "baseline": base, "candidate": cand}
                    for name, base, cand in metric_rows
                ]),
                use_container_width=True,
                hide_index=True,
            )
            if risk.get("level") == "high":
                st.warning("过拟合风险高：" + "；".join(risk.get("warnings", [])))
            elif risk.get("level") == "medium":
                st.info("过拟合风险提示：" + "；".join(risk.get("warnings", [])))
            else:
                st.caption("过拟合检查：" + "；".join(risk.get("warnings", [])))
            st.caption(detail.get("recommendation", ""))

        audits = api("/models/activation-audit", params={"limit": 20})
        if audits.get("ok") and audits.get("data"):
            with st.expander("激活审计记录", expanded=False):
                audit_df = pd.DataFrame(audits["data"])
                show_cols = ["created_at", "model_type", "action", "from_version", "to_version", "operator", "reason"]
                st.dataframe(audit_df[[c for c in show_cols if c in audit_df.columns]], use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════
# 底部状态
# ═══════════════════════════════════════════

st.divider()
st.caption(
    "本系统仅提供数据分析和风险提示，不构成任何投资建议。"
    f" 数据刷新间隔 {st.session_state['_rf_interval']}s · "
    f"北京时间 {_now_beijing().strftime('%Y-%m-%d %H:%M:%S')}"
)
