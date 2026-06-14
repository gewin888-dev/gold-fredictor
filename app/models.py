from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MacroSeries(Base):
    __tablename__ = "macro_series"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    series_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    frequency: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    unit: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="FRED", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class MacroObservation(Base):
    __tablename__ = "macro_observations"
    __table_args__ = (UniqueConstraint("series_id", "timestamp", name="uq_macro_series_timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    series_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="FRED", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class ExternalMarketIndicator(Base):
    """授权/人工维护的外部黄金市场指标。

    用于承接 ETF 持仓与资金流、COMEX 库存/期限结构、期权隐含波动率、
    地缘风险、印度/中国实物需求等暂不适合直接用免费 API 稳定采集的数据。
    """
    __tablename__ = "external_market_indicators"
    __table_args__ = (UniqueConstraint("indicator_id", "timestamp", name="uq_external_indicator_timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    indicator_id: Mapped[str] = mapped_column(String(96), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="MANUAL", nullable=False)
    note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class GoldScoreSnapshot(Base):
    __tablename__ = "gold_score_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    total_score: Mapped[float] = mapped_column(Float, nullable=False)
    direction: Mapped[str] = mapped_column(String(32), nullable=False)
    factor_scores: Mapped[str] = mapped_column(Text, nullable=False)
    risk_flags: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="rule_v1", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class CftcPosition(Base):
    __tablename__ = "cftc_positions"
    __table_args__ = (UniqueConstraint("contract_market_code", "timestamp", name="uq_cftc_contract_timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    market_name: Mapped[str] = mapped_column(String(255), nullable=False)
    contract_market_code: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    exchange_code: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    open_interest: Mapped[int] = mapped_column(Integer, nullable=False)
    noncommercial_long: Mapped[int] = mapped_column(Integer, nullable=False)
    noncommercial_short: Mapped[int] = mapped_column(Integer, nullable=False)
    noncommercial_spreading: Mapped[int] = mapped_column(Integer, nullable=False)
    commercial_long: Mapped[int] = mapped_column(Integer, nullable=False)
    commercial_short: Mapped[int] = mapped_column(Integer, nullable=False)
    noncommercial_net: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="CFTC", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class MacroEvent(Base):
    __tablename__ = "macro_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    event_id: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    country: Mapped[str] = mapped_column(String(64), default="US", nullable=False)
    importance: Mapped[str] = mapped_column(String(32), default="high", nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="MANUAL", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class AppSetting(Base):
    """非密钥运行配置。

    页面只允许写入这些安全开关；API 密钥仍由 .env 管理。
    """
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    value_type: Mapped[str] = mapped_column(String(32), default="str", nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="API", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class ScoreParamsVersion(Base):
    """存储评分模型的参数版本，支持自我进化。"""
    __tablename__ = "score_params_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    version: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    params_json: Mapped[str] = mapped_column(Text, nullable=False)
    hit_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sample_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    backtest_horizon_days: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class PredictionModelVersion(Base):
    """预测模型参数版本，用于追踪和受控迭代预测方法。"""
    __tablename__ = "prediction_model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    version: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    method: Mapped[str] = mapped_column(String(128), nullable=False)
    params_json: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    mae_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mape_price_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    direction_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    evaluated_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class ModelActivationAudit(Base):
    """模型/评分版本激活与回滚审计记录。"""
    __tablename__ = "model_activation_audits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_type: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    action: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    from_version: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    to_version: Mapped[str] = mapped_column(String(128), nullable=False)
    operator: Mapped[str] = mapped_column(String(128), default="dashboard", nullable=False)
    reason: Mapped[str] = mapped_column(Text, default="", nullable=False)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class GoldPredictionSnapshot(Base):
    """每一次金价预测的可追溯快照。"""
    __tablename__ = "gold_prediction_snapshots"
    __table_args__ = (
        UniqueConstraint("run_id", "horizon_days", name="uq_prediction_run_horizon"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    target_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    current_price: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_price: Mapped[float] = mapped_column(Float, nullable=False)
    expected_return_pct: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reliability: Mapped[float] = mapped_column(Float, nullable=False)
    method: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    score_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    score_source: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    input_summary_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="PREDICTOR", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class GoldPredictionEvaluation(Base):
    """预测到期后的真实价格比对与误差记录。"""
    __tablename__ = "gold_prediction_evaluations"
    __table_args__ = (
        UniqueConstraint("prediction_id", name="uq_prediction_evaluation_prediction"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    prediction_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    actual_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    actual_price: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_price: Mapped[float] = mapped_column(Float, nullable=False)
    error_price: Mapped[float] = mapped_column(Float, nullable=False)
    abs_error_price: Mapped[float] = mapped_column(Float, nullable=False)
    abs_pct_error: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted_return_pct: Mapped[float] = mapped_column(Float, nullable=False)
    actual_return_pct: Mapped[float] = mapped_column(Float, nullable=False)
    direction_hit: Mapped[bool] = mapped_column(Boolean, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="AUTO_EVAL", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class ChinaGoldPremium(Base):
    """中国黄金溢价（上海金 vs 国际金价差）。"""
    __tablename__ = "china_gold_premiums"
    __table_args__ = (UniqueConstraint("timestamp", name="uq_china_premium_timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    sge_price_cny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lbma_price_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usdcny: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    premium_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="SGE", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class CentralBankGold(Base):
    """央行黄金储备 / 购金数据（季度）。"""
    __tablename__ = "central_bank_gold"
    __table_args__ = (UniqueConstraint("country", "period", name="uq_cb_gold_country_period"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    country: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    period: Mapped[str] = mapped_column(String(16), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    reserves_tonnes: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    net_change_tonnes: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="WGC", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class NewsSentiment(Base):
    """黄金相关新闻情绪评分。"""
    __tablename__ = "news_sentiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    source_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False)
    relevance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="GDELT", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class GoldPrice(Base):
    """每日金价（Sina/Yahoo 等免费行情源 / 替代 FRED 已下线序列）。"""
    __tablename__ = "gold_prices"
    __table_args__ = (UniqueConstraint("date", name="uq_gold_price_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    open: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="SINA", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class IntradaySnapshot(Base):
    """日内金价快照，用于 24 小时走势图。每分钟记录一个点。"""
    __tablename__ = "intraday_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="SINA", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
