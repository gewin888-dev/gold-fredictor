from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    fred_api_key: str = Field(default="", alias="FRED_API_KEY")
    database_url: str = Field(default="sqlite:///./gold_monitor.db", alias="DATABASE_URL")
    feishu_webhook_url: str = Field(default="", alias="FEISHU_WEBHOOK_URL")
    feishu_secret: str = Field(default="", alias="FEISHU_SECRET")
    newsapi_key: str = Field(default="", alias="NEWSAPI_KEY")
    newsapi_daily_limit: int = Field(default=100, alias="NEWSAPI_DAILY_LIMIT")
    newsapi_timeout_seconds: int = Field(default=12, alias="NEWSAPI_TIMEOUT_SECONDS")
    fred_observation_start: str = Field(default="2018-01-01", alias="FRED_OBSERVATION_START")
    fred_verify_ssl: bool = Field(default=True, alias="FRED_VERIFY_SSL")
    auto_start_scheduler: bool = Field(default=False, alias="AUTO_START_SCHEDULER")
    auto_bootstrap_data: bool = Field(default=False, alias="AUTO_BOOTSTRAP_DATA")
    production_mode: bool = Field(default=True, alias="PRODUCTION_MODE")
    show_low_confidence_data: bool = Field(default=False, alias="SHOW_LOW_CONFIDENCE_DATA")
    prediction_score_sources: str = Field(
        default="backfill_real_v2,rule_v2",
        alias="PREDICTION_SCORE_SOURCES",
    )
    auto_optimize_score_params: bool = Field(default=False, alias="AUTO_OPTIMIZE_SCORE_PARAMS")
    auto_activate_optimized_params: bool = Field(default=False, alias="AUTO_ACTIVATE_OPTIMIZED_PARAMS")
    auto_optimize_min_hit_rate: float = Field(default=0.55, alias="AUTO_OPTIMIZE_MIN_HIT_RATE")
    auto_optimize_n_iter: int = Field(default=80, alias="AUTO_OPTIMIZE_N_ITER")
    auto_optimize_horizon_days: int = Field(default=20, alias="AUTO_OPTIMIZE_HORIZON_DAYS")
    auto_optimize_prediction_model: bool = Field(default=False, alias="AUTO_OPTIMIZE_PREDICTION_MODEL")
    auto_activate_prediction_model: bool = Field(default=False, alias="AUTO_ACTIVATE_PREDICTION_MODEL")
    auto_prediction_n_iter: int = Field(default=80, alias="AUTO_PREDICTION_N_ITER")
    auto_prediction_min_score: float = Field(default=40.0, alias="AUTO_PREDICTION_MIN_SCORE")
    auto_prediction_max_mape_pct: float = Field(default=8.0, alias="AUTO_PREDICTION_MAX_MAPE_PCT")
    auto_prediction_min_direction_accuracy: float = Field(default=0.52, alias="AUTO_PREDICTION_MIN_DIRECTION_ACCURACY")
    auto_prediction_min_samples: int = Field(default=120, alias="AUTO_PREDICTION_MIN_SAMPLES")
    auto_prediction_min_valid_horizons: int = Field(default=3, alias="AUTO_PREDICTION_MIN_VALID_HORIZONS")
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    api_key: str = Field(default="", alias="API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    model_config = SettingsConfigDict(env_file=str(ROOT_DIR / ".env"), extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configured_prediction_sources() -> set[str]:
    settings = get_settings()
    return {
        item.strip()
        for item in settings.prediction_score_sources.split(",")
        if item.strip()
    }
