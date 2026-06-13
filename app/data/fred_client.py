from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import requests

from app.config import get_settings


FRED_BASE_URL = "https://api.stlouisfed.org/fred"


@dataclass(frozen=True)
class FredSeriesConfig:
    series_id: str
    name: str
    unit: str = ""
    frequency: str = ""


FRED_SERIES: list[FredSeriesConfig] = [
    FredSeriesConfig("DGS10", "美国 10 年期国债收益率", "percent", "daily"),
    FredSeriesConfig("DFII10", "美国 10 年期 TIPS 实际收益率", "percent", "daily"),
    FredSeriesConfig("T10YIE", "10 年通胀预期", "percent", "daily"),
    FredSeriesConfig("FEDFUNDS", "联邦基金利率", "percent", "monthly"),
    FredSeriesConfig("VIXCLS", "VIX 恐慌指数", "index", "daily"),
    FredSeriesConfig("DTWEXBGS", "美元广义贸易加权指数", "index", "daily"),
    FredSeriesConfig("SP500", "标普 500 指数", "index", "daily"),
]


class FredClient:
    def __init__(self, api_key: str | None = None, timeout: int = 30) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.fred_api_key
        self.timeout = timeout

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("FRED_API_KEY is not configured. Add it to .env first.")

        response = requests.get(
            f"{FRED_BASE_URL}/{path}",
            params={**params, "api_key": self.api_key, "file_type": "json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if "error_message" in payload:
            raise RuntimeError(payload["error_message"])
        return payload

    def get_observations(
        self,
        series_id: str,
        observation_start: str | date | None = None,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {"series_id": series_id, "sort_order": "asc"}
        if observation_start:
            params["observation_start"] = str(observation_start)

        payload = self._get("series/observations", params)
        rows = payload.get("observations", [])
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "value"])

        df = df[["date", "value"]].rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
        df = df.dropna(subset=["value"])
        return df
