from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import requests
import urllib3

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
    FredSeriesConfig("DFII5", "美国 5 年期 TIPS 实际收益率", "percent", "daily"),
    FredSeriesConfig("DFII10", "美国 10 年期 TIPS 实际收益率", "percent", "daily"),
    FredSeriesConfig("DFII30", "美国 30 年期 TIPS 实际收益率", "percent", "daily"),
    FredSeriesConfig("T10YIE", "10 年通胀预期", "percent", "daily"),
    FredSeriesConfig("FEDFUNDS", "联邦基金利率", "percent", "monthly"),
    FredSeriesConfig("VIXCLS", "VIX 恐慌指数", "index", "daily"),
    FredSeriesConfig("DTWEXBGS", "美元广义贸易加权指数", "index", "daily"),
    FredSeriesConfig("SP500", "标普 500 指数", "index", "daily"),
    FredSeriesConfig("GVZCLS", "CBOE 黄金 ETF 波动率指数", "index", "daily"),
    FredSeriesConfig("THREEFYTP10", "10 年期美债期限溢价", "percent", "daily"),
    FredSeriesConfig("WALCL", "美联储总资产", "millions_usd", "weekly"),
    FredSeriesConfig("WDTGAL", "美国财政部一般账户余额", "millions_usd", "weekly"),
    FredSeriesConfig("RRPONTSYD", "隔夜逆回购使用量", "billions_usd", "daily"),
    FredSeriesConfig("WRESBAL", "存款机构准备金余额", "billions_usd", "weekly"),
    FredSeriesConfig("FYFSD", "美国联邦财政盈余/赤字", "millions_usd", "annual"),
    FredSeriesConfig("GFDEGDQ188S", "美国联邦债务/GDP", "percent", "quarterly"),
    FredSeriesConfig("CPIAUCSL", "美国 CPI 消费者价格指数", "index", "monthly"),
]


class FredClient:
    def __init__(self, api_key: str | None = None, timeout: int = 10) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.fred_api_key
        self.timeout = timeout
        self.verify_ssl = bool(settings.fred_verify_ssl)
        # Python 3.9 + LibreSSL 兼容性
        urllib3.disable_warnings()

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("FRED_API_KEY is not configured. Add it to .env first.")
        request_kwargs = {
            "params": {**params, "api_key": self.api_key, "file_type": "json"},
            "timeout": self.timeout,
            "verify": self.verify_ssl,
        }
        if self.verify_ssl:
            response = requests.get(f"{FRED_BASE_URL}/{path}", **request_kwargs)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", urllib3.exceptions.InsecureRequestWarning)
                response = requests.get(f"{FRED_BASE_URL}/{path}", **request_kwargs)
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
