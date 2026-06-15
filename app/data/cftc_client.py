from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO

import requests


CFTC_LEGACY_FUTURES_ONLY_URL = "https://www.cftc.gov/dea/newcot/deafut.txt"
GOLD_CONTRACT_MARKET_CODE = "088691"


@dataclass(frozen=True)
class CftcPositionRecord:
    market_name: str
    contract_market_code: str
    exchange_code: str
    timestamp: datetime
    open_interest: int
    noncommercial_long: int
    noncommercial_short: int
    noncommercial_spreading: int
    commercial_long: int
    commercial_short: int

    @property
    def noncommercial_net(self) -> int:
        return self.noncommercial_long - self.noncommercial_short


def _to_int(value: str) -> int:
    return int(value.strip())


def parse_legacy_futures_only(text: str, contract_market_code: str = GOLD_CONTRACT_MARKET_CODE) -> CftcPositionRecord:
    for row in csv.reader(StringIO(text)):
        if len(row) < 13:
            continue
        if row[3].strip() != contract_market_code:
            continue
        report_date = datetime.strptime(row[2].strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return CftcPositionRecord(
            market_name=row[0].strip(),
            contract_market_code=row[3].strip(),
            exchange_code=row[4].strip(),
            timestamp=report_date,
            open_interest=_to_int(row[7]),
            noncommercial_long=_to_int(row[8]),
            noncommercial_short=_to_int(row[9]),
            noncommercial_spreading=_to_int(row[10]),
            commercial_long=_to_int(row[11]),
            commercial_short=_to_int(row[12]),
        )
    raise ValueError(f"CFTC contract_market_code {contract_market_code} not found in legacy futures-only report.")


class CftcClient:
    def __init__(self, url: str = CFTC_LEGACY_FUTURES_ONLY_URL, timeout: int = 30) -> None:
        self.url = url
        self.timeout = timeout

    def fetch_current_gold_position(self) -> CftcPositionRecord:
        import sys
        try:
            response = requests.get(self.url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = f"CFTC API 请求失败: url={self.url}, timeout={self.timeout}s, error={type(exc).__name__}: {exc}"
            print(detail, file=sys.stderr)
            raise
        return parse_legacy_futures_only(response.text)
