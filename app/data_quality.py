from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceQuality:
    tier: str
    label: str
    can_score: bool


OFFICIAL_SOURCES = {"FRED", "CFTC", "WGC", "IMF", "SGE", "LBMA"}
DELAYED_FREE_SOURCES = {"YAHOO", "YAHOO FINANCE", "SINA", "SINA FINANCE", "GDELT", "NEWSAPI", "GOOGLE_TRENDS", "MANUAL_ESTIMATE", "MANUAL"}
TEST_SOURCES = {"TEST"}
PLACEHOLDER_SOURCES = {"SAMPLE", "ESTIMATE", "UNKNOWN"}
DERIVED_MODEL_PREFIXES = ("RULE_", "OPTIMIZED_", "AUTO_", "BACKFILL_", "SELF_HEALING")


def source_quality(source: str | None) -> SourceQuality:
    key = (source or "UNKNOWN").upper()
    if key in OFFICIAL_SOURCES:
        return SourceQuality("official", "官方/授权源", True)
    if key in DELAYED_FREE_SOURCES:
        return SourceQuality("delayed_free", "免费/延迟源", True)
    if key in TEST_SOURCES:
        return SourceQuality("test", "测试源", True)
    if key.startswith(DERIVED_MODEL_PREFIXES):
        return SourceQuality("derived_model", "模型派生", True)
    if key in PLACEHOLDER_SOURCES:
        return SourceQuality("placeholder", "样本/占位源", False)
    return SourceQuality("unknown", "未知源", False)
