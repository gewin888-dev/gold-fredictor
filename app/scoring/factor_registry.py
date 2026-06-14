"""评分因子注册表。

先统一 UI、外部指标录入和优化报告的因子元数据；后续评分计算函数会继续向这里迁移。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    group: str
    description: str
    source: str = ""
    scored: bool = True
    optimizable: bool = False
    external_indicator_id: str | None = None
    inactive_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


FACTOR_DEFINITIONS: tuple[FactorDefinition, ...] = (
    FactorDefinition("黄金趋势", "短期动量", "价格在 MA20/MA60 上方=多头趋势；下方=空头。", "GoldPrice", True, True),
    FactorDefinition("短期动量", "短期动量", "近 3 日金价涨跌幅，涨=利多、跌=利空。", "GoldPrice", True, False),
    FactorDefinition("避险情绪", "短期动量", "VIX 上升代表风险厌恶上升，通常增加黄金避险需求。", "FRED:VIXCLS", True, True),
    FactorDefinition("GLD ETF", "短期动量", "GLD 价格和资金流代理黄金 ETF 配置需求。", "SINA/SPDR", True, False),
    FactorDefinition("矿业股GDX", "短期动量", "金矿股走势可作为黄金权益链条的风险偏好代理。", "SINA", True, False),
    FactorDefinition("白银/黄金比", "短期动量", "白银相对黄金走强偏 risk-on，走弱偏避险。", "SINA", True, False),
    FactorDefinition("搜索热度", "短期动量", "搜索热度异常升高可能代表短期关注度和拥挤度。", "GOOGLE_TRENDS", True, False),
    FactorDefinition("新闻情绪", "短期动量", "黄金相关新闻情绪变化影响短期风险偏好。", "NEWSAPI/GDELT", True, False),
    FactorDefinition("实际利率", "中期宏观", "TIPS 实际利率上升提高持有黄金机会成本，通常利空黄金。", "FRED:DFII10", True, True),
    FactorDefinition("实际利率曲线", "中期宏观", "5Y/10Y/30Y TIPS 曲线综合变化，实际利率整体下行通常利多。", "FRED", True, True),
    FactorDefinition("名义利率", "中期宏观", "10Y 美债收益率上升提高债券吸引力，通常压制黄金。", "FRED:DGS10", True, True),
    FactorDefinition("联邦基金", "中期宏观", "政策利率上行代表紧缩环境，通常利空黄金。", "FRED:FEDFUNDS", True, True),
    FactorDefinition("美元指数", "中期宏观", "美元走强使黄金对非美买家更贵，通常利空。", "FRED:DTWEXBGS", True, True),
    FactorDefinition("期限溢价", "中期宏观", "期限溢价上升反映财政/期限风险补偿，可能增强黄金配置需求。", "FRED:THREEFYTP10", True, True),
    FactorDefinition("通胀预期", "中期宏观", "通胀预期上升增强保值需求，但需与利率同看。", "FRED:T10YIE", True, True),
    FactorDefinition("美元流动性", "中期宏观", "Fed资产负债表、准备金、TGA、RRP 综合代理美元流动性。", "FRED", True, True),
    FactorDefinition("CFTC投机仓位", "中期宏观", "非商业净多占比反映投机资金方向和拥挤度。", "CFTC", True, True),
    FactorDefinition("美股分流", "中期宏观", "美股强势可能分流避险资产需求。", "SINA", True, False),
    FactorDefinition("铜/金比", "中期宏观", "铜强金弱偏 risk-on；铜弱金强偏避险。", "SINA", True, False),
    FactorDefinition("原油WTI", "中期宏观", "油价影响通胀预期和宏观风险，需与利率因子联动解释。", "SINA", True, False),
    FactorDefinition("美元人民币", "中期宏观", "人民币汇率影响中国本地金价和溢价。", "SINA/FRED", True, False),
    FactorDefinition("中国溢价", "中期宏观", "中国本地金价溢价高通常代表本地需求强。", "SINA/SGE", True, False),
    FactorDefinition(
        "COMEX库存", "中期宏观", "COMEX 注册库存反映可交割供给压力，需结合期限结构解释。",
        "MANUAL/CME", False, False, "COMEX_REGISTERED_GOLD_OZ",
        "CME 官方库存数据当前未配置稳定免费接口；可接入授权源或手动录入后再评分。",
    ),
    FactorDefinition(
        "COMEX期限结构", "中期宏观", "近远月升贴水反映资金成本、现货紧张度和交割压力。",
        "MANUAL/CME", False, False, "COMEX_GOLD_FRONT_SPREAD_PCT",
        "需要稳定期货曲线数据源；当前可手动维护，完善后再评分。",
    ),
    FactorDefinition("期权隐波偏度", "中期宏观", "期权隐含波动率和偏度反映极端行情定价。", "FRED/CBOE/MANUAL", True, True),
    FactorDefinition("财政压力", "长期宏观", "财政赤字和债务/GDP 压力可能增强长期避险配置需求。", "FRED", True, True),
    FactorDefinition("央行购金", "长期宏观", "全球央行净购金形成长期需求支撑。", "WGC/IMF", True, False),
    FactorDefinition("ETF资金流", "长期宏观", "黄金 ETF 持仓/资金流入为正偏利多，流出偏利空。", "SPDR/MANUAL", True, True, "GLD_FLOW_TONNES"),
    FactorDefinition(
        "地缘风险", "长期宏观", "地缘事件强度和持续时间上升通常增加避险需求。",
        "MANUAL/NEWS", False, False, "GEO_RISK_INTENSITY",
        "尚未建立事件强度和持续时间模型；后续可用新闻事件或人工评分录入。",
    ),
    FactorDefinition(
        "实物需求", "长期宏观", "印度/中国节庆、进口、本地溢价等实物需求强弱。",
        "MANUAL", False, False, "INDIA_CHINA_PHYSICAL_DEMAND",
        "中印实物需求需要进口、节庆和本地溢价数据；当前可手动录入。",
    ),
)

FACTOR_BY_NAME = {item.name: item for item in FACTOR_DEFINITIONS}


def factor_groups() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for item in FACTOR_DEFINITIONS:
        groups.setdefault(item.group, []).append(item.name)
    return groups


def factor_help() -> dict[str, str]:
    return {item.name: item.description for item in FACTOR_DEFINITIONS}


def inactive_factor_reasons() -> dict[str, str]:
    return {item.name: item.inactive_reason for item in FACTOR_DEFINITIONS if item.inactive_reason}


def scored_factor_names() -> set[str]:
    return {item.name for item in FACTOR_DEFINITIONS if item.scored}


def optimizable_factor_names() -> set[str]:
    return {item.name for item in FACTOR_DEFINITIONS if item.optimizable}


def is_scored_factor(name: str) -> bool:
    item = FACTOR_BY_NAME.get(name)
    return bool(item and item.scored)


def is_optimizable_factor(name: str) -> bool:
    item = FACTOR_BY_NAME.get(name)
    return bool(item and item.optimizable)


def factor_registry_payload() -> list[dict[str, object]]:
    return [item.to_dict() for item in FACTOR_DEFINITIONS]
