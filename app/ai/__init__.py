"""DeepSeek AI 黄金市场分析师。

不替代量化评分引擎，而是叠加一层定性分析：
- 解读 25 因子评分结果
- 识别核心驱动因子和矛盾信号
- 生成风险提示和数据质量备注
- 提供多周期观点综合

用法：
    from app.ai import analyze_score, analyze_latest_score, AIAnalysis

    analysis = analyze_score(db, score_snapshot)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

# Python 3.9 + LibreSSL 兼容性
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from app.config import get_settings
from app.models import GoldScoreSnapshot

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 30
MAX_TOKENS = 1536
TEMPERATURE = 0.3


@dataclass
class AIAnalysis:
    """AI 分析结果。"""
    timestamp: datetime
    score_snapshot_id: int | None
    model: str
    overview: str  # 市场概览
    drivers: list[dict[str, str]]  # [{factor, impact, reason}]
    contradictions: list[str]  # 矛盾信号
    risks: list[str]  # 风险提示
    quality_notes: list[str]  # 数据质量备注
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "overview": self.overview,
            "drivers": self.drivers,
            "contradictions": self.contradictions,
            "risks": self.risks,
            "quality_notes": self.quality_notes,
        }


# ── 因子分组 ──────────────────────────────────────────────────────

_SHORT_TERM_FACTORS = {
    "黄金趋势", "短期动量", "避险情绪", "GLD ETF", "矿业股GDX",
    "白银/黄金比", "搜索热度", "新闻情绪",
}

_MID_TERM_FACTORS = {
    "实际利率", "实际利率曲线", "名义利率", "联邦基金", "美元指数",
    "期限溢价", "通胀预期", "美元流动性", "CFTC投机仓位", "美股分流",
    "铜/金比", "原油WTI", "美元人民币", "中国溢价", "COMEX库存",
    "COMEX期限结构", "期权隐波偏度",
}

_LONG_TERM_FACTORS = {
    "财政压力", "央行购金", "ETF资金流", "地缘风险", "实物需求",
}


# ── 提示词构建 ────────────────────────────────────────────────────

def _build_prompt(snapshot: GoldScoreSnapshot) -> str:
    """构建发给 DeepSeek 的分析提示词。"""
    # 解析因子分
    factor_data: dict[str, Any] = {}
    try:
        factor_data = json.loads(snapshot.factor_scores or "{}")
    except json.JSONDecodeError:
        pass

    scores = factor_data.get("scores", {})
    details = factor_data.get("details", {})

    # 多周期贡献
    horizon_detail = details.get("多周期评分", {})

    # 风险标志
    risk_flags: list[str] = []
    try:
        risk_flags = json.loads(snapshot.risk_flags or "[]")
    except json.JSONDecodeError:
        pass

    # 排名 top/bottom 因子
    sorted_factors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_factors[:5]
    bottom_5 = sorted_factors[-5:]

    # 按分组归类
    short_factors = {k: v for k, v in scores.items() if k in _SHORT_TERM_FACTORS}
    mid_factors = {k: v for k, v in scores.items() if k in _MID_TERM_FACTORS}
    long_factors = {k: v for k, v in scores.items() if k in _LONG_TERM_FACTORS}

    return f"""你是一位专业的黄金市场量化分析师。请基于以下量化评分数据，提供简洁的市场解读。

⚠️ 重要：只解读数据，不给出投资建议。用中文回复。

## 评分概览
- 总分：{snapshot.total_score:+.1f}（范围 -100 ~ +100）
- 方向：{snapshot.direction}
- 评分来源：{snapshot.source}
- 时间：{snapshot.timestamp}

## 多周期分解
短线动量贡献：{horizon_detail.get('短线动量_贡献分', 'N/A')}
中期宏观贡献：{horizon_detail.get('中期宏观_贡献分', 'N/A')}
长期结构贡献：{horizon_detail.get('长期结构_贡献分', 'N/A')}

## 利多因子（Top 5）
{json.dumps(dict(top_5), ensure_ascii=False, indent=2)}

## 利空因子（Bottom 5）
{json.dumps(dict(bottom_5), ensure_ascii=False, indent=2)}

## 短线动量因子
{json.dumps(short_factors, ensure_ascii=False, indent=2)}

## 中期宏观因子
{json.dumps(mid_factors, ensure_ascii=False, indent=2)}

## 长期结构因子
{json.dumps(long_factors, ensure_ascii=False, indent=2)}

## 风险标志
{json.dumps(risk_flags, ensure_ascii=False, indent=2)}

---

请按以下 JSON 格式回复（只返回 JSON，不要其他文字）：

{{
  "overview": "2-3 句话概括当前黄金市场状态",
  "drivers": [
    {{"factor": "因子名", "impact": "利多/利空/中性", "reason": "一句话解释"}}
  ],
  "contradictions": ["如果短线和中长期信号矛盾，列出来；无则空数组"],
  "risks": ["基于风险标志的关键风险提示"],
  "quality_notes": ["数据质量相关备注"]
}}"""


# ── API 调用 ──────────────────────────────────────────────────────

def _call_deepseek(prompt: str) -> dict[str, Any] | None:
    """调用 DeepSeek API 进行文本分析。"""
    settings = get_settings()
    api_key = settings.deepseek_api_key
    base_url = settings.deepseek_base_url or "https://api.deepseek.com"
    model = settings.deepseek_model

    if not api_key:
        logger.info("DEEPSEEK_API_KEY not configured, skipping AI analysis")
        return None

    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一位黄金市场量化分析师。只基于提供的量化数据做解读，"
                    "不编造信息，不给出投资建议。始终以 JSON 格式回复。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }

    resp = None
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SECONDS, verify=False)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except json.JSONDecodeError:
        raw_text = resp.text if resp else ""
        match = re.search(r'\{[\s\S]*\}', raw_text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.warning("DeepSeek returned non-JSON response")
        return None
    except (requests.RequestException, KeyError, IndexError) as e:
        logger.warning("DeepSeek API call failed: %s", e)
        return None


# ── 公开接口 ──────────────────────────────────────────────────────

def analyze_score(db: Session, snapshot: GoldScoreSnapshot) -> AIAnalysis | None:
    """对一次评分快照进行 AI 分析。

    Args:
        db: 数据库会话
        snapshot: 评分快照

    Returns:
        AIAnalysis 或 None（API 未配置/调用失败时）
    """
    prompt = _build_prompt(snapshot)
    result = _call_deepseek(prompt)

    if result is None:
        return None

    return AIAnalysis(
        timestamp=datetime.now(timezone.utc),
        score_snapshot_id=int(snapshot.id) if snapshot.id else None,
        model=get_settings().deepseek_model,
        overview=result.get("overview", ""),
        drivers=result.get("drivers", []),
        contradictions=result.get("contradictions", []),
        risks=result.get("risks", []),
        quality_notes=result.get("quality_notes", []),
        raw_response=json.dumps(result, ensure_ascii=False),
    )


def analyze_latest_score(db: Session) -> dict[str, Any] | None:
    """分析最新的评分快照，带缓存（避免频繁调用 API）。

    Returns:
        dict with ok/error + analysis, or None
    """
    snapshot = db.scalar(
        select(GoldScoreSnapshot).order_by(GoldScoreSnapshot.timestamp.desc())
    )
    if snapshot is None:
        return {"ok": False, "error": "No score snapshot available"}

    analysis = analyze_score(db, snapshot)
    if analysis is None:
        return {"ok": False, "error": "AI analysis unavailable (API key not configured or call failed)"}

    return {
        "ok": True,
        "score_id": snapshot.id,
        "score_total": snapshot.total_score,
        "score_direction": snapshot.direction,
        "analysis": analysis.to_dict(),
    }
