"""AI 对话窗口 — DB持久化 + 归档 + 系统维护。

特性：
- 对话记录自动存入 SQLite（chat_sessions + chat_messages 表）
- 支持多会话存档和回溯
- 系统维护模式：AI 可辅助诊断数据质量、调度状态等问题
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import requests
from sqlalchemy import func, select
from sqlalchemy.orm import Session

# Python 3.9 + LibreSSL 兼容性
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from app.config import get_settings
from app.models import ChatMessage, ChatSession, GoldScoreSnapshot

logger = logging.getLogger(__name__)

API_TIMEOUT = 45
API_MAX_TOKENS = 2048
API_TEMP = 0.5
MAX_HISTORY_TURNS = 30
HEALTH_CACHE_TTL = 30  # 系统健康状态缓存秒数

_health_cache: dict = {}
_health_cache_time: float = 0.0


# ==================== DeepSeek 调用 ====================

BASE_SYSTEM_PROMPT = """你是一位黄金市场量化分析师兼系统维护助手。

你的能力包括：
1. **市场分析**：结合量化评分数据，分析地缘/宏观事件对黄金走势的影响
2. **因子调整**：根据外部事件，给出各因子的定性影响和量化调整建议
3. **系统维护**：帮助诊断数据采集、评分引擎、调度器等模块的问题

工作方式：
- 用户提供事件 → 对照当前评分数据给出调整建议（含具体因子和幅度）
- 用户询问系统问题 → 给出诊断思路和排查步骤
- 始终用中文回复，简洁专业，客观中立

重要规则：
- 只做分析参考，不给出投资建议
- 不清楚的地方明确指出
- 评分调整建议用"可能"、"预计"等措辞"""


def _score_context(db: Session) -> str:
    """构建当前评分的自然语言上下文。"""
    snapshot = db.scalar(
        select(GoldScoreSnapshot).order_by(GoldScoreSnapshot.timestamp.desc())
    )
    if snapshot is None:
        return "当前暂无评分数据。"

    factor_data: dict = {}
    try:
        factor_data = json.loads(snapshot.factor_scores or "{}")
    except json.JSONDecodeError:
        pass

    scores = factor_data.get("scores", {})
    details = factor_data.get("details", {})
    horizon = details.get("多周期评分", {})

    sorted_f = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_f[:5]
    bottom_5 = sorted_f[-5:]

    return (
        "## 当前评分上下文\n\n"
        f"- 总分：{snapshot.total_score:+.1f}（范围-100~+100）\n"
        f"- 方向：{snapshot.direction}\n"
        f"- 时间：{snapshot.timestamp}\n"
        f"- 短线动量贡献：{horizon.get('短线动量_贡献分', 'N/A')}\n"
        f"- 中期宏观贡献：{horizon.get('中期宏观_贡献分', 'N/A')}\n"
        f"- 长期结构贡献：{horizon.get('长期结构_贡献分', 'N/A')}\n\n"
        f"利多 Top 5：{json.dumps(dict(top_5), ensure_ascii=False)}\n"
        f"利空 Top 5：{json.dumps(dict(bottom_5), ensure_ascii=False)}\n\n"
        f"完整因子分：{json.dumps(scores, ensure_ascii=False)}"
    )


def _system_health_context(db: Session) -> str:
    """收集系统健康状态信息（缓存 30 秒避免重复查询）。"""
    import time as _time
    global _health_cache, _health_cache_time
    now = _time.time()
    if _health_cache and now - _health_cache_time < HEALTH_CACHE_TTL:
        return _health_cache.get("text", "")

    from app.models import (
        CentralBankGold, CftcPosition, ChinaGoldPremium,
        GoldPrice, MacroObservation, NewsSentiment,
        GoldPredictionSnapshot, GoldPredictionEvaluation,
        ScoreParamsVersion,
    )

    rows = {}
    models = [
        ("macro", MacroObservation),
        ("gold_price", GoldPrice),
        ("cftc", CftcPosition),
        ("cb_gold", CentralBankGold),
        ("china_premium", ChinaGoldPremium),
        ("news_sentiment", NewsSentiment),
        ("score_snapshots", GoldScoreSnapshot),
        ("prediction_snapshots", GoldPredictionSnapshot),
        ("prediction_evaluations", GoldPredictionEvaluation),
        ("param_versions", ScoreParamsVersion),
    ]

    for name, model in models:
        try:
            count = db.scalar(select(func.count()).select_from(model))
            latest = db.scalar(select(model).order_by(model.updated_at.desc()))
            age = None
            if latest and latest.updated_at:
                age = (datetime.now(timezone.utc) - latest.updated_at).total_seconds() / 3600
            rows[name] = {"count": count or 0, "latest_age_hours": round(age, 1) if age else None}
        except Exception:
            rows[name] = {"count": 0, "latest_age_hours": None}

    direction_counts = {}
    try:
        dirs = db.scalars(select(GoldScoreSnapshot.direction).distinct()).all()
        for d in dirs:
            cnt = db.scalar(
                select(func.count()).select_from(GoldScoreSnapshot).where(
                    GoldScoreSnapshot.direction == d
                )
            )
            direction_counts[d] = cnt or 0
    except Exception:
        pass

    text = (
        "## 系统健康状态\n\n"
        f"数据表行数：{json.dumps(rows, ensure_ascii=False, indent=2)}\n\n"
        f"评分方向分布：{json.dumps(direction_counts, ensure_ascii=False)}"
    )
    _health_cache = {"text": text}
    _health_cache_time = _time.time()
    return text


def _call_deepseek(messages: list[dict]) -> str | None:
    settings = get_settings()
    api_key = settings.deepseek_api_key
    base_url = settings.deepseek_base_url or "https://api.deepseek.com"
    model = settings.deepseek_model

    if not api_key:
        return "DeepSeek API Key 未配置，请在 .env 中设置 DEEPSEEK_API_KEY。"

    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": API_TEMP,
        "max_tokens": API_MAX_TOKENS,
    }

    import time as _time
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=API_TIMEOUT, verify=False)
            if resp.status_code in (429, 503):
                wait = min(2 ** attempt, 8)
                logger.warning("DeepSeek rate-limited (HTTP %d), retry %d/3 after %ds", resp.status_code, attempt + 1, wait)
                _time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (429, 503):
                wait = min(2 ** attempt, 8)
                _time.sleep(wait)
                continue
            last_error = e
            break
        except Exception as e:
            last_error = e
            break

    logger.error("DeepSeek chat failed after retries: %s", last_error)
    if last_error and ("SSL" in str(last_error) or "SSLEOF" in str(last_error) or "timeout" in str(last_error).lower()):
        return "AI 服务暂时不可用，请稍后重试。如持续失败，请检查网络连接。"
    return f"AI 服务暂时不可用（{type(last_error).__name__ if last_error else 'unknown'}），请稍后重试。"


# ==================== DB 持久化 ====================

def _ensure_session(db: Session, session_id: str) -> ChatSession:
    row = db.scalar(select(ChatSession).where(ChatSession.session_id == session_id))
    if row is None:
        row = ChatSession(session_id=session_id, title="新对话", message_count=0)
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def _save_message(db: Session, session_id: str, role: str, content: str) -> None:
    msg = ChatMessage(session_id=session_id, role=role, content=content)
    db.add(msg)
    session = _ensure_session(db, session_id)
    session.message_count = (session.message_count or 0) + 1
    if session.title == "新对话" and role == "user":
        session.title = content[:80]
    db.commit()


def _load_history(db: Session, session_id: str, limit: int = 0) -> list[dict]:
    if limit <= 0:
        limit = MAX_HISTORY_TURNS * 2
    rows = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
    ).all()
    return [{"role": r.role, "content": r.content} for r in rows]


# ==================== 公开接口 ====================

def chat(db: Session, message: str, session_id: str = "default") -> dict[str, Any]:
    """发送消息，获取 AI 回复。对话自动持久化到 DB。"""
    _ensure_session(db, session_id)
    score_ctx = _score_context(db)
    history = _load_history(db, session_id)

    maintenance_kw = ["系统", "维护", "诊断", "健康", "数据采集", "调度",
                      "采集器", "错误", "报错", "crash", "挂了", "不工作",
                      "停了", "重启", "日志", "error", "bug"]
    is_maintenance = any(kw in message for kw in maintenance_kw)
    health_ctx = _system_health_context(db) if is_maintenance else ""

    messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]

    if not history:
        ctx = score_ctx
        if health_ctx:
            ctx += "\n\n" + health_ctx
        messages.append({"role": "system", "content": "当前数据：\n\n" + ctx})

    for entry in history[-MAX_HISTORY_TURNS * 2:]:
        messages.append(entry)

    messages.append({"role": "user", "content": message})

    _save_message(db, session_id, "user", message)

    reply = _call_deepseek(messages)
    if reply is None:
        return {"session_id": session_id, "reply": "AI 服务暂时不可用。",
                "messages_count": len(history) // 2}

    _save_message(db, session_id, "assistant", reply)

    return {
        "session_id": session_id,
        "reply": reply,
        "score_total": _latest_score(db),
        "score_direction": _latest_direction(db),
        "messages_count": (len(history) // 2) + 1,
    }


def reset_session(db: Session, session_id: str) -> None:
    """清空会话消息。"""
    msgs = db.scalars(select(ChatMessage).where(ChatMessage.session_id == session_id)).all()
    for m in msgs:
        db.delete(m)
    row = db.scalar(select(ChatSession).where(ChatSession.session_id == session_id))
    if row:
        row.message_count = 0
        row.title = "新对话"
    db.commit()


def get_history(db: Session, session_id: str) -> list[dict]:
    return _load_history(db, session_id)


def list_sessions(db: Session, limit: int = 50) -> list[dict]:
    """列出所有对话会话。"""
    rows = db.scalars(
        select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(limit)
    ).all()
    return [{
        "session_id": r.session_id,
        "title": r.title,
        "message_count": r.message_count,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    } for r in rows]


def get_session_messages(db: Session, session_id: str) -> list[dict]:
    """获取指定会话的完整消息列表。"""
    rows = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    ).all()
    return [{
        "role": r.role,
        "content": r.content,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    } for r in rows]


def delete_session(db: Session, session_id: str) -> bool:
    """删除指定会话及其所有消息。"""
    msgs = db.scalars(select(ChatMessage).where(ChatMessage.session_id == session_id)).all()
    for m in msgs:
        db.delete(m)
    row = db.scalar(select(ChatSession).where(ChatSession.session_id == session_id))
    if row:
        db.delete(row)
    db.commit()
    return True


def _latest_score(db: Session) -> float | None:
    row = db.scalar(select(GoldScoreSnapshot).order_by(GoldScoreSnapshot.timestamp.desc()))
    return row.total_score if row else None


def _latest_direction(db: Session) -> str:
    row = db.scalar(select(GoldScoreSnapshot).order_by(GoldScoreSnapshot.timestamp.desc()))
    return row.direction if row else "未知"
