"""AI 对话窗口 — DB持久化 + 归档 + 全系统深度融合。

特性：
- 对话记录自动存入 SQLite（chat_sessions + chat_messages 表）
- 支持多会话存档和回溯
- AI 可访问全部系统数据：实时金价、评分、预测、宏观、CFTC、情绪等
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
API_MAX_TOKENS = 4096  # 增大以容纳更丰富的回复
API_TEMP = 0.5
MAX_HISTORY_TURNS = 30
CONTEXT_CACHE_TTL = 30  # 系统上下文缓存秒数

_context_cache: dict = {}
_context_cache_time: float = 0.0


# ==================== 系统提示词 ====================

BASE_SYSTEM_PROMPT = """你是黄金走势监控与预测系统的 AI 助手，拥有对系统全部数据的访问权限。

你可以看到的数据和对应的分析能力：

1. **实时金价** — COMEX 黄金期货 + 沪金连续实时报价、日内高低、涨跌幅
2. **25 因子评分** — 多空评分总分、方向、短线/中期/长期因子贡献、完整因子分
3. **价格预测** — 未来 1/7/30/90/180/360 天的金价预测及历史准确率
4. **宏观指标** — 实际利率、名义利率、美元指数、VIX、通胀预期等最新值
5. **CFTC 持仓** — 投机净多仓、商业净空仓、总持仓量
6. **市场情绪** — 最新新闻情绪评分和标题
7. **中国溢价** — 上海金对 COMEX 的溢价/折价
8. **央行购金** — 各国央行月度购金数据
9. **系统健康** — 各数据源的记录数、新鲜度、采集器状态

工作方式：
- 用户问任何问题，先对照系统数据给出基于事实的回答
- 市场分析：结合实时金价和因子评分，指出核心驱动和矛盾信号
- 预测解读：解释当前预测的逻辑和置信度
- 系统诊断：检查数据新鲜度和采集器状态，定位问题
- 始终用中文回复，简洁专业，客观中立

重要规则：
- 只做分析参考，不给出投资建议
- 回答应基于提供的系统数据，不编造信息
- 不清楚的地方明确指出
- 预测和评分调整建议用"可能"、"预计"等措辞"""


# ==================== 全系统上下文构建 ====================

def _build_system_context(db: Session) -> str:
    """构建完整的系统数据上下文，供 AI 分析使用。缓存 30 秒。"""
    import time as _time
    global _context_cache, _context_cache_time
    now = _time.time()
    if _context_cache and now - _context_cache_time < CONTEXT_CACHE_TTL:
        return _context_cache.get("text", "")

    parts: list[str] = []
    now_utc = datetime.now(timezone.utc)

    # ── 1. 实时金价 ──
    try:
        from app.data.gold_price_collector import fetch_gold_price
        gold = fetch_gold_price(use_cache=False)
        if gold.get("ok"):
            parts.append(
                "## 实时金价\n\n"
                f"- COMEX 黄金：${gold['price']:,.2f}（昨收 ${gold.get('previous_close', 0):,.0f}）\n"
                f"- 涨跌：{gold.get('change', 0):+.2f}（{gold.get('change_pct', 0):+.2f}%）\n"
                f"- 日内高/低：${gold.get('day_high', 0):,.0f} / ${gold.get('day_low', 0):,.0f}\n"
                f"- 数据源：{gold.get('source', 'N/A')}（延迟 {gold.get('delay', 'N/A')}）\n"
                f"- 更新时间：{gold.get('timestamp', 'N/A')}"
            )

        # 沪金
        from app.data.gold_price_collector import _fetch_from_sina
        try:
            import requests as _req
            url = "https://hq.sinajs.cn/list=nf_AU0"
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}
            r = _req.get(url, headers=headers, timeout=8, verify=False)
            r.encoding = "gbk"
            fields = r.text.strip().split('"')[1].split(",")
            sh_price = float(fields[5]) if len(fields) > 5 else None
            if sh_price and sh_price > 0:
                sh_high = float(fields[3]) if len(fields) > 3 else None
                sh_low = float(fields[4]) if len(fields) > 4 else None
                parts.append(f"## 沪金连续\n\n- 最新价：¥{sh_price:,.2f}/g\n- 日内高/低：{sh_high:,.0f}/{sh_low:,.0f}")
        except Exception:
            pass
    except Exception:
        parts.append("## 实时金价\n\n暂无法获取实时金价。")

    # ── 2. 评分快照 ──
    try:
        snapshot = db.scalar(select(GoldScoreSnapshot).order_by(GoldScoreSnapshot.timestamp.desc()))
        if snapshot:
            factor_data = {}
            try:
                factor_data = json.loads(snapshot.factor_scores or "{}")
            except json.JSONDecodeError:
                pass
            scores = factor_data.get("scores", {})
            details = factor_data.get("details", {})
            horizon = details.get("多周期评分", {})
            risk_flags = []
            try:
                risk_flags = json.loads(snapshot.risk_flags or "[]")
            except json.JSONDecodeError:
                pass

            top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            bottom_5 = sorted(scores.items(), key=lambda x: x[1])[:5]

            parts.append(
                "## 25 因子评分\n\n"
                f"- 总分：{snapshot.total_score:+.1f}（范围 -100 ~ +100）\n"
                f"- 方向：{snapshot.direction}\n"
                f"- 评分时间：{snapshot.timestamp}\n"
                f"- 短线动量贡献：{horizon.get('短线动量_贡献分', 'N/A')}\n"
                f"- 中期宏观贡献：{horizon.get('中期宏观_贡献分', 'N/A')}\n"
                f"- 长期结构贡献：{horizon.get('长期结构_贡献分', 'N/A')}\n\n"
                f"利多 Top 5：{json.dumps(dict(top_5), ensure_ascii=False)}\n"
                f"利空 Top 5：{json.dumps(dict(bottom_5), ensure_ascii=False)}\n"
                f"风险标志：{json.dumps(risk_flags, ensure_ascii=False)}\n"
                f"完整因子分：{json.dumps(scores, ensure_ascii=False)}"
            )
        else:
            parts.append("## 评分\n\n暂无评分数据。")
    except Exception:
        parts.append("## 评分\n\n评分数据读取失败。")

    # ── 3. 预测数据 ──
    try:
        from app.models import GoldPredictionSnapshot, PredictionModelVersion
        pred = db.scalar(select(GoldPredictionSnapshot).order_by(GoldPredictionSnapshot.timestamp.desc()))
        active_model = db.scalar(select(PredictionModelVersion).where(PredictionModelVersion.is_active == True))
        if pred:
            pred_data = {
                "预测时间": str(pred.timestamp),
                "当前价格": pred.current_price,
                "模型版本": pred.model_version,
            }
            for h in [1, 7, 30, 90, 180, 360]:
                key = f"horizon_{h}d_price"
                val = getattr(pred, key, None)
                if val is not None:
                    pred_data[f"{h}天预测价"] = val
            model_info = ""
            if active_model:
                model_info = (
                    f"\n\n激活模型：{active_model.version}（{active_model.method}）\n"
                    f"方向准确率：{(active_model.direction_accuracy or 0)*100:.0f}%（{active_model.evaluated_count or 0}条评估）\n"
                    f"MAPE：{active_model.mape_price_pct or 0:.1f}%"
                )
            parts.append(f"## 价格预测\n\n{json.dumps(pred_data, ensure_ascii=False)}{model_info}")
    except Exception:
        parts.append("## 价格预测\n\n预测数据暂不可用。")

    # ── 4. 宏观指标快照 ──
    try:
        from app.models import MacroObservation, MacroSeries
        key_series = ["DFII10", "DGS10", "T10YIE", "DTWEXBGS", "VIXCLS", "FEDFUNDS"]
        key_labels = {"DFII10": "实际利率", "DGS10": "10年美债", "T10YIE": "通胀预期",
                       "DTWEXBGS": "美元指数", "VIXCLS": "VIX", "FEDFUNDS": "联邦基金利率"}
        macro_lines = []
        for sid in key_series:
            row = db.scalar(select(MacroObservation).where(MacroObservation.series_id == sid).order_by(MacroObservation.timestamp.desc()))
            if row and row.value is not None:
                label = key_labels.get(sid, sid)
                macro_lines.append(f"- {label}：{row.value:.4f}（{row.timestamp.strftime('%Y-%m-%d') if row.timestamp else '?'}）")
        if macro_lines:
            parts.append("## 宏观指标\n\n" + "\n".join(macro_lines))
    except Exception:
        pass

    # ── 5. CFTC 持仓 ──
    try:
        from app.models import CftcPosition
        cftc = db.scalar(select(CftcPosition).order_by(CftcPosition.timestamp.desc()))
        if cftc:
            parts.append(
                "## CFTC 持仓\n\n"
                f"- 报告日期：{cftc.timestamp}\n"
                f"- 投机净多仓：{cftc.noncommercial_net:,}\n"
                f"- 投机多头：{cftc.noncommercial_long:,} / 空头：{cftc.noncommercial_short:,}\n"
                f"- 商业净空仓：{cftc.commercial_short - cftc.commercial_long:,}\n"
                f"- 总持仓：{cftc.open_interest:,}"
            )
    except Exception:
        pass

    # ── 6. 新闻情绪 ──
    try:
        from app.models import NewsSentiment
        ns = db.scalar(select(NewsSentiment).order_by(NewsSentiment.timestamp.desc()))
        if ns:
            parts.append(
                "## 新闻情绪\n\n"
                f"- 最新情绪分：{ns.sentiment_score:.2f}（范围 -1 ~ +1）\n"
                f"- 标题：{ns.title or '(无)'}\n"
                f"- 时间：{ns.timestamp}\n"
                f"- 来源：{ns.source}"
            )
    except Exception:
        pass

    # ── 7. 中国溢价 ──
    try:
        from app.models import ChinaGoldPremium
        prem = db.scalar(select(ChinaGoldPremium).order_by(ChinaGoldPremium.timestamp.desc()))
        if prem:
            parts.append(
                "## 中国溢价\n\n"
                f"- 溢价率：{prem.premium_pct:+.2f}%\n"
                f"- 时间：{prem.timestamp}\n"
                f"- 来源：{prem.source}"
            )
    except Exception:
        pass

    # ── 8. 数据新鲜度 ──
    try:
        from app.models import GoldPrice, CentralBankGold
        gp = db.scalar(select(GoldPrice).order_by(GoldPrice.date.desc()))
        cbg = db.scalar(select(CentralBankGold).order_by(CentralBankGold.timestamp.desc()))
        lines = []
        if gp:
            age_h = (now_utc - gp.updated_at.replace(tzinfo=None)).total_seconds() / 3600 if gp.updated_at and gp.updated_at.tzinfo is None else None
            lines.append(f"- 金价日线：{gp.date}，{age_h:.1f}h前" if age_h else f"- 金价日线：{gp.date}")
        if cbg:
            lines.append(f"- 央行购金：{cbg.timestamp}")
        if lines:
            parts.append("## 数据新鲜度\n\n" + "\n".join(lines))
    except Exception:
        pass

    text = "\n\n".join(parts)
    _context_cache = {"text": text}
    _context_cache_time = _time.time()
    return text


# ==================== DeepSeek 调用 ====================

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
    """发送消息，获取 AI 回复。每次注入全系统上下文，对话自动持久化。"""
    _ensure_session(db, session_id)
    history = _load_history(db, session_id)

    # 构建完整系统上下文
    system_ctx = _build_system_context(db)

    # 组装消息：系统提示 → 全系统数据 → 历史 → 用户输入
    messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
    messages.append({"role": "system", "content": "## 当前系统数据\n\n" + system_ctx})

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
