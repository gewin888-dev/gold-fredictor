"""AI 对话窗口 — DB持久化 + 归档 + 全系统深度融合 + 主动智能化。

特性：
- 对话记录自动存入 SQLite（chat_sessions + chat_messages 表）
- 支持多会话存档和回溯
- AI 可访问全部系统数据：实时金价、评分、预测、宏观、CFTC、情绪等
- 主动分析：预计算异常检测、交叉验证、数据质量预警
- 自主操作：可触发评分重算、模型切换、数据采集等系统操作
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
# SSL: 使用 certifi CA bundle 替代 verify=False
import certifi

from app.config import get_settings
from app.models import ChatMessage, ChatSession, GoldScoreSnapshot

logger = logging.getLogger(__name__)

API_TIMEOUT = 45
API_MAX_TOKENS = 4096
API_TEMP = 0.5
MAX_HISTORY_TURNS = 30
CONTEXT_CACHE_TTL = 30

_context_cache: dict = {}
_context_cache_time: float = 0.0


# ==================== 系统提示词 ====================

BASE_SYSTEM_PROMPT = """你是黄金走势监控与预测系统的 AI 助手和主动协驾员。

你拥有对系统全部数据的访问权限，并且能够主动分析、预警、建议操作。

你的核心职责：

1. **主动分析** — 不等用户问，你看到数据后自己判断：
   - 金价剧烈波动但评分未更新 → 主动指出时间差和可能的影响
   - 多空因子严重矛盾 → 指出核心分歧和潜在方向
   - 数据源断流 → 立即报告哪些数据过期
   - 模型准确率下降 → 建议切换或重新训练

2. **决策支持** — 基于数据给出可执行的建议：
   - "当前评分已是25分钟前，金价已涨2.5%，建议重新评分"
   - "CFTC 投机净多占比52%，处于极端拥挤水平，注意回调风险"
   - "预测模型准确率仅47%，候选模型准确率60%，建议切换"

3. **自主操作** — 当用户同意时，可触发系统操作。
   如果你的建议后跟 [可执行]，表示可以立即操作：
   - [重新评分] [切换预测模型] [检查采集器] [刷新数据]

工作方式：
- 先给出分析 → 再给出建议 → 最后提供可执行的操作
- 始终用中文回复，简洁专业，客观中立
- 只做系统预测能力分析，不用于黄金买卖参考

重要规则：
- 回答应基于提供的系统数据，不编造信息
- 评分调整建议用"可能"、"预计"等措辞
- 数据异常时优先报告，而不是等用户发现"""


# ==================== 智能系统上下文 ====================

def _build_system_context(db: Session) -> str:
    """构建完整的系统数据上下文 + 预计算智能洞察。缓存 30 秒。"""
    import time as _time
    global _context_cache, _context_cache_time
    now = _time.time()
    if _context_cache and now - _context_cache_time < CONTEXT_CACHE_TTL:
        return _context_cache.get("text", "")

    parts: list[str] = []
    insights: list[str] = []  # 预计算洞察
    now_utc = datetime.now(timezone.utc)

    # ── 1. 实时金价 ──
    gold_price = None
    gold_change_pct = None
    gold_ts = None
    try:
        from app.data.gold_price_collector import fetch_gold_price
        gold = fetch_gold_price(use_cache=False)
        if gold.get("ok"):
            gold_price = gold.get("price")
            gold_change_pct = gold.get("change_pct")
            gold_ts = gold.get("timestamp")
            parts.append(
                "## 实时金价\n\n"
                f"- COMEX：${gold['price']:,.2f}（昨收 ${gold.get('previous_close', 0):,.0f}）\n"
                f"- 涨跌：{gold.get('change', 0):+.2f}（{gold.get('change_pct', 0):+.2f}%）\n"
                f"- 日内：高 ${gold.get('day_high', 0):,.0f} / 低 ${gold.get('day_low', 0):,.0f}\n"
                f"- 更新：{gold.get('timestamp', 'N/A')}"
            )
        # 沪金
        try:
            import requests as _req
            url = "https://hq.sinajs.cn/list=nf_AU0"
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}
            r = _req.get(url, headers=headers, timeout=8, verify=certifi.where())
            r.encoding = "gbk"
            fields = r.text.strip().split('"')[1].split(",")
            sh_price = float(fields[5]) if len(fields) > 5 else None
            if sh_price and sh_price > 0:
                sh_high = float(fields[3]) if len(fields) > 3 else None
                sh_low = float(fields[4]) if len(fields) > 4 else None
                parts.append(f"## 沪金连续\n\n- 最新：¥{sh_price:,.2f}/g\n- 日内：高{sh_high:,.0f} / 低{sh_low:,.0f}")
        except Exception:
            pass
    except Exception:
        parts.append("## 实时金价\n\n暂无法获取。")

    # ── 2. 评分快照 + 与金价交叉验证 ──
    score_total = None
    score_ts = None
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

            score_total = snapshot.total_score
            score_ts = snapshot.timestamp
            top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            bottom_5 = sorted(scores.items(), key=lambda x: x[1])[:5]

            parts.append(
                "## 25 因子评分\n\n"
                f"- 总分：{snapshot.total_score:+.1f} / 方向：{snapshot.direction}\n"
                f"- 时间：{snapshot.timestamp}\n"
                f"- 短线：{horizon.get('短线动量_贡献分', 'N/A')} | "
                f"中期：{horizon.get('中期宏观_贡献分', 'N/A')} | "
                f"长期：{horizon.get('长期结构_贡献分', 'N/A')}\n\n"
                f"利多 Top 5：{json.dumps(dict(top_5), ensure_ascii=False)}\n"
                f"利空 Top 5：{json.dumps(dict(bottom_5), ensure_ascii=False)}\n"
                f"风险标志：{json.dumps(risk_flags, ensure_ascii=False)}\n"
                f"完整因子：{json.dumps(scores, ensure_ascii=False)}"
            )

            # 智能洞察：评分与金价的时间差
            if gold_ts and score_ts and gold_change_pct is not None:
                try:
                    gold_dt = datetime.fromisoformat(gold_ts.replace("Z", "+00:00"))
                    age_min = (gold_dt.replace(tzinfo=None) - score_ts.replace(tzinfo=None)).total_seconds() / 60
                    if abs(gold_change_pct or 0) > 1.5 and age_min > 5:
                        insights.append(
                            f"⚠️ 金价已波动 {gold_change_pct:+.1f}%，但评分是 {age_min:.0f} 分钟前的，"
                            f"建议重新评分以反映最新行情。"
                        )
                except Exception:
                    pass

            # 智能洞察：多空矛盾
            if top_5 and bottom_5:
                max_long = top_5[0][1]
                max_short = bottom_5[0][1]
                if max_long > 5 and max_short < -3:
                    insights.append(
                        f"⚠️ 因子矛盾：最强利多「{top_5[0][0]}」({max_long:+.1f}) vs "
                        f"最强利空「{bottom_5[0][0]}」({max_short:+.1f})，市场存在分歧。"
                    )
        else:
            parts.append("## 评分\n\n暂无评分数据。")
    except Exception:
        parts.append("## 评分\n\n评分数据读取失败。")

    # ── 3. 预测数据 + 模型告警 ──
    try:
        from app.models import GoldPredictionSnapshot, PredictionModelVersion
        pred = db.scalar(select(GoldPredictionSnapshot).order_by(GoldPredictionSnapshot.timestamp.desc()))
        active_model = db.scalar(select(PredictionModelVersion).where(PredictionModelVersion.is_active == True))
        if pred:
            pred_data = {"当前价格": pred.current_price, "预测时间": str(pred.timestamp)}
            for h in [1, 7, 30, 90, 180, 360]:
                key = f"horizon_{h}d_price"
                val = getattr(pred, key, None)
                if val is not None:
                    pred_data[f"{h}天"] = val
            model_info = ""
            if active_model:
                acc = (active_model.direction_accuracy or 0) * 100
                model_info = (
                    f"\n\n模型：{active_model.version} | 方法：{active_model.method}\n"
                    f"方向准确率：{acc:.0f}%（{active_model.evaluated_count or 0}条）| MAPE：{active_model.mape_price_pct or 0:.1f}%"
                )
                if acc < 30 and (active_model.evaluated_count or 0) > 10:
                    insights.append(
                        f"⚠️ 预测模型准确率仅 {acc:.0f}%，建议检查是否有更好的候选模型可切换。"
                    )
            parts.append(f"## 价格预测\n\n{json.dumps(pred_data, ensure_ascii=False)}{model_info}")
    except Exception:
        pass

    # ── 4. 宏观 + CFTC + 情绪 + 溢价 ──
    try:
        from app.models import MacroObservation
        key_series = {"DFII10": "实际利率", "DGS10": "10年美债", "T10YIE": "通胀预期",
                       "DTWEXBGS": "美元指数", "VIXCLS": "VIX", "FEDFUNDS": "联邦基金利率"}
        lines = []
        for sid, label in key_series.items():
            row = db.scalar(select(MacroObservation).where(MacroObservation.series_id == sid).order_by(MacroObservation.timestamp.desc()))
            if row and row.value is not None:
                lines.append(f"- {label}：{row.value:.4f}（{row.timestamp.strftime('%Y-%m-%d') if row.timestamp else '?'}）")
        if lines:
            parts.append("## 宏观指标\n\n" + "\n".join(lines))
    except Exception:
        pass

    try:
        from app.models import CftcPosition
        cftc = db.scalar(select(CftcPosition).order_by(CftcPosition.timestamp.desc()))
        if cftc:
            net_pct = (cftc.noncommercial_net / cftc.open_interest * 100) if cftc.open_interest else 0
            parts.append(
                "## CFTC 持仓\n\n"
                f"- 日期：{cftc.timestamp}\n"
                f"- 投机净多：{cftc.noncommercial_net:,}（占总持仓 {net_pct:.1f}%）\n"
                f"- 多头 {cftc.noncommercial_long:,} / 空头 {cftc.noncommercial_short:,}\n"
                f"- 商业净空：{cftc.commercial_short - cftc.commercial_long:,}\n"
                f"- 总持仓：{cftc.open_interest:,}"
            )
            if net_pct > 40:
                insights.append(f"⚠️ CFTC 投机净多占比 {net_pct:.1f}%，处于拥挤水平，警惕多头踩踏。")
    except Exception:
        pass

    try:
        from app.models import NewsSentiment
        ns = db.scalar(select(NewsSentiment).order_by(NewsSentiment.timestamp.desc()))
        if ns:
            ns_ts = ns.timestamp
            if ns_ts and ns_ts.tzinfo is None:
                ns_ts = ns_ts.replace(tzinfo=timezone.utc)
            age_d = (now_utc - ns_ts).total_seconds() / 86400 if ns_ts else 0
            parts.append(
                "## 新闻情绪\n\n"
                f"- 情绪分：{ns.sentiment_score:.2f}（-1~+1）\n"
                f"- 标题：{ns.title or '(无)'}\n- 时间：{ns.timestamp}"
            )
            if age_d > 2:
                insights.append(f"⚠️ 新闻情绪数据已 {age_d:.0f} 天未更新。")
    except Exception:
        pass

    try:
        from app.models import ChinaGoldPremium
        prem = db.scalar(select(ChinaGoldPremium).order_by(ChinaGoldPremium.timestamp.desc()))
        if prem:
            parts.append(f"## 中国溢价\n\n- 溢价率：{prem.premium_pct:+.2f}%\n- 时间：{prem.timestamp}")
    except Exception:
        pass

    # ── 5. 采集器健康 ──
    try:
        from app.monitoring.collector_health import get_health_summary
        health = get_health_summary()
        critical = health.get("summary", {}).get("critical_issues", [])
        stale = [c["name"] for c in health.get("collectors", []) if c["status"] == "stale"]
        all_issues = critical + stale
        if all_issues:
            insights.append(f"⚠️ 采集器异常：{', '.join(all_issues)}")
            parts.append(f"## 采集器状态\n\n整体：{health.get('overall')} | "
                         f"异常采集器：{', '.join(all_issues) if all_issues else '无'}")
    except Exception:
        pass

    # ── 6. 系统诊断（模型版本、数据量、预测闭环）──
    diag_lines = []
    try:
        from app.models import ScoreParamsVersion, PredictionModelVersion
        score_ver = db.scalar(select(ScoreParamsVersion).where(ScoreParamsVersion.is_active == True))
        pred_ver = db.scalar(select(PredictionModelVersion).where(PredictionModelVersion.is_active == True))
        diag_lines.append(f"- 评分参数: {score_ver.version if score_ver else 'default rule_v2'}" + (f'（命中率 {score_ver.hit_rate:.0%}）' if score_ver and score_ver.hit_rate else ''))
        diag_lines.append(f"- 预测模型: {pred_ver.version if pred_ver else 'N/A'}" + (f'（方向准确率 {(pred_ver.direction_accuracy or 0)*100:.0f}%，{pred_ver.evaluated_count or 0}条）' if pred_ver else ''))
    except Exception:
        pass
    try:
        import sqlite3
        db_raw = sqlite3.connect('gold_monitor.db')
        cftc_n = db_raw.execute('SELECT COUNT(*) FROM cftc_positions').fetchone()[0]
        cftc_latest = db_raw.execute('SELECT MAX(timestamp) FROM cftc_positions').fetchone()[0]
        diag_lines.append(f"- CFTC 数据: {cftc_n} 条，最新 {cftc_latest[:10] if cftc_latest else 'N/A'}")
        if cftc_n < 10:
            insights.append(f"⚠️ CFTC 仅 {cftc_n} 条记录（手工估算），建议接入实时 API。")
        eval_pending = db_raw.execute(
            "SELECT COUNT(*) FROM gold_prediction_snapshots s WHERE s.target_timestamp <= datetime('now') AND s.id NOT IN (SELECT prediction_id FROM gold_prediction_evaluations)"
        ).fetchone()[0]
        diag_lines.append(f"- 到期未评估预测: {eval_pending} 条")
        if eval_pending > 0:
            insights.append(f"⚠️ 有 {eval_pending} 条到期预测尚未评估，方向准确率统计可能滞后。")
        db_raw.close()
    except Exception:
        pass
    if diag_lines:
        parts.append("## 系统诊断\n\n" + "\n".join(diag_lines))
    # ── 智能洞察前置 ──
    if insights:
        parts.insert(0, "## ⚠️ 系统主动告警\n\n" + "\n".join(f"- {i}" for i in insights))

    text = "\n\n".join(parts)
    _context_cache = {"text": text}
    _context_cache_time = _time.time()
    return text


# ==================== 自主操作 ====================

AVAILABLE_ACTIONS = {
    "重新评分": "重新计算一次评分快照（基于最新数据）",
    "切换预测模型": "运行无人值守自我修复闭环，候选模型达标才自动切换",
    "检查采集器": "检查所有数据采集器的健康状态",
    "刷新数据": "立即触发一次完整的数据采集",
    "自我修复": "运行自我评估、自我修正、自我进化闭环",
}

def execute_action(db: Session, action: str) -> dict[str, Any]:
    """执行 AI 建议的系统操作。"""
    if action == "重新评分":
        try:
            from app.scoring.score_optimizer import get_active_params
            from app.scoring.gold_score import compute_and_store_gold_score, compute_and_store_gold_score_with_params
            active_params = get_active_params(db)
            if active_params is not None:
                snap = compute_and_store_gold_score_with_params(db, active_params, source="ai_triggered")
            else:
                snap = compute_and_store_gold_score(db)
            return {"ok": True, "action": action,
                    "result": f"评分完成：总分 {snap.total_score:+.1f}，方向 {snap.direction}"}
        except Exception as e:
            return {"ok": False, "action": action, "error": str(e)}

    if action == "切换预测模型":
        try:
            from app.self_healing import run_self_healing_cycle
            result = run_self_healing_cycle(db, force=True, reason="ai_action_switch_model")
            return {"ok": result.get("ok", False), "action": action, "result": result}
        except Exception as e:
            return {"ok": False, "action": action, "error": str(e)}

    if action == "自我修复":
        try:
            from app.self_healing import run_self_healing_cycle
            result = run_self_healing_cycle(db, force=True, reason="ai_action_self_healing")
            return {"ok": result.get("ok", False), "action": action, "result": result}
        except Exception as e:
            return {"ok": False, "action": action, "error": str(e)}

    if action == "检查采集器":
        try:
            from app.monitoring.collector_health import get_health_summary
            health = get_health_summary()
            return {"ok": True, "action": action, "result": health}
        except Exception as e:
            return {"ok": False, "action": action, "error": str(e)}

    if action == "刷新数据":
        try:
            from app.scheduler import collect_and_score_job
            collect_and_score_job()
            return {"ok": True, "action": action, "result": "数据采集和评分已完成"}
        except Exception as e:
            return {"ok": False, "action": action, "error": str(e)}

    return {"ok": False, "action": action, "error": f"未知操作：{action}"}


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
            resp = requests.post(url, headers=headers, json=payload, timeout=API_TIMEOUT, verify=certifi.where())
            if resp.status_code in (429, 503):
                wait = min(2 ** attempt, 8)
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
        return "AI 服务暂时不可用，请稍后重试。"
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
    """发送消息，获取 AI 回复。每次注入完整系统上下文+智能洞察。"""
    _ensure_session(db, session_id)
    history = _load_history(db, session_id)

    system_ctx = _build_system_context(db)

    # 操作指令识别
    action_keywords = {
        "重新评分": "重新评分", "重算评分": "重新评分", "刷新评分": "重新评分",
        "切换模型": "切换预测模型", "换模型": "切换预测模型", "激活模型": "切换预测模型",
        "检查采集器": "检查采集器", "采集器状态": "检查采集器",
        "刷新数据": "刷新数据", "重新采集": "刷新数据",
    }
    matched_action = None
    for kw, action in action_keywords.items():
        if kw in message:
            matched_action = action
            break

    if matched_action and len(message) < 20:
        # 直接执行操作
        result = execute_action(db, matched_action)
        return {
            "session_id": session_id,
            "reply": json.dumps(result, ensure_ascii=False, indent=2),
            "score_total": _latest_score(db),
            "score_direction": _latest_direction(db),
            "messages_count": len(history) // 2,
            "action": matched_action,
        }

    messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
    messages.append({"role": "system", "content": "## 当前系统数据与洞察\n\n" + system_ctx})

    # 可用操作列表
    actions_hint = "\n\n可执行的系统操作：" + ", ".join(f"[{a}]" for a in AVAILABLE_ACTIONS)
    messages.append({"role": "system", "content": actions_hint})

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


def generate_insight(db: Session) -> dict[str, Any]:
    """生成主动智能洞察（供仪表盘定时拉取）。"""
    ctx = _build_system_context(db)

    messages = [
        {"role": "system", "content": """你是系统智能监控助手。基于当前数据，生成一条简短的主动洞察报告。

要求：
- 1-2 句话，不超过 100 字
- 如果一切正常，报告系统状态
- 如果有异常（数据过期、模型退化、金价异动），优先报告异常
- 用中文，简洁专业
- 格式：只返回文本，不要 JSON 或标记"""},
        {"role": "user", "content": f"当前系统数据：\n\n{ctx}\n\n请生成主动洞察。"},
    ]

    try:
        reply = _call_deepseek(messages)
        return {"ok": True, "insight": reply or "系统运行正常。",
                "generated_at": datetime.now(timezone.utc).isoformat()}
    except Exception:
        return {"ok": False, "insight": "洞察生成失败。",
                "generated_at": datetime.now(timezone.utc).isoformat()}


def reset_session(db: Session, session_id: str) -> None:
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
    rows = db.scalars(select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(limit)).all()
    return [{"session_id": r.session_id, "title": r.title, "message_count": r.message_count,
             "created_at": r.created_at.isoformat() if r.created_at else None,
             "updated_at": r.updated_at.isoformat() if r.updated_at else None} for r in rows]


def get_session_messages(db: Session, session_id: str) -> list[dict]:
    rows = db.scalars(select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc())).all()
    return [{"role": r.role, "content": r.content, "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]


def delete_session(db: Session, session_id: str) -> bool:
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
