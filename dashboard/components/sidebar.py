"""侧边栏组件"""
from __future__ import annotations
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
import plotly.express as px

import streamlit as st
import sys

def render_sidebar(**kwargs):
    """渲染侧边栏导航和设置"""
    globals().update({k:v for k,v in kwargs.items() if not k.startswith("_")})
    _main = sys.modules.get("__main__", None)
    if _main:
        for name in dir(_main):
            if not name.startswith("__") and name not in globals():
                try:
                    globals()[name] = getattr(_main, name)
                except (TypeError, AttributeError):
                    pass
    _main = sys.modules.get("__main__", None)
    if _main:
        for name in dir(_main):
            if not name.startswith("__") and name not in globals():
                try:
                    globals()[name] = getattr(_main, name)
                except (TypeError, AttributeError):
                    pass
    with st.sidebar:
        st.markdown("#### 🧭 模块导航")
        st.markdown(
            """<div style="display:flex;flex-direction:column;gap:4px;font-size:0.82rem;margin-bottom:12px">
            <a href="#gold-score" style="color:#3b82f6;text-decoration:none;padding:3px 8px;background:#eff6ff;border-radius:4px">📊 多空评分</a>
            <a href="#gold-price" style="color:#3b82f6;text-decoration:none;padding:3px 8px;background:#eff6ff;border-radius:4px">💰 金价走势</a>
            <a href="#gold-predict" style="color:#3b82f6;text-decoration:none;padding:3px 8px;background:#eff6ff;border-radius:4px">🔮 金价预测</a>
            <a href="#macro-indicators" style="color:#3b82f6;text-decoration:none;padding:3px 8px;background:#eff6ff;border-radius:4px">📈 宏观指标</a>
            <a href="#cftc" style="color:#3b82f6;text-decoration:none;padding:3px 8px;background:#eff6ff;border-radius:4px">📋 CFTC</a>
            <a href="#central-bank" style="color:#3b82f6;text-decoration:none;padding:3px 8px;background:#eff6ff;border-radius:4px">🏦 央行购金</a>
            <a href="#macro-events" style="color:#3b82f6;text-decoration:none;padding:3px 8px;background:#eff6ff;border-radius:4px">📅 宏观事件</a>
            <a href="#news-sentiment" style="color:#3b82f6;text-decoration:none;padding:3px 8px;background:#eff6ff;border-radius:4px">📰 新闻情绪</a>
            <a href="#score-evolution" style="color:#3b82f6;text-decoration:none;padding:3px 8px;background:#eff6ff;border-radius:4px">🧬 模型进化</a>
            </div>""",
            unsafe_allow_html=True,
        )
        st.divider()
        # 💬 AI 助手 — App 内独立窗口打开
        st.markdown("### 💬 AI 助手")
        st.markdown(
            '<a href="http://127.0.0.1:8000/ai/ui" target="_blank" style="text-decoration:none">'
            '<button style="width:100%;padding:8px;background:#334155;border:none;border-radius:6px;'
            'color:#e2e8f0;font-size:13px;cursor:pointer">'
            '💬 在 App 内打开 AI 对话</button></a>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("### ⚙️ 自动驾驶")
        auto_config = api("/settings/auto-optimize")
        auto_settings = auto_config.get("settings", {})
        new_full_auto = st.toggle("实验全自动进化", value=auto_settings.get("AUTO_EVOLUTION_FULL_AUTO", True))
        new_healing = st.toggle("自动评估与修复", value=auto_settings.get("AUTO_SELF_HEALING_ENABLED", True))
        new_autofix = st.toggle("达标后自动修正", value=auto_settings.get("AUTO_SELF_HEALING_AUTOFIX", True))
        with st.expander("高级模型开关", expanded=False):
            st.caption("实验全自动模式不会放宽质量标准；候选版本必须通过严格门控才会自动激活。")
            new_score = st.toggle("评分参数自动搜索", value=auto_settings.get("AUTO_OPTIMIZE_SCORE_PARAMS", True))
            new_activate = st.toggle("评分参数达标激活", value=auto_settings.get("AUTO_ACTIVATE_OPTIMIZED_PARAMS", True))
            new_pred = st.toggle("预测模型自动搜索", value=auto_settings.get("AUTO_OPTIMIZE_PREDICTION_MODEL", True))
            new_pred_activate = st.toggle("预测模型达标激活", value=auto_settings.get("AUTO_ACTIVATE_PREDICTION_MODEL", True))
        current_toggle = {
            "AUTO_EVOLUTION_FULL_AUTO": new_full_auto,
            "AUTO_SELF_HEALING_ENABLED": new_healing,
            "AUTO_SELF_HEALING_AUTOFIX": new_autofix,
            "AUTO_OPTIMIZE_SCORE_PARAMS": new_score,
            "AUTO_ACTIVATE_OPTIMIZED_PARAMS": new_activate,
            "AUTO_OPTIMIZE_PREDICTION_MODEL": new_pred,
            "AUTO_ACTIVATE_PREDICTION_MODEL": new_pred_activate,
        }
        if "_auto_initial" not in st.session_state:
            st.session_state["_auto_initial"] = dict(current_toggle)
        has_unsaved = current_toggle != st.session_state["_auto_initial"]
        save_label = "💾 保存" + (" ●" if has_unsaved else "")
        if st.button(save_label, use_container_width=True):
            r = api("/settings/auto-optimize", "post", json={
                "AUTO_EVOLUTION_FULL_AUTO": new_full_auto,
                "AUTO_SELF_HEALING_ENABLED": new_healing,
                "AUTO_SELF_HEALING_AUTOFIX": new_autofix,
                "AUTO_OPTIMIZE_SCORE_PARAMS": new_score,
                "AUTO_ACTIVATE_OPTIMIZED_PARAMS": new_activate,
                "AUTO_OPTIMIZE_PREDICTION_MODEL": new_pred,
                "AUTO_ACTIVATE_PREDICTION_MODEL": new_pred_activate,
            })
            if r.get("ok"):
                st.session_state["_auto_initial"] = dict(current_toggle)
                st.success("已保存")
                st.rerun()
        st.divider()
        st.caption("当前模式：AI 可行性验证全自动 + 严格质量门控。")
        st.divider()
    
        # ── API 密钥配置 ──
        with st.expander("🔑 API 配置", expanded=False):
            st.caption("配置后点击保存即可生效，无需重启。")
            env_config = api("/settings/env")
            current_env = env_config.get("env", {}) if env_config.get("ok") else {}
    
            def _mask(v: str) -> str:
                return v[:8] + "***" if len(v) > 8 else ("***" if v else "")
    
            new_ds = st.text_input("DeepSeek API Key", value=_mask(current_env.get("DEEPSEEK_API_KEY", "")),
                                   type="password", placeholder="sk-...")
            new_news = st.text_input("NewsAPI Key", value=_mask(current_env.get("NEWSAPI_KEY", "")),
                                      type="password", placeholder="32位hex...")
            new_fred = st.text_input("FRED API Key", value=_mask(current_env.get("FRED_API_KEY", "")),
                                      type="password", placeholder="32位hex...")
            new_feishu_url = st.text_input("飞书 Webhook URL", value=current_env.get("FEISHU_WEBHOOK_URL", ""),
                                            type="password", placeholder="https://open.feishu.cn/...")
            new_feishu_secret = st.text_input("飞书签名密钥", value=_mask(current_env.get("FEISHU_SECRET", "")),
                                               type="password")
    
            if st.button("💾 保存配置", use_container_width=True):
                updates = {}
                if new_ds and "***" not in new_ds: updates["DEEPSEEK_API_KEY"] = new_ds
                if new_news and "***" not in new_news: updates["NEWSAPI_KEY"] = new_news
                if new_fred and "***" not in new_fred: updates["FRED_API_KEY"] = new_fred
                if new_feishu_url and "***" not in new_feishu_url: updates["FEISHU_WEBHOOK_URL"] = new_feishu_url
                if new_feishu_secret and "***" not in new_feishu_secret: updates["FEISHU_SECRET"] = new_feishu_secret
                if updates:
                    r = api("/settings/env", "post", json={"updates": updates})
                    if r.get("ok"):
                        st.success("已保存，重启 API 后生效。")
                    else:
                        st.error(r.get("reason", "保存失败"))
                else:
                    st.info("没有检测到新的配置变更。")
    
    
    health_payload = auto_config.get("health", {}) if isinstance(auto_config, dict) else {}
    score_ok = health_payload.get("score_sample_ready", False)
    pred_ok = health_payload.get("prediction_sample_ready", False)
    all_ok = score_ok and pred_ok
    health_label = f"模型健康 · {'✅ 正常' if all_ok else '⚠️ 需关注'} · 评分{'足' if score_ok else '不足'}/预测{'足' if pred_ok else '不足'}"
    with st.expander(health_label, expanded=False):
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("评分参数版本", health_payload.get("score_params_version", "default"))
        h2.metric("预测模型版本", health_payload.get("prediction_model_version") or "—")
        h3.metric("已评估预测", health_payload.get("prediction_evaluated_count", 0))
        h4.metric("到期未评估", health_payload.get("prediction_due_pending_count", 0))
    
        h5, h6, h7, h8 = st.columns(4)
        h5.metric("评分样本条件", "已达进化门槛" if health_payload.get("score_sample_ready") else "样本不足，暂不进化")
        h6.metric("预测样本条件", "已达进化门槛" if health_payload.get("prediction_sample_ready") else "样本不足，暂不进化")
        h7.metric("自动搜索", "开" if (
            auto_settings.get("AUTO_OPTIMIZE_SCORE_PARAMS") or auto_settings.get("AUTO_OPTIMIZE_PREDICTION_MODEL")
        ) else "关")
        h8.metric("自动激活", "开" if (
            auto_settings.get("AUTO_ACTIVATE_OPTIMIZED_PARAMS") or auto_settings.get("AUTO_ACTIVATE_PREDICTION_MODEL")
        ) else "关")
    
        auto_evo = health_payload.get("auto_evolution") or {}
        e_left, e_right = st.columns(2)
        with e_left:
            st.caption(
                "短周期进化："
                f"目标 {auto_evo.get('target_horizons', [1, 7, 30])} · "
                f"样本门槛 {auto_evo.get('sample_threshold', 120)} · "
                f"{'可进化' if auto_evo.get('can_evolve_prediction') else '样本不足'}"
            )
        with e_right:
            action = auto_evo.get("last_prediction_action") or {}
            st.caption(
                "最近预测动作："
                f"{action.get('action', '—')} · "
                f"{action.get('from_version', '—')} → {action.get('to_version', '—')}"
            )
    
        latest_score_candidate = health_payload.get("latest_score_candidate") or {}
        latest_prediction_candidate = health_payload.get("latest_prediction_candidate") or {}
        c_left, c_right = st.columns(2)
        with c_left:
            st.caption(
                "最近评分候选："
                f"{latest_score_candidate.get('version', '—')} · "
                f"命中率 {latest_score_candidate.get('hit_rate', '—')} · "
                f"样本 {latest_score_candidate.get('sample_count', '—')}"
            )
        with c_right:
            st.caption(
                "最近预测候选："
                f"{latest_prediction_candidate.get('version', '—')} · "
                f"方向准确率 {latest_prediction_candidate.get('direction_accuracy', '—')} · "
                f"MAPE {latest_prediction_candidate.get('mape_price_pct', '—')}"
            )
        for reason in health_payload.get("reasons", []):
            st.caption(f"• {reason}")
    
    # ── 采集器健康 + AI 洞察（主动可查）──
    try:
        collector_health = api("/health/collectors")
        if collector_health and collector_health.get("summary"):
            summary = collector_health.get("summary", {})
            critical_issues = summary.get("critical_issues", [])
            healthy = summary.get("healthy", 0)
            total = summary.get("total", 0)
            overall = collector_health.get("overall", "healthy")
            status_icon = {"healthy": "✅", "degraded": "⚠️", "critical": "🔴", "initializing": "🔄"}.get(overall, "❓")
            status_label = f"{status_icon} 系统健康：{healthy}/{total} 采集器正常"
    
            with st.expander(status_label, expanded=False):
                # 每个采集器状态
                for c in collector_health.get("collectors", []):
                    s_icon = {"healthy": "✅", "degraded": "⚠️", "stale": "🕐", "no_data": "❌"}.get(c["status"], "❓")
                    age = f"{c['age_hours']:.1f}h" if c.get("age_hours") is not None else "从未"
                    err = f" — {c['last_error'][:60]}" if c.get("last_error") else ""
                    st.caption(f"{s_icon} {c['label']}：{c['status']}（{age}）{err}")
                if st.button("🔄 检查采集器", key="check_collectors_btn"):
                    api("/ai/action", "post", params={"action": "检查采集器"}, headers={"Authorization": f"Bearer {_get_api_token()}"})
                    st.rerun()
    except Exception:
        pass
    
    try:
        insight = api("/ai/insight")
        if insight.get("ok") and insight.get("insight"):
            st.info(f"🤖 {insight['insight']}")
    except Exception:
        pass
    
    # ═══════════════════════════════════════════
