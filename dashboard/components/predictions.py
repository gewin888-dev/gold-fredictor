"""金价预测组件"""
from __future__ import annotations
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
import plotly.express as px

import streamlit as st
import sys

def render_predictions(**kwargs):
    """渲染金价预测面板和模型健康"""
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
    st.subheader("金价预测")
    
    @st.cache_data(ttl=90)
    def get_prediction() -> tuple[dict, str]:
        """返回 (data_dict, error_msg)。error_msg 为空时表示成功。"""
        try:
            with httpx.Client(timeout=httpx.Timeout(45)) as c:
                r = c.get(f"{API_BASE_URL}/predict/gold")
                if r.status_code != 200:
                    return {}, f"API 返回状态 {r.status_code}"
                data = r.json()
                if not data.get("ok"):
                    return data, data.get("reason", "未知错误")
                return data, ""
        except Exception as e:
            return {}, f"连接后端失败：{e}"
    
    
    @st.cache_data(ttl=60)
    def get_prediction_evaluation() -> dict:
        return api("/predict/evaluation")
    
    
    @st.cache_data(ttl=60)
    def get_prediction_models() -> dict:
        return api("/predict/models")
    
    
    @st.cache_data(ttl=45)
    def get_prediction_due_status() -> dict:
        return api("/predict/due-status")
    
    pred, pred_error = get_prediction()
    if pred.get("ok"):
        current = pred.get("current_price")
        due_status = get_prediction_due_status()
        evaluated_count = int(due_status.get("evaluated_count") or 0)
        due_pending_count = int(due_status.get("due_pending_count") or 0)
        future_pending_count = int(due_status.get("future_pending_count") or 0)
        st.caption(
            f"v2 多信号集成：模型 {pred.get('model_version', '—')}，训练源 {', '.join(pred.get('training_sources', []))}。"
            "短期动量+评分回归，长期宏观基准+调整。"
            f"  UTC {_now_utc().strftime('%Y-%m-%d %H:%M')} / 北京 {_now_beijing().strftime('%Y-%m-%d %H:%M')}"
        )
        st.caption(
            f"预测闭环状态：已评估 {evaluated_count} 条，到期待评估 {due_pending_count} 条，"
            f"待到期 {future_pending_count} 条。{due_status.get('message', '')}"
        )
        if due_status.get("by_horizon"):
            short_rows = [
                row for row in due_status.get("by_horizon", [])
                if int(row.get("horizon_days") or 0) in {1, 7, 30}
            ]
            if short_rows:
                short_df = pd.DataFrame(short_rows).rename(columns={
                    "horizon_days": "期限",
                    "evaluated_count": "已评估",
                    "due_pending_count": "到期待评估",
                    "future_pending_count": "待到期",
                })
                st.dataframe(short_df[["期限", "已评估", "到期待评估", "待到期"]], use_container_width=True, hide_index=True)
        if due_status.get("cannot_evolve_reasons"):
            st.caption("短周期进化门槛：" + "；".join(due_status.get("cannot_evolve_reasons", [])))
    
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 2])
        target_evaluated = int(due_status.get("target_evaluated_count") or 0)
        candidate_help = "1/7/30天样本少于 120 条，不建议生成候选模型。" if target_evaluated < 120 else "短周期样本条件基本满足，可生成候选模型。"
        st.caption(candidate_help)
        with ctrl1:
            if st.button("保存本次预测", use_container_width=True):
                api("/predict/gold/snapshot", "post")
                st.cache_data.clear()
                st.rerun()
        with ctrl2:
            if st.button(f"补评估到期预测({due_pending_count})", use_container_width=True):
                api("/predict/evaluate", "post")
                st.cache_data.clear()
                st.rerun()
        with ctrl3:
            if st.button("生成候选模型", use_container_width=True):
                r = api("/predict/models/optimize", "post", params={"n_iter": 40, "top_k": 5, "save_best": True, "auto_activate": bool(auto_settings.get("AUTO_ACTIVATE_PREDICTION_MODEL"))})
                st.session_state["prediction_optimize_result"] = r
                st.cache_data.clear()
                st.rerun()
        opt_result = st.session_state.get("prediction_optimize_result")
        if opt_result:
            if opt_result.get("ok"):
                best = opt_result.get("best") or {}
                activation = opt_result.get("activation") or {}
                overfit = activation.get("overfit_risk") or {}
                mode = "已自动激活" if activation.get("activated") else "等待人工激活"
                st.success(
                    f"已生成候选模型 {opt_result.get('saved_version')}："
                    f"综合分 {best.get('optimization_score')}，"
                    f"MAPE {best.get('weighted_mape_price_pct')}%，"
                    f"方向准确率 {best.get('weighted_direction_accuracy')}，"
                    f"近期 {best.get('weighted_recent_direction_accuracy')}，"
                    f"相对baseline {activation.get('baseline_lift')}。"
                    f"{mode}。"
                )
                if overfit.get("level"):
                    st.caption("过拟合检测：" + overfit.get("level", "—") + " · " + "；".join(overfit.get("warnings", [])))
                if activation.get("reasons"):
                    st.caption("自动激活判断：" + "；".join(activation.get("reasons", [])))
            else:
                st.warning(f"候选模型生成失败：{opt_result.get('reason', '未知原因')}")
    
        preds = pred.get("predictions", [])
        if preds:
            # Hover tooltip 样式
            st.markdown("""<style>
            .pred-wrap { position:relative; display:inline-block; width:100%; }
            .pred-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
                padding:8px 10px; cursor:pointer; transition:box-shadow 0.15s; }
            .pred-card:hover { box-shadow:0 2px 8px rgba(0,0,0,0.12); }
            .pred-card h4 { margin:0 0 2px 0; font-size:0.8rem; color:#64748b; }
            .pred-card .price { font-size:1.15rem; font-weight:700; color:#1e293b; }
            .pred-card .delta { font-size:0.82rem; margin-left:6px; }
            .pred-card .range { font-size:0.72rem; color:#94a3b8; margin-top:2px; }
            .pred-card .confidence { font-size:0.68rem; color:#64748b; margin-top:1px;
                background:#f1f5f9; display:inline-block; padding:1px 6px; border-radius:3px; }
            .pred-tip { visibility:hidden; opacity:0; transition:opacity 0.2s;
                position:absolute; z-index:999; bottom:110%; left:-10px;
                background:#ffffff; border:1px solid #d7dde6; border-radius:8px; padding:14px 18px;
                font-size:0.78rem; box-shadow:0 10px 28px rgba(15,23,42,0.16);
                min-width:460px; pointer-events:none; white-space:normal; }
            .pred-wrap:last-child .pred-tip { left:auto; right:-10px; }
            .pred-wrap:first-child .pred-tip { left:0; }
            .pred-tip-right { left:auto !important; right:0 !important; }
            .pred-tip-left { left:0 !important; }
            .pred-card:hover + .pred-tip,
            .pred-card:active + .pred-tip,
            .pred-card:focus + .pred-tip { visibility:visible; opacity:1; }
            .pred-tip td { padding:4px 12px; vertical-align:top; color:#334155; line-height:1.55; }
            .pred-tip td:first-child { border-right:1px solid #e2e8f0; width:180px; }
            .pred-tip .th { color:#64748b; font-size:0.7rem; text-transform:uppercase; }
            .pred-tip .h1 { color:#18212f; font-size:0.95rem; font-weight:700; }
            .pred-tip .h2 { color:#18212f; font-size:0.82rem; }
            .pred-tip .h3 { color:#475569; font-size:0.72rem; }
            </style>""", unsafe_allow_html=True)
    
            cols = st.columns(len(preds))
            for i, p in enumerate(preds):
                samples = p.get("samples", 0)
                no_data = samples == 0
                rel = p.get("reliability", 0) if not no_data else 0
                note = p.get("note", "")
                low = p.get("low", 0)
                high = p.get("high", 0)
                return_pct = p.get("return_pct", 0)
                icon = "🟢" if return_pct > 0 else "🔴"
                color = "#16a34a" if return_pct > 0 else "#dc2626"
                predicted = p.get("predicted", 0)

                note_html = html.escape(note).replace("\n", "<br>") if note else "暂无详情"

                # tip_class 提前计算，no_data 和正常分支共用
                tip_class = ""
                if i == 0:
                    tip_class = " pred-tip-left"
                elif i == len(preds) - 1:
                    tip_class = " pred-tip-right"

                with cols[i]:
                    if no_data:
                        st.markdown(f"""
                    <div class="pred-wrap">
                      <div class="pred-card" tabindex="0" style="opacity:0.55">
                        <h4>{p.get('horizon', '?')}</h4>
                        <span class="price" style="color:#94a3b8">数据不足</span>
                        <div class="confidence" style="background:#fef3c7;color:#92400e">缺少历史数据</div>
                      </div>
                      <div class="pred-tip{tip_class}">
                        <table><tr>
                        <td>
                          <div class="th">预测价格</div><div class="h1" style="color:#94a3b8">数据不足</div>
                          <div class="th">原因</div><div class="h3">评分历史数据不足，无法评估 {p.get('horizon', '?')} 期限的预测准确率。</div>
                          <div class="th">样本量</div><div class="h3">{samples} 条</div>
                        </td>
                        <td>
                          <div class="th">预测理由</div>
                          <div class="h3">积累更多评分快照后（约需 7 天以上历史），系统将自动补全短周期预测。</div>
                        </td>
                        </tr></table>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                    <div class="pred-wrap">
                      <div class="pred-card" tabindex="0">
                        <h4>{p.get('horizon', '?')}</h4>
                        <span class="price">${predicted:,.0f}</span>
                        <span class="delta" style="color:{color}">{icon} {return_pct:+.1f}%</span>
                        <div class="range">${low:,.0f} — ${high:,.0f}</div>
                        <div class="confidence">置信度 {rel:.0%}</div>
                      </div>
                      <div class="pred-tip{tip_class}">
                        <table><tr>
                        <td>
                          <div class="th">预测价格</div><div class="h1">${predicted:,.0f}</div>
                          <div class="th">预期收益 · 置信度</div><div class="h2" style="color:{color}">{icon} {return_pct:+.1f}% · {rel:.0%}</div>
                          <div class="th">波动区间</div><div class="h3">${low:,.0f} — ${high:,.0f}</div>
                          <div class="th">样本量</div><div class="h3">{samples} 条</div>
                        </td>
                        <td>
                          <div class="th">预测理由</div>
                          <div class="h3">{note_html}</div>
                        </td>
                        </tr></table>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
    
        st.markdown("<div style='height:54px'></div>", unsafe_allow_html=True)
    
        # 预测曲线图 — 仅包含有数据的预测
        if current and preds:
            import plotly.graph_objects as go
            valid_preds = [p for p in preds if p.get("samples", 0) > 0]
            if not valid_preds:
                # 没有任何有效预测，跳过图表
                pass
            else:
                horizons = [p["horizon"] for p in valid_preds]
                pred_prices = [p["predicted"] for p in valid_preds]
                lows = [p["low"] for p in valid_preds]
                highs = [p["high"] for p in valid_preds]

                x_labels = ["当前"] + horizons
                y_vals = [current] + pred_prices
                y_low = [current] + lows
                y_high = [current] + highs

                fig_p = go.Figure()
                # 置信区间填充
                fig_p.add_trace(go.Scatter(
                    x=x_labels + x_labels[::-1],
                    y=y_high + y_low[::-1],
                    fill="toself", fillcolor="rgba(240,185,11,0.15)",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))
                # 预测线
                fig_p.add_trace(go.Scatter(
                    x=x_labels, y=y_vals, mode="lines+markers",
                    line=dict(color="#f0b90b", width=1.5), marker=dict(size=6),
                    hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
                ))
                fig_p.update_layout(
                    **PLOTLY_LIGHT_LAYOUT,
                    height=300, margin=dict(l=0,r=0,t=0,b=20),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(title=None, showgrid=True, gridcolor="#f1f5f9"),
                    hovermode="x unified", showlegend=False
                )
                st.plotly_chart(fig_p, use_container_width=True)
    
            with st.expander("预测理论与理由", expanded=False):
                st.caption(
                    "当前点为真实价格；未来点为模型估计。以下内容自动随 /predict/gold 重新计算。"
                    f"  UTC {_now_utc().strftime('%Y-%m-%d %H:%M')} / 北京 {_now_beijing().strftime('%Y-%m-%d %H:%M')}"
                )
                for p in preds:
                    if p.get("samples", 0) == 0:
                        st.markdown(
                            f"**{p.get('horizon', '?')} · ⚠️ 数据不足**  \\n"
                            f"评分历史数据尚不够长，无法评估该期限的预测准确率。积累更多数据后自动补全。"
                        )
                        continue
                    note = p.get("note") or "暂无预测理由。"
                    err = p.get("error_metrics") or {}
                    err_text = ""
                    if err.get("ok"):
                        err_text = (
                            f"历史误差：MAE ${err.get('mae_price', 0):,.0f}，"
                            f"MAPE {err.get('mape_price_pct', 0):.1f}%，"
                            f"方向准确率 {err.get('direction_accuracy', 0):.0%}。"
                        )
                    st.markdown(
                        f"**{p.get('horizon', '?')} · 可靠性{p.get('reliability_label', '低')}**  \\n"
                        f"预测价 `${p.get('predicted', 0):,.0f}`，"
                        f"预期收益 `{p.get('return_pct', 0):+.1f}%`，"
                        f"区间 `${p.get('low', 0):,.0f} - ${p.get('high', 0):,.0f}`。"
                    )
                    if err_text:
                        st.caption(err_text)
                    st.markdown(note.replace("\n", "  \n"))
    
            eval_data = get_prediction_evaluation()
            model_data = get_prediction_models()
            if eval_data.get("ok"):
                summary = eval_data.get("summary", {})
                st.markdown(
                    f"#### 预测验证闭环  UTC {_now_utc().strftime('%Y-%m-%d %H:%M')} / 北京 {_now_beijing().strftime('%Y-%m-%d %H:%M')}"
                )
                e1, e2, e3, e4, e5 = st.columns(5)
                e1.metric("已验证", f"{summary.get('evaluated_count', 0)} 条")
                e2.metric("待到期", f"{summary.get('future_pending_count', 0)} 条")
                e3.metric("到期待评估", f"{summary.get('due_pending_count', 0)} 条")
                mae = summary.get("mae_price")
                mape = summary.get("mape_price_pct")
                acc = summary.get("direction_accuracy")
                e4.metric("MAE", f"${mae:,.0f}" if mae is not None else "—")
                e5.metric("方向准确率", f"{acc:.0%}" if acc is not None else "—", delta=f"MAPE {mape:.1f}%" if mape is not None else None)
    
                by_h = eval_data.get("by_horizon", [])
                if by_h:
                    hdf = pd.DataFrame(by_h)
                    hdf = hdf.rename(columns={
                        "horizon_days": "期限",
                        "count": "样本",
                        "mae_price": "MAE($)",
                        "mape_price_pct": "MAPE(%)",
                        "direction_accuracy": "方向准确率",
                    })
                    hdf["方向准确率"] = hdf["方向准确率"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
                    st.dataframe(hdf, use_container_width=True, hide_index=True)
                else:
                    st.caption("还没有到期预测。保存快照后，等对应 horizon 到期并采集到真实金价，系统会自动比对。")
    
            if model_data.get("ok") and model_data.get("data"):
                with st.expander("预测模型版本", expanded=False):
                    mdf = pd.DataFrame(model_data["data"])
                    show_cols = [
                        "version", "method", "is_active", "evaluated_count",
                        "mae_price", "mape_price_pct", "direction_accuracy", "notes",
                    ]
                    existing_cols = [c for c in show_cols if c in mdf.columns]
                    st.dataframe(mdf[existing_cols], use_container_width=True, hide_index=True)
                    st.caption("预测候选围绕 1/7/30 天方向命中率评估；开启自动激活后，只有通过样本、baseline、近期窗口、MAPE 和过拟合门控才会上线。")
    else:
        if pred_error:
            st.warning(f"预测数据加载失败：{pred_error}")
            if st.button("重试加载预测", key="retry_pred"):
                st.cache_data.clear()
                st.rerun()
        else:
            st.caption(
                "暂无预测数据（需要足够的评分历史或已配置的同版本评分源）。"
                " 请确认已采集 FRED/CFTC 等核心数据并至少执行过一次评分计算。"
            )
    
    # ═══════════════════════════════════════════
    # 宏观指标
    # ═══════════════════════════════════════════
    
    st.divider()
    st.markdown('<a id="macro-indicators"></a>', unsafe_allow_html=True)

