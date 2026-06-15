"""金价卡片组件"""
from __future__ import annotations
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
import plotly.express as px

import streamlit as st
import sys

def render_price_cards(**kwargs):
    """渲染 COMEX 和上海金价卡片 + 日内走势"""
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
    # 时间条 + 金价卡片
    # ═══════════════════════════════════════════
    
    health = get_health()
    health_status = health.get("status", "unknown")
    health_label = {"ok": "正常", "warn": "延迟", "error": "异常"}.get(health_status, health_status)
    health_color = {"ok": "#047857", "warn": "#b45309", "error": "#b91c1c"}.get(health_status, "#64748b")
    
    gold = get_gold()
    
    # 时间条（时钟 + 数据源/更新/状态）—— 用 Streamlit 列对齐下面卡片
    utc_now = _now_utc()
    beijing_now = _now_beijing()
    tc1, tc2, tc3, tc4, tc5, tc6 = st.columns(6)
    with tc1:
        st.markdown(
            f'<div class="time-box time-box-muted time-box-compact">'
            f'<div class="time-value"><span style="font-size:0.65rem;color:#94a3b8;">北京 </span>{beijing_now.strftime("%H:%M")}</div>'
            f'<div style="color:#94a3b8;font-size:0.74rem;">{beijing_now.strftime("%m月%d日")}</div></div>',
            unsafe_allow_html=True)
    with tc2:
        st.markdown(
            f'<div class="time-box time-box-muted time-box-compact">'
            f'<div class="time-value"><span style="font-size:0.65rem;color:#94a3b8;">UTC </span>{utc_now.strftime("%H:%M")}</div>'
            f'<div style="color:#94a3b8;font-size:0.74rem;">{utc_now.strftime("%m月%d日")}</div></div>',
            unsafe_allow_html=True)
    with tc3:
        st.markdown(
            f'<div class="time-box time-box-muted time-box-compact">'
            f'<div class="time-label">自动刷新</div>'
            f'<div class="time-value">{st.session_state["_rf_interval"]}s</div></div>',
            unsafe_allow_html=True)
    with tc4:
        src = gold.get("source", "—") if gold.get("ok") else "—"
        st.markdown(
            f'<div class="time-box time-box-muted time-box-compact">'
            f'<div class="time-label">数据源</div>'
            f'<div class="time-value">{src}</div></div>',
            unsafe_allow_html=True)
    with tc5:
        upd = _ago(gold.get("timestamp", "")) if gold.get("ok") else "—"
        st.markdown(
            f'<div class="time-box time-box-muted time-box-compact">'
            f'<div class="time-label">更新</div>'
            f'<div class="time-value">{upd}</div></div>',
            unsafe_allow_html=True)
    with tc6:
        st.markdown(
            f'<div class="time-box time-box-muted time-box-compact">'
            f'<div class="time-label">状态</div>'
            f'<div class="time-value" style="color:{health_color};font-weight:600;">{health_label}</div></div>',
            unsafe_allow_html=True)
    
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    
    if gold.get("ok") and gold.get("price"):
        st.markdown('<a id="gold-price"></a>', unsafe_allow_html=True)
        fr = gold.get("freshness", "stale")
        fr_icon = {"live": "🟢", "delayed": "🟡", "stale": "🔴"}.get(fr, "")
        chg = gold.get("change")
        chg_pct = gold.get("change_pct")
        delta = f"{chg:+.2f} ({chg_pct:+.2f}%)" if chg is not None else None
    
        g1, g2, g3, g4, g5, g6 = st.columns(6)
        g1.metric(f"{fr_icon} COMEX 金价", f"${gold['price']:,.2f}", delta=delta)
        g2.metric("今日最高", f"${gold.get('day_high'):,.0f}" if gold.get('day_high') else "—")
        g3.metric("今日最低", f"${gold.get('day_low'):,.0f}" if gold.get('day_low') else "—")
    
        # 沪金连续（带交易状态，红停绿交）
        sh = get_shanghai_gold()
        if sh.get("ok"):
            bj_now = _now_beijing()
            h, m = bj_now.hour, bj_now.minute
            t = h * 60 + m
            in_day = (540 <= t <= 690) or (810 <= t <= 900)
            in_night = t >= 1260 or t <= 150
            if in_day or in_night:
                trading = True
                status_zh = "交易中"
            else:
                trading = False
                status_zh = "停盘中"
            status_color_hex = "#047857" if trading else "#b91c1c"
            status_icon = "🟢" if trading else "🔴"
            g4.metric(
                "沪金连续",
                f"¥{sh['price']:,.2f}/g",
                delta=f"{'🟢' if trading else '🔴'} {status_zh}  ·  高{sh['high']:,.0f} 低{sh['low']:,.0f}",
                delta_color="off",
            )
        else:
            g4.metric("沪金连续", "—")
    
        prem_df_for_metric = get_premium()
        premium_value = "—"
        premium_delta = None
        premium_src = prem_df_for_metric.iloc[0].get("来源", "") if not prem_df_for_metric.empty else ""
        if (
            not prem_df_for_metric.empty
            and pd.notna(prem_df_for_metric.iloc[0]["溢价"])
            and not (SETTINGS.production_mode and _is_low_confidence_source(prem_df_for_metric.iloc[0].get("来源")))
        ):
            premium_value = f"{prem_df_for_metric.iloc[0]['溢价']:+.2f}%"
            if premium_src.upper() == "SINA":
                premium_delta = "展示用，不参与评分"
        elif SETTINGS.production_mode and not prem_df_for_metric.empty:
            premium_value = "待接入"
            premium_delta = "接入真实数据后可用"
        g5.metric("中国溢价", premium_value, delta=premium_delta, delta_color="off")
    
        # 第6列聚合：来源 + 更新
        source_info = gold.get("source", "—")
        g6.metric(f"{fr_icon} 数据", f"{source_info}", delta=_ago(gold.get("timestamp", "")))
    
        # 金价走势图 — 多周期可选
        st.caption("金价走势")
        range_tab = st.radio("周期", ["7天", "30天", "360天"], horizontal=True, index=0, label_visibility="collapsed")
        range_days = {"7天": 7, "30天": 30, "360天": 360}[range_tab]
    
        import plotly.graph_objects as go
        db = SessionLocal()
        try:
            from app.models import GoldPrice as GP
            rows = db.scalars(select(GP).order_by(GP.date.desc()).limit(range_days)).all()
            if rows:
                df = pd.DataFrame([{"日期": r.date, "收盘": r.close} for r in reversed(rows)])
                # ── 沪金日线 ──
                sh_daily = get_shanghai_daily()
                sh_line = None
                if not sh_daily.empty:
                    sh_daily["日期"] = pd.to_datetime(sh_daily["d"])
                    sh_daily["收盘_cny"] = sh_daily["c"].astype(float)
                    # 如果今天还没日线数据，用实时报价补一条
                    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
                    if today_str not in sh_daily["d"].values:
                        sh_live = get_shanghai_gold()
                        if sh_live.get("ok"):
                            sh_daily = pd.concat([
                                sh_daily,
                                pd.DataFrame([{"d": today_str, "c": sh_live["price"],
                                               "日期": pd.Timestamp(today_str),
                                               "收盘_cny": float(sh_live["price"])}])
                            ], ignore_index=True)
                    start_d = df["日期"].min()
                    sh_line = sh_daily[(sh_daily["日期"] >= start_d - pd.Timedelta(days=2)) & (sh_daily["日期"] <= df["日期"].max() + pd.Timedelta(days=2))]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["日期"], y=df["收盘"], mode="lines+markers",
                    name="COMEX",
                    line=dict(color="#f0b90b", width=1.5), marker=dict(size=3),
                    hovertemplate="%{x|%Y-%m-%d}<br>COMEX $%{y:,.0f}<extra></extra>",
                ))
                # 沪金第二trace
                sh_trace = None
                if sh_line is not None and not sh_line.empty:
                    sh_trace = go.Scatter(
                        x=sh_line["日期"], y=sh_line["收盘_cny"], mode="lines+markers",
                        name="沪金",
                        line=dict(color="#dc2626", width=1.2), marker=dict(size=2),
                        yaxis="y2",
                        hovertemplate="%{x|%Y-%m-%d}<br>沪金 ¥%{y:,.1f}/g<extra></extra>",
                    )
                    fig.add_trace(sh_trace)
                # 右上角图例标注
                fig.add_annotation(
                    x=0.005, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top",
                    text="<span style='color:#c9972b'>● COMEX</span>  <span style='color:#dc2626'>● 沪金</span>",
                    showarrow=False, font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.82)", borderpad=4,
                )
                fig.update_layout(
                    **PLOTLY_LIGHT_LAYOUT,
                    height=300, margin=dict(l=0,r=0,t=0,b=30),
                    xaxis=dict(tickformat="%m/%d", tickangle=-45, showgrid=False),
                    yaxis=dict(title=None, showgrid=True, gridcolor="#f1f5f9"),
                    yaxis2=dict(
                        title=None, overlaying="y", side="right",
                        showgrid=False, color="#dc2626",
                    ),
                    hovermode="x unified", showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                latest_daily = pd.to_datetime(df["日期"].iloc[-1]).strftime("%Y-%m-%d %H:%M UTC")
                st.caption(f"金价日线：COMEX（黄） + 沪金连续（红，¥/g）。最新日线：{latest_daily}。COMEX 实时报价见上方卡片。")
        finally:
            db.close()
    
        # 24 小时金价走势趋势图
        from app.data.gold_price_collector import is_comex_market_closed as _market_closed
        comex_raw = get_intraday()
        # 休市且数据陈旧（>2小时）→ 跳过图表，显示提示
        _data_stale = comex_raw.empty or (
            not comex_raw.empty and
            (pd.Timestamp.utcnow() - pd.to_datetime(comex_raw["timestamp"].max(), utc=True)).total_seconds() > 7200
        )
        if _market_closed() and _data_stale:
            st.divider()
            st.caption("⏸ COMEX 休市中（周末维护窗口 UTC 周六至周日 22:00），24小时走势图将在开盘后自动恢复。")
        elif not comex_raw.empty:
            coverage_label, intraday_notes = _intraday_coverage_label(comex_raw)
            st.divider()
            col_i1, col_i2 = st.columns([4, 1])
            with col_i1:
                st.caption(coverage_label)
            with col_i2:
                st.markdown(f'<span class="live-dot"></span> <span style="color:#94a3b8;font-size:0.72rem;">实时监控中 · {time.strftime("%H:%M:%S")}</span>', unsafe_allow_html=True)
    
            # COMEX: resample 1-min raw to 5-min OHLC
            comex_raw["timestamp"] = pd.to_datetime(comex_raw["timestamp"], errors="coerce")
            comex_raw["close"] = pd.to_numeric(comex_raw["close"], errors="coerce")
            comex_raw = comex_raw.dropna(subset=["timestamp", "close"])
            comex_raw = comex_raw.set_index("timestamp").sort_index()
            comex_5m = comex_raw.resample("5min").agg({"close": "last"}).dropna()
            idf = comex_5m.reset_index().rename(columns={"index": "timestamp"})
    
            # Shanghai: keep original Sina timestamps (now with full date)
            sh_intra = get_shanghai_intraday()
            sh_price_text = ""
    
            fig_i = go.Figure()
            if not idf.empty:
                # ── COMEX 金价（黄色） ──
                fig_i.add_trace(go.Scatter(
                    x=idf["timestamp"], y=idf["close"], mode="lines+markers",
                    name="COMEX",
                    line=dict(color="#f0b90b", width=1.5), marker=dict(size=2),
                    hovertemplate="%{x|%m/%d %H:%M}<br>COMEX $%{y:,.2f}<extra></extra>",
                ))
            # ── 沪金日内（红色） ──
            if not sh_intra.empty:
                sh_price_text = f"沪金 ¥{sh_intra['close'].iloc[-1]:,.1f}/g · "
                fig_i.add_trace(go.Scatter(
                    x=sh_intra["timestamp"], y=sh_intra["close"], mode="lines+markers",
                    name="沪金",
                    line=dict(color="#dc2626", width=1.2), marker=dict(size=2),
                    yaxis="y2",
                    hovertemplate="%{x|%m/%d %H:%M}<br>沪金 ¥%{y:,.1f}/g<extra></extra>",
                ))
            else:
                _sh = get_shanghai_gold()
                if _sh.get("ok"):
                    sh_price_text = f"沪金 ¥{_sh['price']:,.1f}/g · "
            # 右上角图例标注
            fig_i.add_annotation(
                x=0.005, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top",
                text="<span style='color:#c9972b'>● COMEX</span>  <span style='color:#dc2626'>● 沪金</span>",
                showarrow=False, font=dict(size=11),
                bgcolor="rgba(255,255,255,0.82)", borderpad=4,
            )
            fig_i.update_layout(
                **PLOTLY_LIGHT_LAYOUT,
                height=300, margin=dict(l=0,r=0,t=0,b=20),
                xaxis=dict(tickformat="%m/%d %H:%M", showgrid=False),
                yaxis=dict(title=None, showgrid=True, gridcolor="#f1f5f9"),
                yaxis2=dict(
                    title=None, overlaying="y", side="right",
                    showgrid=False, color="#dc2626",
                ),
                hovermode="x unified", showlegend=False
            )
            st.plotly_chart(fig_i, use_container_width=True)
            caption_parts = [f"{coverage_label} · {sh_price_text}5分钟聚合，来源：新浪财经"]
            caption_parts.extend(note.rstrip("。") for note in intraday_notes)
            st.caption("；".join(caption_parts) + "。")
        else:
            st.caption("日内金价快照不足，后台记录器启动并采集到数据后会显示走势。")
    else:
        st.warning("实时金价暂时无法获取。")
    
    st.divider()
    
    # ═══════════════════════════════════════════
    # 评分
    # ═══════════════════════════════════════════
    
    # ═══════════════════════════════════════════
    # 评分
    # ═══════════════════════════════════════════
    
    scores = get_scores()
    
    if scores.empty:
        st.warning("暂无评分数据。")
    else:
        latest = scores.iloc[-1]
        raw_factors = json.loads(latest["因子"])
        # v2 格式: {"scores": {...}, "details": {...}}
        if isinstance(raw_factors, dict) and "scores" in raw_factors:
            factors = raw_factors["scores"]
            factor_details = raw_factors.get("details", {})
        else:
            factors = raw_factors  # v1 兼容
            factor_details = {}
        risks = json.loads(latest["风险"])
    
        st.markdown('<a id="gold-score"></a>', unsafe_allow_html=True)
        sh_col1, sh_col2 = st.columns([4, 1])
        with sh_col1:
            st.subheader("黄金多空评分")
        with sh_col2:
            if st.button("🔄 采集+评分", use_container_width=True, type="primary"):
                api("/score/compute", "post")
                st.cache_data.clear()
    
        s1, s2, s3, s4 = st.columns(4)
        score_val = latest["评分"]
        direction_icon = "🟢" if score_val >= 30 else ("🔴" if score_val <= -30 else "🟡")
        # 综合评分突出显示
        score_color = "#047857" if score_val >= 30 else ("#b91c1c" if score_val <= -30 else "#b45309")
        score_bg = "#f0fdf4" if score_val >= 30 else ("#fef2f2" if score_val <= -30 else "#fffbeb")
        dir_label = "偏多" if score_val >= 30 else ("偏空" if score_val <= -30 else "中性")
        s1.markdown(
            f'<div style="text-align:center;padding:12px 8px;border-radius:8px;'
            f'background:{score_bg};border:2px solid {score_color};min-width:120px">'
            f'<div style="font-size:0.72rem;color:#64748b;margin-bottom:2px">综合评分 · {dir_label}</div>'
            f'<div style="font-size:1.55rem;font-weight:800;color:{score_color}">{direction_icon} {score_val:+.1f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        bull = sum(v for v in factors.values() if v > 0)
        bear = sum(abs(v) for v in factors.values() if v < 0)
        net = bull - bear
        s2.metric("因子总数", len(factors))
        s3.metric("利多合计", f"+{bull:.0f}分")
        s4.metric("利空合计", f"-{bear:.0f}分")
        # ── 评分参考说明 ──
        st.markdown(
            '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px 14px;margin:4px 0 8px 0">'
            '<span style="font-size:0.78rem;color:#64748b">📖 '
            '<b>评分解读</b>：+30 以上偏多，−30 以下偏空，中间为中性。'
            f'当前评分基于 {len(factors)} 项已入库因子（利率、美元、流动性、持仓、情绪、央行等）加权计算，'
            '正分=利多黄金，负分=利空黄金。</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    
        factor_help = registry_factor_help()
        factor_groups = registry_factor_groups()
        inactive_factor_reasons = registry_inactive_reasons()

        # ── 因子气泡网格（4列紧凑卡片 + 分组标题）──
    
        factor_help = registry_factor_help()
        factor_groups = registry_factor_groups()
        inactive_factor_reasons = registry_inactive_reasons()

        for group_name, names in factor_groups.items():
            group_items = [
                (name, factors.get(name))
                for name in names
                if name in factors or name in inactive_factor_reasons
            ]
            if not group_items:
                continue

            st.markdown(f'<div class="section-label">{group_name}</div>', unsafe_allow_html=True)

            # 构建 HTML 卡片气泡，4 列网格
            cards_html = '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin-bottom:8px">'
            for name, val in group_items:
                if val is None:
                    bg = "#f1f5f9"
                    border = "#cbd5e1"
                    color = "#94a3b8"
                    display = "—"
                    reason = inactive_factor_reasons.get(name, "暂无可信数据。")
                    tip = html.escape(reason, quote=True)
                elif val > 0:
                    intensity = min(1.0, abs(val) / 10)
                    bg = f"#f0fdf4"
                    border = "#10b981"
                    color = "#047857"
                    display = f"+{val:.1f}" if val >= 0 else f"{val:.1f}"
                    tip = html.escape(factor_help.get(name, ""), quote=True)
                elif val < 0:
                    intensity = min(1.0, abs(val) / 10)
                    bg = "#fef2f2"
                    border = "#ef4444"
                    color = "#b91c1c"
                    display = f"{val:.1f}"
                    tip = html.escape(factor_help.get(name, ""), quote=True)
                else:
                    bg = "#ffffff"
                    border = "#d1d5db"
                    color = "#6b7280"
                    display = "0.0"
                    tip = html.escape(factor_help.get(name, ""), quote=True)

                cards_html += (
                    f'<div title="{tip}" style="background:{bg};border:1px solid {border};'
                    f'border-radius:6px;padding:6px 8px;text-align:center;cursor:default">'
                    f'<div style="font-size:0.72rem;color:#64748b;margin-bottom:2px;'
                    f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
                    f'{html.escape(name)}</div>'
                    f'<div style="font-size:0.95rem;font-weight:700;color:{color}">'
                    f'{html.escape(display)}</div>'
                    f'</div>'
                )

            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)

        # 兜底：computed 中有因子不在任何注册表分组里
        all_grouped = set()
        for names in factor_groups.values():
            all_grouped.update(names)
        orphan = {n: v for n, v in factors.items() if n not in all_grouped}
        if orphan:
            st.markdown('<div class="section-label">其他</div>', unsafe_allow_html=True)
            cards_html = '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin-bottom:8px">'
            for name, val in sorted(orphan.items(), key=lambda x: x[1], reverse=True):
                if val > 0:
                    bg, border, color = "#f0fdf4", "#10b981", "#047857"
                    display = f"+{val:.1f}"
                elif val < 0:
                    bg, border, color = "#fef2f2", "#ef4444", "#b91c1c"
                    display = f"{val:.1f}"
                else:
                    bg, border, color = "#ffffff", "#d1d5db", "#6b7280"
                    display = "0.0"
                tip = html.escape(factor_help.get(name, ""), quote=True)
                cards_html += (
                    f'<div title="{tip}" style="background:{bg};border:1px solid {border};'
                    f'border-radius:6px;padding:6px 8px;text-align:center;cursor:default">'
                    f'<div style="font-size:0.72rem;color:#64748b;margin-bottom:2px;'
                    f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
                    f'{html.escape(name)}</div>'
                    f'<div style="font-size:0.95rem;font-weight:700;color:{color}">'
                    f'{html.escape(display)}</div>'
                    f'</div>'
                )
            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)

                # 计算详情
        if factor_details:
            with st.expander("📐 计算详情", expanded=False):
                st.dataframe(
                    pd.DataFrame([{"因子": n, **{k: f"{v:+.3f}" for k,v in d.items()}} for n,d in factor_details.items()]),
                    use_container_width=True, hide_index=True,
                )
    
        # 评分走势图 — Plotly 竖线+悬浮
        chart_scores = scores[["时间", "评分"]].copy()
        chart_scores["时间"] = pd.to_datetime(chart_scores["时间"])
    
        col_scr1, col_scr2 = st.columns([3, 1])
        with col_scr1:
            days_back = st.select_slider("范围", options=[7,14,30,90,180,360], value=30, label_visibility="collapsed")
        with col_scr2:
            st.caption(f"近 {days_back} 天")
    
        cutoff = pd.Timestamp.now(tz=chart_scores["时间"].iloc[-1].tz) - pd.Timedelta(days=days_back)
        chart_scores = chart_scores[chart_scores["时间"] >= cutoff].set_index("时间").sort_index()
    
        import plotly.graph_objects as go
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=chart_scores.index, y=chart_scores["评分"], mode="lines",
            line=dict(color="#c9972b", width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>评分: %{y:+.1f}<extra></extra>",
        ))
        fig_s.add_hline(y=0, line=dict(color="#94a3b8", dash="dash", width=0.5))
        # 默认放大到最近 2/3 区域
        default_start = chart_scores.index[max(0, len(chart_scores) - max(3, len(chart_scores) * 2 // 3))]
        default_end = chart_scores.index[-1]
        fig_s.update_layout(
            **PLOTLY_LIGHT_LAYOUT,
            height=280, margin=dict(l=0,r=0,t=0,b=20),
            yaxis=dict(title=None, showgrid=True, gridcolor="#f1f5f9"),
            hovermode="x unified", showlegend=False,
            xaxis=dict(showgrid=False, rangeslider=dict(visible=False),
                       range=[default_start, default_end]),
            dragmode="pan",
        )
        st.plotly_chart(fig_s, use_container_width=True, config={
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "displaylogo": False,
            "scrollZoom": True,
        })
        st.caption(f"最新评分时间：{pd.to_datetime(latest['时间']).strftime('%Y-%m-%d %H:%M UTC')}。评分曲线为日频/快照数据，不代表逐秒行情。")
    
        # 风险提示 — 可折叠
        with st.expander(f"💡 风险提示（{len(risks)} 条）", expanded=len(risks) <= 3):
            for i, r in enumerate(risks):
                color = "#f0fdf4" if i % 2 == 0 else "#fffbeb"
                icon = "📌"
                st.markdown(
                    f'<div style="background:{color};border:1px solid #e2e8f0;border-radius:6px;'
                    f'padding:8px 14px;margin:4px 0;font-size:0.88rem;color:#475569;">'
                    f'{icon} {r}</div>',
                    unsafe_allow_html=True,
                )
    
        # AI 解读 — DeepSeek 分析
        with st.expander("🤖 AI 解读", expanded=False):
            try:
                ai = api("/ai/analysis")
                if ai.get("ok"):
                    analysis = ai.get("analysis", {})
                    # 概览
                    st.markdown(f"**市场概览**：{analysis.get('overview', '')}")
                    # 核心驱动因子
                    drivers = analysis.get("drivers", [])
                    if drivers:
                        st.markdown("**核心驱动因子**")
                        for d in drivers[:5]:
                            ico = {"利多": "🟢", "利空": "🔴", "中性": "⚪"}.get(d.get("impact", ""), "⚪")
                            st.markdown(f"- {ico} **{d.get('factor', '')}**（{d.get('impact', '')}）：{d.get('reason', '')}")
                    # 矛盾信号
                    contradictions = analysis.get("contradictions", [])
                    if contradictions:
                        st.markdown("**⚠️ 矛盾信号**")
                        for c in contradictions:
                            st.markdown(f"- {c}")
                    # AI 风险提示
                    ai_risks = analysis.get("risks", [])
                    if ai_risks:
                        st.markdown("**AI 风险提示**")
                        for r in ai_risks:
                            st.markdown(f"- {r}")
                    # 数据质量备注
                    quality = analysis.get("quality_notes", [])
                    if quality:
                        st.markdown("**数据质量备注**")
                        for q in quality:
                            st.markdown(f"- {q}")
                    ts_str = analysis.get('timestamp', '')
                    ts_display = ts_str[:16].replace('T', ' ') if ts_str else '—'
                    st.caption(f"模型：{analysis.get('model', 'DeepSeek')} · UTC {ts_display}")
                else:
                    st.caption(f"AI 分析暂不可用：{ai.get('error', '未知')}（需在 .env 中配置 DEEPSEEK_API_KEY）")
            except Exception as e:
                st.caption(f"AI 分析加载失败：{e}")
    
    # ═══════════════════════════════════════════
    # 预测
    # ═══════════════════════════════════════════
    
    st.divider()
    st.markdown('<a id="gold-predict"></a>', unsafe_allow_html=True)

