"""宏观数据展示组件"""
from __future__ import annotations
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
import plotly.express as px

import streamlit as st
import sys

def render_data_sections(**kwargs):
    """渲染宏观指标、CFTC、央行购金、事件、新闻情绪、评分进化"""
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
    st.subheader("宏观指标")
    
    macro = get_macro()
    if not macro.empty:
        import plotly.express as px
        series_ids = macro["series_id"].unique()
        preferred_series = [
            "DFII5", "DFII10", "DFII30", "THREEFYTP10",
            "WALCL", "WDTGAL", "RRPONTSYD", "WRESBAL",
            "GFDEGDQ188S", "FYFSD",
        ]
        default_series = [sid for sid in preferred_series if sid in series_ids][:5] or list(series_ids[:4])
        selected = st.multiselect("选择指标", series_ids, default=default_series)
        if selected:
            mdf = _finite_chart_frame(
                macro[macro["series_id"].isin(selected)],
                required_cols=["时间", "值"],
                numeric_cols=["值"],
            )
            if not mdf.empty:
                macro_chart = px.line(mdf, x="时间", y="值", color="指标")
                macro_chart.update_layout(
                    **PLOTLY_LIGHT_LAYOUT,
                    height=220,
                    margin=dict(l=0, r=0, t=8, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    yaxis_title=None,
                    xaxis_title=None,
                )
                st.plotly_chart(macro_chart, use_container_width=True)
            else:
                st.caption("所选宏观指标暂无可绘制数值。")
    
    external_df = get_external_indicators()
    if not external_df.empty:
        with st.expander("ETF / COMEX / 期权 / 地缘 / 实物需求外部指标"):
            st.dataframe(external_df, use_container_width=True, hide_index=True)
    
    with st.expander("手动录入外部指标", expanded=False):
        catalog_payload = api("/external/indicators/catalog")
        catalog_rows = catalog_payload.get("data", []) if catalog_payload.get("ok") else []
        manual_candidates = [
            row for row in catalog_rows
            if row.get("indicator_id") in {
                "COMEX_REGISTERED_GOLD_OZ",
                "COMEX_GOLD_FRONT_SPREAD_PCT",
                "GEO_RISK_INTENSITY",
                "INDIA_CHINA_PHYSICAL_DEMAND",
                "GLD_FLOW_TONNES",
                "GOLD_OPTION_IV_30D",
                "GOLD_OPTION_SKEW_25D",
            }
        ]
        if not manual_candidates:
            st.caption("暂无可手动录入的外部指标目录。")
        else:
            # ── 按类别分组，带 scored/gray 标记 ──
            group_order = ["ETF", "期货结构", "期权", "风险事件", "实物需求"]
            grouped: dict[str, list[dict]] = {}
            for row in manual_candidates:
                grouped.setdefault(row.get("category", "其他"), []).append(row)
    
            label_to_meta: dict[str, dict] = {}
            select_options: list[str] = []
            for grp in group_order:
                items = grouped.get(grp)
                if not items:
                    continue
                select_options.append(f"── {grp} ──")
                for row in items:
                    scored_badge = "📊" if row.get("scored") else "⬜"
                    lbl = f"{scored_badge} {row.get('name')}  [{row.get('indicator_id')}]"
                    label_to_meta[lbl] = row
                    select_options.append(lbl)
    
            selected_label = st.selectbox(
                "选择指标",
                select_options,
                index=min(1, len(select_options)-1) if select_options and select_options[0].startswith("──") else 0,
                key="manual_indicator_select",
            )
            # 跳过分组标题行
            if selected_label.startswith("──"):
                selected_label = next((o for o in select_options if not o.startswith("──")), select_options[-1])
            selected_meta = label_to_meta.get(selected_label, manual_candidates[0])
    
            # ── 显示该指标最新已入库值 ──
            last_val = None
            if not external_df.empty:
                prev_rows = external_df[external_df["指标ID"] == selected_meta.get("indicator_id")]
                if not prev_rows.empty:
                    latest = prev_rows.iloc[0]
                    last_val = latest.get("value")
                    last_ts = latest.get("timestamp", "")
                    last_src = latest.get("source", "")
                    st.caption(
                        f"📌 最近记录: **{last_val}** {selected_meta.get('unit','')}"
                        f" · {last_ts} · 来源: {last_src}"
                    )
    
            # ── 输入区 ──
            m1, m2, m3 = st.columns([5, 3, 3])
            with m1:
                manual_value = st.number_input(
                    f"数值（{selected_meta.get('unit') or ''}）",
                    value=None,
                    placeholder="输入数值…",
                    key="manual_indicator_value",
                )
            with m2:
                manual_date = st.date_input("日期", value=_now_beijing().date(), key="manual_indicator_date")
            with m3:
                manual_source = st.text_input("来源", value="MANUAL", key="manual_indicator_source")
            manual_note = st.text_input("备注（可选）", value="", key="manual_indicator_note")
    
            # ── 说明 ──
            reason = str(selected_meta.get("reason") or "")
            score_status = "✅ 参与评分" if selected_meta.get("scored") else "⬜ 仅展示，不参与评分"
            st.caption(f"{score_status} · {reason}")
    
            # ── 按钮行 ──
            b1, b2, b3 = st.columns([2, 1, 1])
            with b1:
                if st.button("💾 保存外部指标", use_container_width=True, type="primary"):
                    if manual_value is None:
                        st.error("请输入数值")
                    else:
                        ts = dt.datetime.combine(manual_date, dt.time.min, tzinfo=UTC_TZ).isoformat()
                        result = api(
                            "/external/indicators",
                            "post",
                            json={
                                "indicator_id": selected_meta.get("indicator_id"),
                                "timestamp": ts,
                                "value": manual_value,
                                "source": manual_source or "MANUAL",
                                "name": selected_meta.get("name"),
                                "category": selected_meta.get("category"),
                                "unit": selected_meta.get("unit"),
                                "note": manual_note,
                            },
                        )
                        if result.get("ok"):
                            st.toast(f"✅ 已保存 {selected_meta.get('name')}: {manual_value}", icon="✅")
                            st.cache_data.clear()
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"保存失败：{result.get('reason', '未知错误')}")
            with b3:
                if st.button("🔄 重置", use_container_width=True):
                    for k in list(st.session_state.keys()):
                        if isinstance(k, str) and k.startswith("manual_indicator_"):
                            del st.session_state[k]
                    st.rerun()
    
    # ═══════════════════════════════════════════
    # CFTC（投机情绪温度计）
    # ═══════════════════════════════════════════
    
    st.divider()
    st.markdown('<a id="cftc"></a>', unsafe_allow_html=True)
    st.subheader("CFTC 投机情绪温度计")
    cftc = get_cftc()
    if not cftc.empty and "净多占比%" in cftc.columns:
        import plotly.graph_objects as go
        import numpy as np

        ratio = cftc["净多占比%"].iloc[-1]
        if ratio >= 55:
            zone = "🟡 极端偏多区域（拥挤交易风险）"
        elif ratio >= 40:
            zone = "🟢 正常偏多区域"
        elif ratio >= 25:
            zone = "⚪ 中性区域"
        else:
            zone = "🔴 偏空区域"

        st.caption(
            f"当前净多占比 **{ratio:.1f}%** — {zone}  "
            f"· 非商业持仓反映投机资金的多空倾向。"
            f'净多占比>55% 通常意味着“拥挤交易”，反向风险上升；<25% 则反映极度悲观。'
        )

        # ── 主图：净多占比% 折线 + 背景色温区 ──
        fig = go.Figure()

        # 背景色带
        fig.add_hrect(y0=55, y1=100, fillcolor="#fbbf24", opacity=0.08, line_width=0, name="极端偏多")
        fig.add_hrect(y0=40, y1=55, fillcolor="#10b981", opacity=0.06, line_width=0, name="偏多")
        fig.add_hrect(y0=25, y1=40, fillcolor="#d1d5db", opacity=0.06, line_width=0, name="中性")
        fig.add_hrect(y0=0, y1=25, fillcolor="#ef4444", opacity=0.06, line_width=0, name="偏空")

        # 占比折线
        fig.add_trace(go.Scatter(
            x=cftc["时间"], y=cftc["净多占比%"],
            mode="lines+markers",
            name="净多占比%",
            line=dict(color="#f59e0b", width=2.5),
            marker=dict(size=6, color="#f59e0b"),
            hovertemplate="%{x|%Y-%m-%d}<br>净多占比: %{y:.1f}%<extra></extra>",
        ))

        # 参考线
        fig.add_hline(y=55, line_dash="dash", line_color="#fbbf24", line_width=1, annotation_text="55% 拥挤线")
        fig.add_hline(y=25, line_dash="dash", line_color="#ef4444", line_width=1, annotation_text="25% 偏空线")

        fig.update_layout(
            **PLOTLY_LIGHT_LAYOUT,
            height=180,
            margin=dict(l=0, r=0, t=8, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="净多占比 %", range=[0, min(70, max(cftc["净多占比%"]) + 15)], showgrid=True, gridcolor="#f1f5f9"),
            hovermode="x unified",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── 副图：净多合约数（柱状） ──
        fig2 = go.Figure()
        net_colors = ["#10b981" if v >= 0 else "#ef4444" for v in cftc["净多"]]
        fig2.add_trace(go.Bar(
            x=cftc["时间"], y=cftc["净多"],
            marker_color=net_colors,
            name="净多合约数",
            hovertemplate="%{x|%Y-%m-%d}<br>净多: %{y:,} 合约<extra></extra>",
        ))
        fig2.update_layout(
            **PLOTLY_LIGHT_LAYOUT,
            height=140,
            margin=dict(l=0, r=0, t=0, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="净多合约数", showgrid=True, gridcolor="#f1f5f9"),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── 数据表（折叠） ──
        with st.expander("持仓明细", expanded=False):
            show = cftc.sort_values("时间", ascending=False).copy()
            show["时间"] = show["时间"].apply(lambda t: t.strftime("%Y-%m-%d") if hasattr(t, "strftime") else str(t)[:10])
            for col in ["多", "空", "净多", "总持仓"]:
                show[col] = show[col].apply(lambda x: f"{x:,.0f}")
            show["净多占比%"] = show["净多占比%"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(show[["时间", "多", "空", "净多", "净多占比%", "总持仓"]], use_container_width=True, hide_index=True)
    else:
        st.caption("暂无 CFTC 数据。")
    
    # ═══════════════════════════════════════════
    # 央行购金
    # ═══════════════════════════════════════════
    
    st.divider()
    st.markdown('<a id="central-bank"></a>', unsafe_allow_html=True)
    st.subheader("央行购金")
    global_cb, country_cb = get_cb_gold()
    if (
        SETTINGS.production_mode
        and not global_cb.empty
        and global_cb["来源"].astype(str).str.upper().isin(LOW_CONFIDENCE_SOURCES).all()
        and not SETTINGS.show_low_confidence_data
    ):
        st.caption("央行购金数据来源：WGC/IMF IFS，基于季度报告按月估算。")
    elif not global_cb.empty:
        import plotly.express as px
        cb_plot = _finite_chart_frame(
            global_cb,
            required_cols=["月份", "净购金(吨)"],
            numeric_cols=["净购金(吨)"],
        )
        if not cb_plot.empty:
            cb_chart = px.bar(cb_plot, x="月份", y="净购金(吨)", hover_data=["来源"])
            cb_chart.update_traces(marker_color="#c9972b")
            cb_chart.update_layout(
                **PLOTLY_LIGHT_LAYOUT,
                height=190,
                margin=dict(l=0, r=0, t=8, b=20),
                yaxis_title="吨",
                xaxis_title=None,
                showlegend=False,
            )
            st.plotly_chart(cb_chart, use_container_width=True)
        else:
            st.caption("央行购金暂无可绘制数值。")
    else:
        st.caption("央行购金可验证来源待接入。")
    
    # ═══════════════════════════════════════════
    # 宏观事件
    # ═══════════════════════════════════════════
    
    st.divider()
    st.markdown('<a id="macro-events"></a>', unsafe_allow_html=True)
    st.subheader("宏观事件")
    events = get_events()
    if not events.empty:
        evt_df = events[["时间", "事件", "国家", "重要性"]].copy()
        evt_df["时间"] = evt_df["时间"].apply(lambda t: t.strftime("%Y-%m-%d %H:%M") if hasattr(t, "strftime") else str(t)[:16])
        st.dataframe(evt_df, use_container_width=True, hide_index=True)
    else:
        st.caption("未来60天暂无宏观事件。")
    
    # ═══════════════════════════════════════════
    # 新闻情绪 (VADER 情感分析)
    # ═══════════════════════════════════════════
    
    st.divider()
    st.markdown('<a id="news-sentiment"></a>', unsafe_allow_html=True)
    st.subheader("新闻情绪")
    sent_score, sent_df, daily_trend = get_sentiment()
    st.caption(
        f"情感引擎：VADER NLP \u00b7 新闻源：NewsAPI（每日限额 {SETTINGS.newsapi_daily_limit} 次）"
    )
    if sent_score is None:
        st.caption("暂无 NewsAPI/GDELT 新闻情绪数据。")
    else:
        # ── 情绪热力条 ──
        clamped = max(-5, min(5, sent_score))
        pct = (clamped + 5) / 10
        bar_html = (
            '<div style="margin:8px 0">'
            '<div style="display:flex;align-items:center;gap:10px">'
            '<span style="font-size:0.8rem;color:#64748b">利空</span>'
            '<div style="flex:1;height:14px;border-radius:7px;'
            'background:linear-gradient(to right,#ef4444,#fbbf24 25%,#d1d5db 50%,#86efac 75%,#10b981)"></div>'
            '<span style="font-size:0.8rem;color:#64748b">利多</span>'
            '</div>'
            '<div style="position:relative;height:20px;margin-top:-17px">'
            '<div style="position:absolute;left:' + str(round(pct*100)) + '%;transform:translateX(-50%)">'
            '<div style="font-size:1.2rem;font-weight:800;color:#1e293b;text-align:center">' + ('%.2f' % clamped) + '</div>'
            '<div style="width:2px;height:14px;background:#1e293b;margin:0 auto;border-radius:1px"></div>'
            '</div></div></div>')
        st.markdown(bar_html, unsafe_allow_html=True)

        # ── 关键指标卡 ──
        c1, c2, c3, c4 = st.columns(4)
        direction = "利多 \u25b2" if sent_score > 0.5 else ("利空 \u25bc" if sent_score < -0.5 else "中性 \u2015")
        c1.metric("信号方向", direction)
        if daily_trend and len(daily_trend) >= 2:
            today = daily_trend[-1]
            yesterday = daily_trend[-2]
            change = today["avg_score"] - yesterday["avg_score"]
            c2.metric("日趋势", f"{change:+.2f}", f"{today['avg_score']:+.2f}")
            c3.metric("今日篇数", today["count"])
            c4.metric("看多占比", f"{today['bullish_pct']:.0%}")
        else:
            c2.metric("日趋势", "\u2014")
            c3.metric("篇数", len(sent_df) if not sent_df.empty else 0)
            c4.metric("看多占比", "\u2014")

        # ── 文章散点分布 ──
        if not sent_df.empty:
            recent = sent_df.tail(50).copy()
            dots = []
            for _, row in recent.iterrows():
                ts = row["时间"]
                s = row["情绪"]
                cd = "#10b981" if s > 0 else ("#ef4444" if s < 0 else "#94a3b8")
                ttl = str(row.get("标题", ""))[:40]
                dots.append(
                    '<span title="' + str(ts)[:19] + ': ' + ("%.2f" % s) + ' | ' + ttl + '"'
                    ' style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' + cd + ';'
                    'margin:1px;cursor:default"></span>'
                )
            st.markdown('<div style="line-height:1;margin-top:6px">' + ''.join(dots) + '</div>', unsafe_allow_html=True)
            st.caption(f"每条圆点 = 一篇新闻（绿=利多，红=利空，灰=中性），悬停查看详情 \u00b7 共 {len(recent)} 篇")

        # ── 可折叠新闻列表 ──
        if not sent_df.empty:
            with st.expander("新闻列表", expanded=False):
                show = sent_df.sort_values("时间", ascending=False).copy()
                show["时间"] = show["时间"].apply(
                    lambda t: t.strftime("%Y-%m-%d %H:%M") if hasattr(t, "strftime") else str(t)[:16]
                )
                def _mk(row):
                    url = row.get("来源", ""); title = row["标题"]
                    if url and str(url).startswith("http"):
                        return '<a href="' + html.escape(url, quote=True) + '" target="_blank">' + html.escape(title) + '</a>'
                    return html.escape(title)
                show["标题"] = show.apply(_mk, axis=1)
                show["情绪"] = show["情绪"].apply(lambda x: f"{x:+.2f}")
                st.write(show[["时间","标题","情绪","数据源"]].to_html(index=False,escape=False), unsafe_allow_html=True)

    
    st.markdown('<a id="score-evolution"></a>', unsafe_allow_html=True)
    # 评分模型自我进化
    # ═══════════════════════════════════════════
    
    st.divider()
    with st.expander("评分模型自我进化", expanded=False):
        st.caption(
            "通过随机搜索 + 滚动回测自动寻找最优因子权重。"
            "每次优化保存一个参数版本，可追溯、可回滚。"
        )
    
        col_opt1, col_opt2, col_opt3 = st.columns([2, 1, 1])
        with col_opt1:
            n_iter = st.slider("搜索迭代次数", 20, 200, 50, 10, key="opt_n_iter")
        with col_opt2:
            horizon_days = st.selectbox("回测展望期（天）", [10, 20, 30, 60], index=1, key="opt_horizon")
        with col_opt3:
            do_opt = st.button("🚀 开始优化", use_container_width=True, type="primary")
    
        if do_opt:
            with st.spinner(f"随机搜索 {n_iter} 次，评估 {horizon_days} 天命中率（约需 {n_iter*5//60} 分钟）..."):
                import httpx
                try:
                    resp = httpx.post(
                        f"{API_BASE_URL}/score/optimize",
                        params={"n_iter": n_iter, "horizon_days": horizon_days},
                        timeout=httpx.Timeout(600),
                        trust_env=False,
                    )
                    r = resp.json() if resp.status_code == 200 else {}
                except Exception:
                    r = {}
            if r.get("ok"):
                best = r.get("best", {})
                baseline = r.get("baseline") or {}
                best_hr = best.get("hit_rate")
                base_hr = baseline.get("hit_rate")
                st.success(f"优化完成！版本 {r.get('version', '?')}")
                c1, c2 = st.columns(2)
                c1.metric("最优命中率", f"{best_hr*100:.1f}%" if best_hr else "—")
                c2.metric("默认命中率", f"{base_hr*100:.1f}%" if base_hr else "—")
            else:
                st.error(r.get("reason", "优化失败"))
    
        # 显示历史版本
        @st.cache_data(ttl=120)
        def get_param_versions() -> pd.DataFrame:
            payload = api("/score/params")
            rows = payload.get("data", []) if payload.get("ok") else []
            return pd.DataFrame([{
                "版本": r.get("version"),
                "命中率": f"{r.get('hit_rate')*100:.1f}%" if r.get("hit_rate") else "—",
                "样本数": r.get("sample_count") or "—",
                "激活": "✅" if r.get("is_active") else "",
                "创建时间": str(r.get("created_at") or "")[:16].replace("T", " "),
                "备注": r.get("notes") or "",
            } for r in rows])
    
        @st.cache_data(ttl=300)
        def get_param_compare(version: str) -> dict:
            return api(f"/score/params/{version}/compare")
    
        versions = get_param_versions()
        if not versions.empty:
            st.caption("参数版本历史")
            st.dataframe(versions, use_container_width=True, hide_index=True)
    
            # 激活/回滚
            col_a1, col_a2, col_a3 = st.columns([2, 1, 1])
            with col_a1:
                active_ver = st.selectbox("选择版本", versions["版本"].tolist(), key="activate_ver")
            with col_a2:
                if st.button("✅ 激活", use_container_width=True):
                    rr = api(
                        f"/score/params/{active_ver}/activate",
                        "post",
                        json={"operator": "dashboard", "reason": "仪表盘调试覆盖激活"},
                    )
                    if rr.get("ok"):
                        st.success(f"已激活 {active_ver}")
                        risk = rr.get("overfit_risk") or {}
                        if risk.get("not_recommended_for_direct_activation"):
                            st.warning("该版本存在过拟合风险：" + "；".join(risk.get("warnings", [])))
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(rr.get("reason", "失败"))
            with col_a3:
                if st.button("🔄 恢复默认", use_container_width=True):
                    rr = api(
                        "/score/params/deactivate",
                        "post",
                        json={"operator": "dashboard", "reason": "仪表盘调试覆盖恢复默认评分规则"},
                    )
                    if rr.get("ok"):
                        st.success("已恢复默认规则 v2")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(rr.get("reason", "失败"))
    
            compare = get_param_compare(active_ver)
            if compare.get("ok"):
                detail = compare.get("data", {})
                candidate = detail.get("candidate") or {}
                baseline = detail.get("baseline") or {}
                risk = detail.get("overfit_risk", {})
                st.markdown("#### 候选对比详情")
                metric_rows = [
                    ("保存命中率", None, candidate.get("stored_hit_rate")),
                    ("命中率", baseline.get("hit_rate"), candidate.get("hit_rate")),
                    ("信号覆盖率", baseline.get("signal_ratio"), candidate.get("signal_ratio")),
                    ("方向样本数", baseline.get("signal_count"), candidate.get("signal_count")),
                    ("多头样本", baseline.get("long_signal_count"), candidate.get("long_signal_count")),
                    ("空头样本", baseline.get("short_signal_count"), candidate.get("short_signal_count")),
                    ("尾部收益", baseline.get("worst_decile_return"), candidate.get("worst_decile_return")),
                    ("近期窗口命中率", baseline.get("recent_hit_rate"), candidate.get("recent_hit_rate")),
                    ("相对baseline提升", 0, candidate.get("baseline_lift")),
                ]
                st.dataframe(
                    pd.DataFrame([
                        {"指标": name, "baseline": base, "candidate": cand}
                        for name, base, cand in metric_rows
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )
                if risk.get("level") == "high":
                    st.warning("过拟合风险高：" + "；".join(risk.get("warnings", [])))
                elif risk.get("level") == "medium":
                    st.info("过拟合风险提示：" + "；".join(risk.get("warnings", [])))
                else:
                    st.caption("过拟合检查：" + "；".join(risk.get("warnings", [])))
                st.caption(detail.get("recommendation", ""))
    
    st.divider()
    audits = api("/models/activation-audit", params={"limit": 20})
    if audits.get("ok") and audits.get("data"):
        with st.expander("激活审计记录", expanded=False):
            audit_df = pd.DataFrame(audits["data"])
            show_cols = ["created_at", "model_type", "action", "from_version", "to_version", "operator", "reason"]
            st.dataframe(audit_df[[c for c in show_cols if c in audit_df.columns]], use_container_width=True, hide_index=True)
    
    # ═══════════════════════════════════════════
    # 底部状态
    # ═══════════════════════════════════════════
    
    st.divider()
    st.caption(
        "本系统用于验证 AI 黄金预测系统可行性，不用于黄金买卖参考。"
        f" 数据刷新间隔 {st.session_state['_rf_interval']}s · "
        f"UTC {_now_utc().strftime('%Y-%m-%d %H:%M')} / 北京 {_now_beijing().strftime('%Y-%m-%d %H:%M:%S')}"
    )
