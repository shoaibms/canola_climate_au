# -*- coding: utf-8 -*-
"""
Canola Trial Site Climate Explorer
===================================
Streamlit + Plotly interactive dashboard
Phase 2 MVP -- click site to explore rainfall signature

Run with:
    cd C:/Users/ms/Desktop/canola/code/site_weather_code/version_3
    streamlit run canola_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

# -- Directory configuration --------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# -- Page config -------------------------------------------------------------
st.set_page_config(
    page_title="Canola Trial Site Climate Explorer",
    page_icon="canola",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 0.2rem !important; padding-left: 1.5rem; padding-right: 1.5rem; }
    header[data-testid="stHeader"] { height: 0rem; visibility: hidden; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .metric-box {
        background: #1e2130; border: 1px solid #2e3450;
        border-radius: 8px; padding: 10px 14px; text-align: center;
    }
    .metric-box .val { font-size: 1.5rem; font-weight: 700; color: #81d4a0; }
    .metric-box .lab { font-size: 0.72rem; color: #90a4ae; text-transform: uppercase; letter-spacing: 0.05em; }
    .note { font-size: 0.72rem; color: #546e7a; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# -- Data ---------------------------------------------------------------------
@st.cache_data
def load_data(clim_mtime, stats_mtime):
    clim  = pd.read_csv(DATA_DIR / "site_rainfall_climatology.csv")
    stats = pd.read_csv(DATA_DIR / "site_rainfall_stats.csv")
    return clim, stats

_clim_mtime  = (DATA_DIR / "site_rainfall_climatology.csv").stat().st_mtime
_stats_mtime = (DATA_DIR / "site_rainfall_stats.csv").stat().st_mtime
clim, stats  = load_data(_clim_mtime, _stats_mtime)

MONTHS        = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
SEASON_MONTHS = list(range(4, 11))
STATE_COLORS  = {"NSW": "#1D9E75", "VIC": "#378ADD", "WA": "#EF9F27"}
STATE_EDGE    = {"NSW": "#0F6E56", "VIC": "#185FA5", "WA": "#BA7517"}
REGIME_SHAPE  = {"WA": "diamond", "VIC": "square", "NSW": "circle"}
REGIME_LABEL  = {"WA": "Mediterranean (WA)", "VIC": "Transitional (VIC)", "NSW": "Continental (NSW)"}

with st.sidebar:
    st.markdown("### Controls")
    show_labels = st.toggle("Show site labels on map", value=True)
    st.markdown("---")
    st.markdown("<span class='note'>Climatology from BOM representative normals. "
                "Replace with real data via fetch_silo_data.py.</span>", unsafe_allow_html=True)

valid_stats = stats[stats["lat"].notna()].copy()

st.markdown("## Canola Trial Site Climate Explorer")
st.markdown("<span class='note'>Select a site to view its rainfall signature. "
            "Background = Apr--Oct median rainfall (canola growing season).</span>",
            unsafe_allow_html=True)

col_map, col_panel = st.columns([1.55, 1], gap="medium")

if "selected_site" not in st.session_state:
    st.session_state.selected_site = "NW"

# ===============================================================================
# LEFT COLUMN
# ===============================================================================
with col_map:

    # -- Map ------------------------------------------------------------------
    fig_map = go.Figure()

    for _, row in valid_stats.iterrows():
        if pd.isna(row["lat"]): continue
        is_sel = (row["site_code"] == st.session_state.selected_site)
        color  = STATE_COLORS.get(row["state"], "#ffffff")
        hover  = (f"<b>{row['site_name']} [{row['site_code']}]</b><br>"
                  f"State: {row['state']}<br>"
                  f"Apr--Oct: <b>{row['apr_oct_mm']:.0f} mm</b><br>"
                  f"Annual: {row['annual_median_mm']:.0f} mm  CV: {row['mean_cv']:.2f}")
        fig_map.add_trace(go.Scattergeo(
            lat=[row["lat"]], lon=[row["lon"]],
            mode="markers+text" if show_labels else "markers",
            text=[row["site_code"]], textposition="top center",
            textfont=dict(size=11, color="#eceff1"),
            marker=dict(size=16 if is_sel else 10, color=color,
                        line=dict(color="white", width=3 if is_sel else 1.5)),
            hovertemplate=hover + "<extra></extra>",
            name=row["site_code"],
        ))

    fig_map.update_layout(
        geo=dict(
            scope="world", resolution=50,
            lataxis_range=[-44, -9], lonaxis_range=[112, 155],
            showland=True, landcolor="#1c2433",
            showocean=True, oceancolor="#0d1117",
            showcoastlines=True, coastlinecolor="#455a64",
            showlakes=True, lakecolor="#0d1117",
            showcountries=False, bgcolor="#0f1117", showframe=False
        ),
        paper_bgcolor="#0f1117",
        margin=dict(l=0, r=0, t=0, b=0), height=460,
        showlegend=False,
        hoverlabel=dict(bgcolor="#1e2130", bordercolor="#2e3450",
                        font=dict(color="white", size=12))
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # -- Dropdown -------------------------------------------------------------
    available = valid_stats[["site_code","site_name","state"]].copy()
    available["label"] = (available["site_code"] + " -- " +
                          available["site_name"] + " (" + available["state"] + ")")
    default_pos = available.reset_index(drop=True).index[
        available["site_code"].reset_index(drop=True) ==
        st.session_state.selected_site].tolist()
    chosen = st.selectbox(
        "Select site to inspect", available["label"].tolist(),
        index=int(default_pos[0]) if default_pos else 0,
        key="site_dropdown"
    )
    st.session_state.selected_site = available[
        available["label"] == chosen]["site_code"].values[0]


# ===============================================================================
# RIGHT COLUMN
# ===============================================================================
with col_panel:
    sc        = st.session_state.selected_site
    site_info = valid_stats[valid_stats["site_code"] == sc].iloc[0]
    site_clim = (clim[(clim["site_code"] == sc) & clim["median_mm"].notna()]
                 .sort_values("month_num"))

    st.markdown(
        f"### {site_info['site_name']} "
        f"<span style='font-size:0.85rem;color:#90a4ae;'>[{sc}] -- {site_info['state']}</span>",
        unsafe_allow_html=True
    )

    mc1, mc2, mc3 = st.columns(3)
    season_pct = site_info["apr_oct_mm"] / site_info["annual_median_mm"] * 100
    for col, val, lab in [
        (mc1, f"{site_info['annual_median_mm']:.0f}", "Annual mm"),
        (mc2, f"{site_info['apr_oct_mm']:.0f}",       "Apr--Oct mm"),
        (mc3, f"{season_pct:.0f}%",                    "Season share"),
    ]:
        with col:
            st.markdown(f"<div class='metric-box'><div class='val'>{val}</div>"
                        f"<div class='lab'>{lab}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -- Chart 1: Monthly Climatology -- rainfall bars + temperature lines ------
    season_vals = site_clim[site_clim["month_num"].between(4,10)]["median_mm"]
    peak_val    = season_vals.max() if len(season_vals) > 0 else 1
    win_thresh  = peak_val * 0.60

    def bar_color(month_num, median_mm):
        if month_num not in SEASON_MONTHS: return "#546e7a"
        if median_mm >= win_thresh:        return "#185FA5"
        return "#378ADD"

    bar_colors = [bar_color(m, v)
                  for m, v in zip(site_clim["month_num"], site_clim["median_mm"])]

    # Gracefully handle both old (no temp) and new (with temp) climatology files
    has_temp = ("tmax_median" in site_clim.columns and
                site_clim["tmax_median"].notna().any() and
                "tmin_median" in site_clim.columns and
                site_clim["tmin_median"].notna().any())

    fig_clim = go.Figure()

    # Rainfall bars (primary y-axis)
    fig_clim.add_trace(go.Bar(
        x=site_clim["month"], y=site_clim["median_mm"],
        error_y=dict(type="data",
                     array=site_clim["std_mm"].tolist(),
                     arrayminus=np.minimum(
                         site_clim["std_mm"].values,
                         site_clim["median_mm"].clip(lower=0).values
                     ).tolist(),
                     visible=True, color="#546e7a", thickness=1.2),
        marker_color=bar_colors,
        yaxis="y",
        hovertemplate="<b>%{x}</b><br>Rain: %{y:.0f} mm +/- %{error_y.array:.0f}<extra></extra>",
        name="Rainfall"
    ))

    if has_temp:
        # Tmax line (secondary y-axis)
        fig_clim.add_trace(go.Scatter(
            x=site_clim["month"], y=site_clim["tmax_median"],
            mode="lines+markers",
            line=dict(color="#EF9F27", width=2),
            marker=dict(size=5, color="#EF9F27"),
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Tmax: %{y:.1f} C<extra></extra>",
            name="Tmax"
        ))
        # Tmin line (secondary y-axis)
        fig_clim.add_trace(go.Scatter(
            x=site_clim["month"], y=site_clim["tmin_median"],
            mode="lines+markers",
            line=dict(color="#1D9E75", width=2),
            marker=dict(size=5, color="#1D9E75"),
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Tmin: %{y:.1f} C<extra></extra>",
            name="Tmin"
        ))

    fig_clim.update_layout(
        title=dict(text="Monthly Climatology  --  Rainfall & Temperature",
                   font=dict(color="#cfd8dc", size=13), x=0),
        paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
        xaxis=dict(tickfont=dict(color="#90a4ae", size=11),
                   gridcolor="#2e3450", title=""),
        yaxis=dict(tickfont=dict(color="#90a4ae", size=11),
                   gridcolor="#2e3450",
                   title=dict(text="Rainfall (mm)", font=dict(color="#90a4ae", size=11))),
        yaxis2=dict(tickfont=dict(color="#90a4ae", size=11),
                    title=dict(text="Temp (C)", font=dict(color="#90a4ae", size=11)),
                    overlaying="y", side="right",
                    showgrid=False, zeroline=False) if has_temp else dict(
                    overlaying="y", side="right", showgrid=False,
                    visible=False, showticklabels=False),
        showlegend=False,
        margin=dict(l=50, r=55, t=70, b=35), height=290,
        barmode="overlay",
        annotations=[
            dict(
                text=("  <span style='color:#185FA5;font-weight:bold'> Peak rain window</span>"
                      "   <span style='color:#378ADD;font-weight:bold'> Growing season</span>"
                      "   <span style='color:#546e7a;font-weight:bold'> Off-season</span>"),
                x=0, y=1.16, xref="paper", yref="paper",
                showarrow=False, font=dict(size=10, color="#90a4ae")
            ),
            dict(
                text=("   <span style='color:#EF9F27;font-weight:bold'> Tmax</span>"
                      "   <span style='color:#1D9E75;font-weight:bold'> Tmin</span>"
                      if has_temp else ""),
                x=1, y=1.16, xref="paper", yref="paper",
                xanchor="right", showarrow=False,
                font=dict(size=10, color="#90a4ae")
            ),
        ]
    )
    st.plotly_chart(fig_clim, use_container_width=True, config={"displayModeBar": False})


    # -- Chart 3: Hierarchical Clustering (Dendrogram) --------------------------
    clim_valid = clim[clim["median_mm"].notna()]
    features = {}
    for site, grp in clim_valid.groupby("site_code"):
        grp = grp.sort_values("month_num")
        if len(grp) < 12: continue
        row_feat = []
        for col in ["median_mm", "std_mm", "p25", "p75"]:
            row_feat.extend(grp[col].fillna(0).tolist() if col in grp.columns else [0]*12)
        features[site] = row_feat

    feat_df = pd.DataFrame(features).T.dropna()

    if len(feat_df) >= 4:
        # -- Chart 2: Rainfall Reliability Bubble Chart (Plotly) ----------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<span style='color:#cfd8dc;font-size:0.95rem;font-weight:600'>"
            "Rainfall Reliability</span>"
            "<span style='color:#546e7a;font-size:0.88rem;'>"
            "  annual rainfall vs inter-annual variability</span>",
            unsafe_allow_html=True
        )

        bubble_data = valid_stats.copy()
        apr_vals    = bubble_data["apr_oct_mm"].values
        b_min, b_max = float(apr_vals.min()), float(apr_vals.max())

        # Scale Apr-Oct rain to marker size 10-40px diameter
        def bsz(v):
            return 10 + (v - b_min) / (b_max - b_min) * 30

        fig_bub = go.Figure()
        for _, s in bubble_data.iterrows():
            color     = STATE_COLORS.get(s["state"], "#888")
            edge      = STATE_EDGE.get(s["state"], "#555")
            is_hl     = s["site_code"] == sc
            show_name = s["site_name"]
            fig_bub.add_trace(go.Scatter(
                x=[s["annual_median_mm"]],
                y=[s["mean_cv"]],
                mode="markers+text",
                text=[s["site_code"]],
                textposition="top center",
                textfont=dict(size=11, color="#eceff1" if is_hl else "#90a4ae"),
                marker=dict(
                    size=bsz(s["apr_oct_mm"]),
                    color=color,
                    opacity=0.85,
                    line=dict(color="white" if is_hl else edge,
                              width=2.5 if is_hl else 0.8),
                ),
                name=s["state"],
                legendgroup=s["state"],
                showlegend=False,
                hovertemplate=(
                    f"<b>{show_name} [{s['site_code']}]</b><br>"
                    f"State: {s['state']}<br>"
                    f"Annual: {s['annual_median_mm']:.0f} mm<br>"
                    f"Apr-Oct: {s['apr_oct_mm']:.0f} mm<br>"
                    f"CV: {s['mean_cv']:.2f} (higher = less reliable)"
                    "<extra></extra>"
                )
            ))

        # State legend traces (invisible markers just for legend)
        for st_code in ["NSW","VIC","WA"]:
            fig_bub.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=STATE_COLORS[st_code],
                            line=dict(color=STATE_EDGE[st_code], width=0.8)),
                name=st_code, legendgroup=st_code, showlegend=True
            ))

        fig_bub.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
            xaxis=dict(
                title=dict(text="Annual median rainfall (mm)",
                           font=dict(color="#90a4ae", size=11)),
                tickfont=dict(color="#90a4ae", size=10),
                gridcolor="#2e3450"
            ),
            yaxis=dict(
                title=dict(text="CV (inter-annual variability)",
                           font=dict(color="#90a4ae", size=11)),
                tickfont=dict(color="#90a4ae", size=10),
                gridcolor="#2e3450"
            ),
            legend=dict(font=dict(color="#90a4ae", size=10),
                        bgcolor="rgba(0,0,0,0)",
                        title=dict(text="State", font=dict(color="#546e7a", size=10))),
            margin=dict(l=55, r=10, t=20, b=40), height=364,
            hoverlabel=dict(bgcolor="#1e2130", bordercolor="#2e3450",
                            font=dict(color="white", size=12))
        )
        st.plotly_chart(fig_bub, use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(
            "<span class='note'>Bubble size = Apr-Oct rainfall. "
            "X = annual total. Y = CV (higher = less reliable). "
            "Bold border = selected site. Hover for details.</span>",
            unsafe_allow_html=True
        )
        X        = StandardScaler().fit_transform(feat_df.values)
        dist_vec = pdist(X, metric="correlation")
        Z        = linkage(dist_vec, method="ward")

        N_CLUSTERS    = 3
        labels        = fcluster(Z, N_CLUSTERS, criterion="maxclust")
        CLUSTER_COLS  = {1: "#e74c3c", 2: "#f39c12", 3: "#2ecc71"}
        CLUSTER_NAMES = {1: "Group A", 2: "Group B", 3: "Group C"}

        clust_df = (pd.DataFrame({"site_code": feat_df.index, "cluster": labels})
                    .merge(valid_stats[["site_code","site_name","state",
                                        "apr_oct_mm","annual_median_mm","mean_cv"]],
                           on="site_code"))
        clust_df["highlight"] = clust_df["site_code"] == sc

        ddata      = dendrogram(Z, labels=list(feat_df.index),
                                color_threshold=0, no_plot=True)
        leaf_order = ddata["ivl"]
        n_leaves   = len(leaf_order)
        leaf_y     = [5.0 * (i + 1) for i in range(n_leaves)]

        fig_dend = go.Figure()
        for xs, ys in zip(ddata["icoord"], ddata["dcoord"]):
            fig_dend.add_trace(go.Scatter(
                x=ys, y=xs, mode="lines",
                line=dict(color="#378ADD", width=1.8),
                hoverinfo="skip", showlegend=False
            ))
        for i, code in enumerate(leaf_order):
            r = clust_df[clust_df["site_code"] == code]
            if r.empty: continue
            r = r.iloc[0]
            fig_dend.add_trace(go.Scatter(
                x=[0], y=[leaf_y[i]], mode="markers+text",
                text=[f"  {code}"], textposition="middle right",
                textfont=dict(size=11, color="#ffffff" if r["highlight"] else "#cfd8dc"),
                marker=dict(size=10 if r["highlight"] else 7,
                            color=CLUSTER_COLS.get(r["cluster"], "#90a4ae"),
                            line=dict(color="white", width=2 if r["highlight"] else 0)),
                hovertemplate=(f"<b>{r['site_name']} [{code}]</b><br>"
                               f"State: {r['state']}<br>"
                               f"Apr-Oct: {r['apr_oct_mm']:.0f} mm<br>"
                               f"Cluster: {CLUSTER_NAMES[r['cluster']]}"
                               "<extra></extra>"),
                showlegend=False
            ))

        # -- Cluster bracket labels on right side -----------------------------
        # Group leaves by cluster, draw a vertical bracket + label per group
        from collections import defaultdict
        cluster_leaf_positions = defaultdict(list)
        for i, code in enumerate(leaf_order):
            r = clust_df[clust_df["site_code"] == code]
            if r.empty: continue
            cluster_leaf_positions[int(r.iloc[0]["cluster"])].append(leaf_y[i])

        # Max x of dendrogram segments (right edge of tree)
        all_x = [x for xs in ddata["dcoord"] for x in xs]
        x_max = max(all_x) if all_x else 1.0
        bracket_x = x_max * 1.08   # bracket sits just right of tree
        label_x   = x_max * 1.22   # label sits further right

        for cl_id, positions in sorted(cluster_leaf_positions.items()):
            y_bot  = min(positions) - 1.5
            y_top  = max(positions) + 1.5
            y_mid  = (y_bot + y_top) / 2
            color  = CLUSTER_COLS.get(cl_id, "#90a4ae")
            cname  = CLUSTER_NAMES.get(cl_id, f"Group {cl_id}")

            # Vertical bracket line
            fig_dend.add_shape(type="line",
                x0=bracket_x, x1=bracket_x, y0=y_bot, y1=y_top,
                line=dict(color=color, width=3),
                xref="x", yref="y"
            )
            # Top tick
            fig_dend.add_shape(type="line",
                x0=bracket_x, x1=bracket_x * 0.98, y0=y_top, y1=y_top,
                line=dict(color=color, width=2),
                xref="x", yref="y"
            )
            # Bottom tick
            fig_dend.add_shape(type="line",
                x0=bracket_x, x1=bracket_x * 0.98, y0=y_bot, y1=y_bot,
                line=dict(color=color, width=2),
                xref="x", yref="y"
            )
            # Group label
            fig_dend.add_annotation(
                x=label_x, y=y_mid,
                text=f"<b>{cname}</b>",
                showarrow=False, xref="x", yref="y",
                font=dict(size=12, color=color),
                xanchor="left", yanchor="middle"
            )

        fig_dend.update_layout(
            title=dict(text="Rainfall Environment Clustering",
                       font=dict(color="#cfd8dc", size=14), x=0),
            paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
            xaxis=dict(title=dict(text="Distance (dissimilarity)",
                                  font=dict(color="#90a4ae", size=11)),
                       tickfont=dict(color="#90a4ae", size=10),
                       gridcolor="#2e3450", zeroline=False, side="top"),
            yaxis=dict(showticklabels=False, gridcolor="#2e3450",
                       zeroline=False, range=[0, (n_leaves + 1) * 5.0]),
            margin=dict(l=10, r=130, t=50, b=40), height=320,
            showlegend=False,
            hoverlabel=dict(bgcolor="#1e2130", bordercolor="#2e3450",
                            font=dict(color="white", size=12))
        )
        st.plotly_chart(fig_dend, use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(
            "<span class='note'>Ward linkage, correlation distance on 48 features "
            "(median + std + P25 + P75 per month). Leaf colour = cluster. "
            "Highlighted = selected site.</span>",
            unsafe_allow_html=True
        )