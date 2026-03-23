"""
Immunological Supply Chain — Streamlit Dashboard
=================================================
PES University Capstone  PW26_RGP_01

Run:  streamlit run app.py
"""

import os, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Immunological Supply Chain",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

BASE   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(BASE, "output")
FIGS   = os.path.join(OUT,  "figures")
MODELS = os.path.join(BASE, "models")
EXTRA  = os.path.join(BASE, "data", "supplementary")

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label { color: #888; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { color: #e0e0ff; font-size: 1.6rem; font-weight: 700; margin-top: 0.2rem; }
    .metric-card .sub   { color: #666; font-size: 0.75rem; margin-top: 0.1rem; }
    h1 { color: #c8c8ff !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


# ── Data loaders (cached) ──────────────────────────────────────────────────
@st.cache_data
def load_anomalies():
    p = os.path.join(OUT, "anomalies.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_risk():
    p = os.path.join(OUT, "risk_scores.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_gnn():
    p = os.path.join(OUT, "gnn_risk_scores.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_stress():
    p = os.path.join(OUT, "macro_stress_scores.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"])
    return df

@st.cache_data
def load_forecast():
    p = os.path.join(OUT, "stress_forecast.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"])
    return df

@st.cache_data
def load_recovery_preds():
    p = os.path.join(OUT, "recovery_predictions.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_disruption_data():
    p = os.path.join(EXTRA, "disruption_processed.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

@st.cache_resource
def load_graph():
    p = os.path.join(MODELS, "supplychain_graph.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_rf_regressor():
    p = os.path.join(MODELS, "recovery_regressor.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_rf_classifier():
    p = os.path.join(MODELS, "recovery_classifier.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_embeddings():
    p = os.path.join(MODELS, "node_embeddings.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ── Feature names used by RF models ───────────────────────────────────────
RF_FEATURES = [
    "disruption_type_enc", "industry_enc", "supplier_region_enc",
    "supplier_size_enc",   "response_type_enc", "disruption_severity",
    "production_impact_pct", "has_backup_supplier",
]

# ── Header ─────────────────────────────────────────────────────────────────
st.title("Immunological Supply Chain")
st.caption("Self-Healing Supply Chains with AI Digital Antibodies  |  PES University  |  PW26_RGP_01")
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Overview",
    "Supply Chain Graph",
    "Anomaly Detection",
    "Risk Analysis",
    "Macro Stress & LSTM",
    "Recovery Predictor",
    "SHAP Explainability",
    "Multi-Domain Risk",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    G    = load_graph()
    anom = load_anomalies()
    risk = load_risk()
    gnn  = load_gnn()
    fc   = load_forecast()

    n_nodes   = G.number_of_nodes()   if G    else "—"
    n_edges   = G.number_of_edges()   if G    else "—"
    n_anom    = len(anom[anom["anomaly_score"] >= 2]) if not anom.empty else "—"
    n_high    = len(risk[risk["risk_score"] > 0.7])   if not risk.empty else "—"

    forecast_level = "—"
    if not fc.empty:
        upcoming = fc[fc["type"] == "forecast"]
        if not upcoming.empty:
            forecast_level = upcoming.iloc[0]["stress_level"]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Nodes in Graph",    f"{n_nodes:,}" if isinstance(n_nodes, int) else n_nodes)
    with c2: metric_card("Edges in Graph",    f"{n_edges:,}" if isinstance(n_edges, int) else n_edges)
    with c3: metric_card("High-Conf Anomalies", f"{n_anom:,}"  if isinstance(n_anom, int)  else n_anom, "score >= 2")
    with c4: metric_card("High-Risk Entities",  f"{n_high:,}"  if isinstance(n_high, int)  else n_high, "risk > 0.70")
    with c5: metric_card("Next-Week Forecast",  forecast_level, "macro stress level")

    st.markdown("#### Pipeline Architecture")
    stages = [
        ("1",  "Preprocessing",              "Universal CSV ingestion + column mapping"),
        ("2",  "Graph Construction",          "NetworkX DiGraph  |  manufacturer → distributor → retailer"),
        ("3",  "Anomaly Detection",           "4-signal Z-score  |  dynamic thresholds from Stage 9"),
        ("4",  "Risk Analysis",              "Betweenness + PageRank + in-degree centrality"),
        ("5",  "Disruption Routing",         "Dijkstra-based recovery re-routing"),
        ("6",  "Visualization",              "Static figure export  (fig1–fig4)"),
        ("7",  "GNN Node Encoder",           "2-layer GCN autoencoder  |  16-dim embeddings"),
        ("8",  "Recovery Predictor",         "Random Forest regressor + classifier"),
        ("9",  "Macro Stress Scorer",        "5-signal freight indicator weighting"),
        ("10", "LSTM Forecaster",            "Seq2seq LSTM  |  4-week stress forecast"),
        ("11", "PPO Routing Agent",          "Proximal Policy Optimisation  |  risk-aware re-routing"),
    ]
    for num, name, desc in stages:
        st.markdown(
            f"<div style='padding:0.35rem 0; border-bottom:1px solid #1e1e3a'>"
            f"<span style='color:#4fc3f7;font-weight:600;min-width:2rem;display:inline-block'>S{num}</span>"
            f"<span style='color:#ddd;min-width:14rem;display:inline-block'>{name}</span>"
            f"<span style='color:#777;font-size:0.82rem'>{desc}</span></div>",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — SUPPLY CHAIN GRAPH
# ══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    G    = load_graph()
    risk = load_risk()

    if G is None:
        st.warning("Run the pipeline first to generate the graph.")
    else:
        risk_lookup = dict(zip(risk["entity"], risk["risk_score"])) if not risk.empty else {}

        node_types = {n: d.get("type", "unknown") for n, d in G.nodes(data=True)}
        mfrs  = [n for n, t in node_types.items() if t == "manufacturer"]
        dists = [n for n, t in node_types.items() if t == "distributor"]

        # Subgraph: all manufacturers + distributors + their top-3 connected retailers
        sub_nodes = set(mfrs + dists)
        for d in dists:
            retailer_succs = sorted(
                [n for n in G.successors(d) if node_types.get(n) == "retailer"],
                key=lambda r: risk_lookup.get(r, 0), reverse=True
            )[:3]
            sub_nodes.update(retailer_succs)

        SG  = G.subgraph(sub_nodes)
        pos = nx.spring_layout(SG, seed=42, k=1.5)

        colour_map = {"manufacturer": "#f4a261", "distributor": "#4fc3f7", "retailer": "#a8d8a8"}

        edge_x, edge_y = [], []
        for u, v in SG.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.4, color="#334"),
            hoverinfo="none",
        )

        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for n in SG.nodes():
            x, y = pos[n]
            node_x.append(x); node_y.append(y)
            t    = node_types.get(n, "unknown")
            rs   = risk_lookup.get(n, 0.0)
            node_text.append(f"{n}<br>Type: {t}<br>Risk: {rs:.3f}")
            node_color.append(colour_map.get(t, "#888"))
            node_size.append(14 if t == "manufacturer" else 10 if t == "distributor" else 6)

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers",
            marker=dict(size=node_size, color=node_color, line=dict(width=0.5, color="#111")),
            text=node_text, hoverinfo="text",
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                margin=dict(l=10, r=10, t=40, b=10),
                title=dict(
                    text=f"Supply Chain Network  ({SG.number_of_nodes()} nodes shown)",
                    font=dict(color="#ccc", size=13),
                ),
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=560,
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        l1, l2, l3 = st.columns(3)
        with l1: st.markdown("<span style='color:#f4a261'>●</span> Manufacturer", unsafe_allow_html=True)
        with l2: st.markdown("<span style='color:#4fc3f7'>●</span> Distributor",  unsafe_allow_html=True)
        with l3: st.markdown("<span style='color:#a8d8a8'>●</span> Retailer",     unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    anom = load_anomalies()

    if anom.empty:
        st.warning("No anomaly data found. Run Stage 3 first.")
    else:
        high = anom[anom["anomaly_score"] >= 2].copy()
        susp = anom[anom["anomaly_score"] == 1].copy()

        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Total Flagged",         f"{len(anom):,}")
        with c2: metric_card("High-Confidence (2+)",  f"{len(high):,}")
        with c3: metric_card("Suspect (1 signal)",    f"{len(susp):,}")

        st.markdown("#### Anomaly Score Distribution")
        score_counts = anom["anomaly_score"].value_counts().sort_index()
        fig = px.bar(
            x=score_counts.index, y=score_counts.values,
            labels={"x": "Anomaly Score", "y": "Transactions"},
            color_discrete_sequence=["#4fc3f7"],
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a", height=260)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Top High-Confidence Anomalies")
        cols_show = [c for c in ["manufacturer", "distributor", "retailer", "quantity", "anomaly_score"]
                     if c in high.columns]
        st.dataframe(
            high[cols_show].sort_values("anomaly_score", ascending=False).head(50)
                           .reset_index(drop=True),
            use_container_width=True, height=320,
        )

        if "date" in anom.columns:
            st.markdown("#### Anomaly Timeline")
            anom["date"] = pd.to_datetime(anom["date"], errors="coerce")
            timeline = anom.dropna(subset=["date"])
            timeline = timeline.set_index("date").resample("ME")["anomaly_score"].count().reset_index()
            timeline.columns = ["date", "count"]
            fig2 = px.line(
                timeline, x="date", y="count",
                labels={"count": "Anomalies", "date": "Month"},
                template="plotly_dark", color_discrete_sequence=["#ff6b6b"],
            )
            fig2.update_layout(paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a", height=260)
            st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    risk = load_risk()
    gnn  = load_gnn()

    if risk.empty:
        st.warning("No risk scores found. Run Stage 4 first.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Entities Scored", f"{len(risk):,}")
        with c2:
            top = risk.iloc[0]
            metric_card("Highest Risk Entity", top["entity"][:22], f"score {top['risk_score']:.3f}")
        with c3:
            hr = risk[risk["risk_score"] > 0.7]
            metric_card("Entities Risk > 0.70", str(len(hr)))

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Top 20 — Centrality Risk Score")
            top20 = risk.head(20)
            fig = px.bar(
                top20, x="risk_score", y="entity", orientation="h",
                color="risk_score", color_continuous_scale="Reds",
                template="plotly_dark",
                labels={"risk_score": "Risk Score", "entity": ""},
            )
            fig.update_layout(
                paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False, height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            if not gnn.empty:
                st.markdown("#### Top 20 — GNN Enhanced Risk")
                top_gnn = gnn.sort_values("enhanced_risk", ascending=False).head(20)
                fig2 = px.bar(
                    top_gnn, x="enhanced_risk", y="entity", orientation="h",
                    color="enhanced_risk", color_continuous_scale="Blues",
                    template="plotly_dark",
                    labels={"enhanced_risk": "Enhanced Risk", "entity": ""},
                )
                fig2.update_layout(
                    paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False, height=480,
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Run Stage 7 (GNN Encoder) to see enhanced risk scores.")

        if not gnn.empty:
            st.markdown("#### GNN Embedding Space")
            fig_path = os.path.join(FIGS, "fig5_gnn_embeddings.png")
            if os.path.exists(fig_path):
                st.image(fig_path, use_column_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — MACRO STRESS & LSTM FORECAST
# ══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    stress   = load_stress()
    forecast = load_forecast()

    if stress.empty:
        st.warning("Run Stage 9 first to generate stress scores.")
    else:
        latest     = stress.iloc[-1]
        high_wks   = int((stress["stress_level"] == "HIGH").sum())
        med_wks    = int((stress["stress_level"] == "MEDIUM").sum())
        low_wks    = int((stress["stress_level"] == "LOW").sum())

        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Latest Score",    f"{latest['stress_score']:.3f}", str(latest["stress_level"]))
        with c2: metric_card("HIGH stress wks", str(high_wks))
        with c3: metric_card("MEDIUM stress wks", str(med_wks))
        with c4: metric_card("LOW stress wks",  str(low_wks))

        st.markdown("#### Macro Freight Stress Timeline + LSTM Forecast")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stress["date"], y=stress["stress_score"],
            mode="lines", name="Historical", line=dict(color="#4fc3f7", width=1.2),
        ))

        if not forecast.empty:
            fc_only = forecast[forecast["type"] == "forecast"]
            if not fc_only.empty:
                # Connect last historical point to first forecast point
                bridge_x = [stress["date"].iloc[-1], fc_only["date"].iloc[0]]
                bridge_y = [stress["stress_score"].iloc[-1], fc_only["stress_score"].iloc[0]]
                fig.add_trace(go.Scatter(
                    x=bridge_x, y=bridge_y,
                    mode="lines", line=dict(color="#ff6b6b", width=1.5, dash="dot"),
                    showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=fc_only["date"], y=fc_only["stress_score"],
                    mode="lines+markers", name="LSTM Forecast",
                    line=dict(color="#ff6b6b", width=2, dash="dot"),
                    marker=dict(size=8),
                ))

        fig.add_hline(y=0.65, line_dash="dash", line_color="red",   opacity=0.5, annotation_text="HIGH")
        fig.add_hline(y=0.40, line_dash="dash", line_color="orange", opacity=0.5, annotation_text="MEDIUM")

        fig.update_layout(
            paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
            xaxis=dict(color="#888"), yaxis=dict(color="#888", range=[0, 1]),
            legend=dict(bgcolor="#1a1a2e", font=dict(color="#ccc")),
            height=380, margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        if not forecast.empty:
            fc_only = forecast[forecast["type"] == "forecast"]
            if not fc_only.empty:
                st.markdown("#### 4-Week Forward Forecast")
                display = fc_only[["date", "stress_score", "stress_level"]].copy()
                display["date"] = display["date"].dt.strftime("%Y-%m-%d")
                display.columns = ["Week", "Stress Score", "Level"]
                st.dataframe(display.reset_index(drop=True), use_container_width=True, height=180)

# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — RECOVERY PREDICTOR
# ══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    regressor  = load_rf_regressor()
    classifier = load_rf_classifier()
    dis_data   = load_disruption_data()

    if regressor is None:
        st.warning("Run Stage 8 first to train the recovery models.")
    else:
        st.markdown("#### Predict Recovery Time for a Disruption Scenario")
        st.caption("Select disruption parameters below. The Random Forest models predict "
                   "full recovery days and the most likely response strategy.")

        if not dis_data.empty:
            dis_types   = sorted(dis_data["disruption_type"].dropna().unique().tolist())
            industries  = sorted(dis_data["industry"].dropna().unique().tolist())
            regions     = sorted(dis_data["supplier_region"].dropna().unique().tolist())
            sizes       = sorted(dis_data["supplier_size"].dropna().unique().tolist())
            resp_types  = sorted(dis_data["response_type"].dropna().unique().tolist())
        else:
            dis_types  = ["Cyber Attack", "Factory Incident", "Natural Disaster", "Logistics Delay"]
            industries = ["Pharmaceuticals", "Electronics", "Automotive", "Retail"]
            regions    = ["Asia-Pacific", "Europe", "North America", "South America"]
            sizes      = ["Small", "Medium", "Large"]
            resp_types = ["Alternative Supplier", "Combined Strategy", "Customer Delay"]

        col_a, col_b = st.columns(2)
        with col_a:
            sel_type    = st.selectbox("Disruption Type",    dis_types)
            sel_ind     = st.selectbox("Industry",           industries)
            sel_region  = st.selectbox("Supplier Region",    regions)
        with col_b:
            sel_size    = st.selectbox("Supplier Size",      sizes)
            sel_sev     = st.slider("Disruption Severity",   1, 5, 3)
            sel_impact  = st.slider("Production Impact (%)", 0, 100, 40)

        sel_backup = st.checkbox("Has Backup Supplier", value=False)

        # Encode inputs using value counts from the data
        def encode(series, val):
            cats = sorted(series.dropna().unique())
            return cats.index(val) if val in cats else 0

        if not dis_data.empty:
            type_enc   = encode(dis_data["disruption_type"],   sel_type)
            ind_enc    = encode(dis_data["industry"],          sel_ind)
            reg_enc    = encode(dis_data["supplier_region"],   sel_region)
            size_enc   = encode(dis_data["supplier_size"],     sel_size)
            resp_enc   = encode(dis_data["response_type"],     resp_types[0])
        else:
            type_enc = ind_enc = reg_enc = size_enc = resp_enc = 0

        feat_vec = np.array([[
            type_enc, ind_enc, reg_enc, size_enc, resp_enc,
            sel_sev, sel_impact, int(sel_backup),
        ]], dtype=float)

        if st.button("Predict Recovery", type="primary"):
            days     = float(regressor.predict(feat_vec)[0])
            strategy = classifier.predict(feat_vec)[0]

            if not dis_data.empty:
                resp_cats = sorted(dis_data["response_type"].dropna().unique())
                strategy_idx = int(strategy)   # cast numpy int → Python int
                strategy_label = (resp_cats[strategy_idx]
                                  if 0 <= strategy_idx < len(resp_cats)
                                  else str(strategy))
            else:
                strategy_label = str(strategy)

            r1, r2 = st.columns(2)
            with r1: metric_card("Predicted Full Recovery", f"{days:.0f} days")
            with r2: metric_card("Recommended Strategy",    strategy_label)

            severity_labels = {1: "Minimal", 2: "Low", 3: "Moderate", 4: "High", 5: "Critical"}
            st.info(
                f"Scenario: **{sel_type}** in **{sel_ind}** | "
                f"Region: {sel_region} | Size: {sel_size} | "
                f"Severity: {severity_labels.get(sel_sev, sel_sev)} | "
                f"Backup supplier: {'Yes' if sel_backup else 'No'}"
            )

        st.divider()
        st.markdown("#### Model Performance")
        metrics_path = os.path.join(OUT, "recovery_metrics.txt")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as f:
                    st.code(f.read(), language=None)
            except Exception:
                st.info("Model performance metrics available — restart the app to load.")

        fig_path = os.path.join(FIGS, "fig6_recovery.png")
        if os.path.exists(fig_path):
            st.image(fig_path, use_column_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 7 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════

# Human-readable names for the 8 features the RF model was trained on
# (matches FEATURE_COLS order in recovery_predictor.py)
RECOVERY_FEATURE_NAMES = [
    "Disruption Type",
    "Industry",
    "Supplier Region",
    "Supplier Size",
    "Response Type",
    "Disruption Severity",
    "Production Impact %",
    "Has Backup Supplier",
]

with tabs[6]:
    st.subheader("SHAP Explainability — Recovery Time Predictor")
    st.caption(
        "SHAP (SHapley Additive exPlanations) measures each feature's contribution to a prediction. "
        "Positive values push recovery time higher; negative values push it lower."
    )

    reg_path  = "models/recovery_regressor.pkl"
    data_path = "data/supplementary/disruption_processed.csv"

    if not os.path.exists(reg_path):
        st.info("Run Stage 8 first: python3 main.py --only 8")
    else:
        try:
            import joblib, shap, matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            reg_model = joblib.load(reg_path)

            FEATURE_COLS = [
                "disruption_type_enc", "industry_enc", "supplier_region_enc",
                "supplier_size_enc", "response_type_enc", "disruption_severity",
                "production_impact_pct", "has_backup_supplier",
            ]

            # ── SECTION 1: Global feature importance with real names ──────
            st.markdown("#### Global Feature Importance")
            st.caption("How much each feature contributes to recovery time predictions on average, ranked by importance.")

            importances = reg_model.feature_importances_
            fi_df = pd.DataFrame({
                "Feature":    RECOVERY_FEATURE_NAMES,
                "Importance": importances,
            }).sort_values("Importance", ascending=True)

            fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
            colors = ["#e74c3c" if v > fi_df["Importance"].median() else "#3498db"
                      for v in fi_df["Importance"]]
            ax_fi.barh(fi_df["Feature"], fi_df["Importance"], color=colors, edgecolor="none")
            ax_fi.set_xlabel("Mean Decrease in Impurity (Feature Importance)")
            ax_fi.set_title("Random Forest — Feature Importance for Recovery Days")
            ax_fi.axvline(fi_df["Importance"].median(), color="white", linestyle="--",
                          alpha=0.4, linewidth=1, label="Median")
            ax_fi.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_fi)
            plt.close()

            # ── SECTION 2: SHAP values on a sample ───────────────────────
            if os.path.exists(data_path):
                st.markdown("#### SHAP Values — Sample of 200 Disruptions")
                st.caption(
                    "Each bar shows the average absolute SHAP value for that feature — "
                    "how much it shifts the predicted recovery days away from the baseline."
                )

                dis_df = pd.read_csv(data_path)
                dis_df["has_backup_supplier"] = (
                    dis_df["has_backup_supplier"]
                    .map({True: 1, False: 0, "True": 1, "False": 0})
                    .fillna(0).astype(int)
                )
                dis_df = dis_df.dropna(subset=FEATURE_COLS)
                sample = dis_df[FEATURE_COLS].sample(
                    min(200, len(dis_df)), random_state=42
                ).values

                with st.spinner("Computing SHAP values..."):
                    explainer   = shap.TreeExplainer(reg_model)
                    shap_values = explainer.shap_values(sample)   # shape (200, 8)

                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                shap_df = pd.DataFrame({
                    "Feature":         RECOVERY_FEATURE_NAMES,
                    "Mean |SHAP|":     mean_abs_shap,
                }).sort_values("Mean |SHAP|", ascending=True)

                fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
                shap_colors = ["#e74c3c" if v > shap_df["Mean |SHAP|"].median() else "#f39c12"
                               for v in shap_df["Mean |SHAP|"]]
                ax_shap.barh(shap_df["Feature"], shap_df["Mean |SHAP|"],
                             color=shap_colors, edgecolor="none")
                ax_shap.set_xlabel("Mean |SHAP Value| — avg impact on recovery days (days)")
                ax_shap.set_title("SHAP Feature Impact — Recovery Time Predictor")
                plt.tight_layout()
                st.pyplot(fig_shap)
                plt.close()

                # ── SECTION 3: SHAP direction table ──────────────────────
                st.markdown("#### What Each Feature Means")
                mean_shap_signed = shap_values.mean(axis=0)
                direction_df = pd.DataFrame({
                    "Feature":           RECOVERY_FEATURE_NAMES,
                    "Avg SHAP (days)":   mean_shap_signed.round(2),
                    "Direction":         ["↑ Increases recovery time" if v > 0
                                         else "↓ Decreases recovery time"
                                         for v in mean_shap_signed],
                }).sort_values("Avg SHAP (days)", ascending=False)
                st.dataframe(direction_df, use_container_width=True, hide_index=True)

                st.caption(
                    "Avg SHAP > 0 means the feature tends to push predicted recovery time up. "
                    "Avg SHAP < 0 means it tends to push it down. "
                    "Values are in days."
                )
            else:
                st.info("disruption_processed.csv not found — showing feature importance only.")

        except Exception as e:
            st.error(f"SHAP computation error: {e}")
            st.info("Ensure shap is installed: pip install shap")

# ══════════════════════════════════════════════════════════════════════════
# TAB 8 — MULTI-DOMAIN RISK
# ══════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.subheader("Multi-Domain Risk Modelling — Objective 4")
    st.caption(
        "Same XGBoost architecture trained across 5 industries — "
        "proving the framework generalises beyond a single domain."
    )

    f1_path  = "output/multi_domain_f1.csv"
    fig_path = "output/figures/fig10_multi_domain_risk.png"

    if not os.path.exists(f1_path):
        st.warning("Run Stage 12 first: python3 main.py --from 12")
    else:
        df_f1 = pd.read_csv(f1_path)
        df_f1.columns = [str(c).strip().replace(" ", "_") for c in df_f1.columns]

        if "Industry" not in df_f1.columns or "F1_Score" not in df_f1.columns:
            st.warning("multi_domain_f1.csv is missing required columns: Industry, F1_Score")
            st.dataframe(df_f1, use_container_width=True)
            if os.path.exists(fig_path):
                st.image(fig_path, use_column_width=True)
        else:
            best_pool   = df_f1[df_f1["Industry"].astype(str).str.lower() != "all domains"]
            best_row    = best_pool.sort_values("F1_Score", ascending=False).iloc[0] if not best_pool.empty else None
            overall_row = df_f1[df_f1["Industry"] == "All Domains"]
            overall_f1  = overall_row["F1_Score"].values[0] if len(overall_row) else "N/A"

            col1, col2, col3 = st.columns(3)
            col1.metric("Best Domain",    best_row["Industry"] if best_row is not None else "N/A")
            col2.metric("Best Domain F1", f"{best_row['F1_Score']:.3f}" if best_row is not None else "N/A")
            col3.metric("Overall F1",     f"{float(overall_f1):.3f}"
                                          if overall_f1 != "N/A" else "N/A")

            st.dataframe(df_f1, use_container_width=True)

            if os.path.exists(fig_path):
                st.image(fig_path, use_column_width=True)
