"""
Stage 2 — Supply Chain Graph Construction
Reads clean_chain.csv, builds a directed weighted graph
(manufacturer → distributor → retailer) using NetworkX.

Node attributes computed and stored:
  in_volume, out_volume, in_degree, out_degree, total_transactions
  betweenness_centrality, pagerank, composite_risk_score

Centrality visualisation saved to output/figures/fig_centrality.png
Composite risk scores saved to output/graph_risk_scores.csv
Graph saved to models/supplychain_graph.pkl
"""

import pandas as pd
import numpy as np
import networkx as nx
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT   = os.path.join(ROOT, "data", "processed", "clean_chain.csv")
OUTPUT  = os.path.join(ROOT, "models", "supplychain_graph.pkl")
FIG_OUT = os.path.join(ROOT, "output", "figures", "fig_centrality.png")
RISK_OUT= os.path.join(ROOT, "output", "graph_risk_scores.csv")

print("=" * 55)
print("  STAGE 2 — SUPPLY CHAIN GRAPH CONSTRUCTION")
print("=" * 55)

if not os.path.exists(INPUT):
    print(f"[ERROR] {INPUT} not found. Run preprocess.py first.")
    exit(1)

df = pd.read_csv(INPUT)
print(f"Loaded {len(df):,} transactions.")

# ─────────────────────────────────────────────
# BUILD DIRECTED WEIGHTED GRAPH
# ─────────────────────────────────────────────
G = nx.DiGraph()

# Vectorised edge accumulation using groupby (faster than row iteration)
mfr_dist = (
    df.groupby(["manufacturer", "distributor"])
    .agg(weight=("quantity", "sum"), transactions=("quantity", "count"))
    .reset_index()
)
dist_ret = (
    df.groupby(["distributor", "retailer"])
    .agg(weight=("quantity", "sum"), transactions=("quantity", "count"))
    .reset_index()
)

for _, r in mfr_dist.iterrows():
    G.add_edge(r["manufacturer"], r["distributor"],
               weight=float(r["weight"]), transactions=int(r["transactions"]))

for _, r in dist_ret.iterrows():
    G.add_edge(r["distributor"], r["retailer"],
               weight=float(r["weight"]), transactions=int(r["transactions"]))

# ─────────────────────────────────────────────
# NODE TYPE TAGGING
# ─────────────────────────────────────────────
manufacturers = set(df["manufacturer"].unique())
distributors  = set(df["distributor"].unique())
retailers     = set(df["retailer"].unique())

for node in G.nodes():
    if node in manufacturers:
        G.nodes[node]["type"] = "manufacturer"
    elif node in distributors:
        G.nodes[node]["type"] = "distributor"
    else:
        G.nodes[node]["type"] = "retailer"

# ─────────────────────────────────────────────
# NODE ATTRIBUTES: volume, degree, transaction count
# ─────────────────────────────────────────────
for node in G.nodes():
    in_vol   = sum(d.get("weight", 0) for _, _, d in G.in_edges(node,  data=True))
    out_vol  = sum(d.get("weight", 0) for _, _, d in G.out_edges(node, data=True))
    in_tx    = sum(d.get("transactions", 0) for _, _, d in G.in_edges(node,  data=True))
    out_tx   = sum(d.get("transactions", 0) for _, _, d in G.out_edges(node, data=True))
    G.nodes[node]["in_volume"]          = float(in_vol)
    G.nodes[node]["out_volume"]         = float(out_vol)
    G.nodes[node]["in_degree"]          = G.in_degree(node)
    G.nodes[node]["out_degree"]         = G.out_degree(node)
    G.nodes[node]["total_transactions"] = int(in_tx + out_tx)

print(f"\nNode attributes computed (in_volume, out_volume, degree, transactions).")

# ─────────────────────────────────────────────
# CENTRALITY METRICS
# betweenness: how critical is this node for all shortest paths
# pagerank:    how many important nodes point to it
# ─────────────────────────────────────────────
print("Computing betweenness centrality (this may take a moment)...")
betweenness = nx.betweenness_centrality(G, normalized=True, weight=None)
pagerank    = nx.pagerank(G, alpha=0.85, weight="weight", max_iter=200)

for node in G.nodes():
    G.nodes[node]["betweenness_centrality"] = betweenness.get(node, 0.0)
    G.nodes[node]["pagerank"]               = pagerank.get(node, 0.0)

# ─────────────────────────────────────────────
# COMPOSITE RISK SCORE
# Combines betweenness (structural criticality),
# in_degree (how many depend on this node), and
# pagerank (global importance) into a single 0–1 score.
# Weights: betweenness=0.5, in_degree_norm=0.3, pagerank=0.2
# ─────────────────────────────────────────────
bc_vals  = np.array([betweenness[n] for n in G.nodes()])
pr_vals  = np.array([pagerank[n]    for n in G.nodes()])
ind_vals = np.array([G.in_degree(n) for n in G.nodes()], dtype=float)

def _norm(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)

bc_norm  = _norm(bc_vals)
pr_norm  = _norm(pr_vals)
ind_norm = _norm(ind_vals)

nodes_list = list(G.nodes())
composite  = 0.5 * bc_norm + 0.3 * ind_norm + 0.2 * pr_norm

for i, node in enumerate(nodes_list):
    G.nodes[node]["composite_risk"] = float(composite[i])

print("Composite risk score computed  (betweenness×0.5 + in_degree×0.3 + pagerank×0.2).")

# ─────────────────────────────────────────────
# CENTRALITY VISUALISATION
# Top-20 nodes by betweenness centrality,
# coloured by node type
# ─────────────────────────────────────────────
os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)

type_color = {"manufacturer": "#e74c3c", "distributor": "#f39c12", "retailer": "#3498db"}

sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]
labels = [n[:28] + "…" if len(n) > 28 else n for n, _ in sorted_nodes]
scores = [v for _, v in sorted_nodes]
colors = [type_color.get(G.nodes[n].get("type", "retailer"), "#95a5a6") for n, _ in sorted_nodes]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Supply Chain Graph — Centrality Analysis", fontsize=14, fontweight="bold")

# Left: betweenness centrality bar chart
axes[0].barh(labels[::-1], scores[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
axes[0].set_xlabel("Betweenness Centrality (normalised)")
axes[0].set_title("Top 20 Nodes — Betweenness Centrality")
axes[0].tick_params(axis="y", labelsize=8)
from matplotlib.patches import Patch
legend_handles = [Patch(color=c, label=t.capitalize()) for t, c in type_color.items()]
axes[0].legend(handles=legend_handles, loc="lower right", fontsize=8)

# Right: composite risk score for top-20 by composite
sorted_comp = sorted(
    [(n, G.nodes[n]["composite_risk"]) for n in G.nodes()],
    key=lambda x: x[1], reverse=True
)[:20]
comp_labels = [n[:28] + "…" if len(n) > 28 else n for n, _ in sorted_comp]
comp_scores = [v for _, v in sorted_comp]
comp_colors = [type_color.get(G.nodes[n].get("type", "retailer"), "#95a5a6") for n, _ in sorted_comp]

axes[1].barh(comp_labels[::-1], comp_scores[::-1], color=comp_colors[::-1], edgecolor="white", linewidth=0.5)
axes[1].set_xlabel("Composite Risk Score (0–1)")
axes[1].set_title("Top 20 Nodes — Composite Risk Score")
axes[1].tick_params(axis="y", labelsize=8)
axes[1].legend(handles=legend_handles, loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig(FIG_OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Centrality visualisation saved → {FIG_OUT}")

# ─────────────────────────────────────────────
# SAVE RISK SCORES CSV
# ─────────────────────────────────────────────
risk_rows = []
for node in G.nodes():
    nd = G.nodes[node]
    risk_rows.append({
        "entity":                node,
        "type":                  nd.get("type", "unknown"),
        "in_volume":             nd.get("in_volume", 0),
        "out_volume":            nd.get("out_volume", 0),
        "in_degree":             nd.get("in_degree", 0),
        "out_degree":            nd.get("out_degree", 0),
        "total_transactions":    nd.get("total_transactions", 0),
        "betweenness_centrality":nd.get("betweenness_centrality", 0),
        "pagerank":              nd.get("pagerank", 0),
        "composite_risk":        nd.get("composite_risk", 0),
    })

risk_df = pd.DataFrame(risk_rows).sort_values("composite_risk", ascending=False)
os.makedirs(os.path.dirname(RISK_OUT), exist_ok=True)
risk_df.to_csv(RISK_OUT, index=False)
print(f"Graph risk scores saved → {RISK_OUT}")

# ─────────────────────────────────────────────
# SAVE GRAPH
# ─────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "wb") as f:
    pickle.dump(G, f)

print(f"\nGraph built successfully.")
print(f"  Nodes         : {G.number_of_nodes():,}")
print(f"  Edges         : {G.number_of_edges():,}")
print(f"  Manufacturers : {len(manufacturers)}")
print(f"  Distributors  : {len(distributors)}")
print(f"  Retailers     : {len(retailers)}")
print(f"  Top node by composite risk: {risk_df.iloc[0]['entity']}  ({risk_df.iloc[0]['composite_risk']:.4f})")
print(f"Saved → {OUTPUT}")
