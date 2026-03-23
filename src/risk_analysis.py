"""
Stage 4 — Risk and Vulnerability Analysis
Maps to: Innate Immunity — identifying high-danger nodes in the network

Uses three graph centrality metrics:
  - Betweenness Centrality : nodes that act as critical bridges
  - In-Degree Centrality   : nodes with many incoming dependencies
  - PageRank               : nodes with influential upstream suppliers

Composite risk score = weighted average of the three.
Saves top risky nodes to output/risk_scores.csv
"""

import networkx as nx
import pickle
import pandas as pd
import os

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL  = os.path.join(ROOT, "models", "supplychain_graph.pkl")
OUTPUT = os.path.join(ROOT, "output", "risk_scores.csv")

print("=" * 55)
print("  STAGE 4 — RISK & VULNERABILITY ANALYSIS")
print("=" * 55)

if not os.path.exists(MODEL):
    print(f"[ERROR] {MODEL} not found. Run build_chain.py first.")
    exit(1)

with open(MODEL, "rb") as f:
    G = pickle.load(f)

print(f"Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
print("Computing centrality metrics...")

# Betweenness centrality — critical bridge nodes
betweenness = nx.betweenness_centrality(G, normalized=True)

# In-degree centrality — highly depended-upon nodes
in_degree = nx.in_degree_centrality(G)

# PageRank — influential nodes (weighted by volume if possible)
try:
    pagerank = nx.pagerank(G, weight="weight", max_iter=500)
except:
    pagerank = nx.pagerank(G, max_iter=500)

# Build risk dataframe
nodes = list(G.nodes())
risk_df = pd.DataFrame({
    "entity"       : nodes,
    "betweenness"  : [betweenness.get(n, 0) for n in nodes],
    "in_degree"    : [in_degree.get(n, 0) for n in nodes],
    "pagerank"     : [pagerank.get(n, 0) for n in nodes],
    "node_type"    : [G.nodes[n].get("type", "unknown") for n in nodes],
})

# Normalise each metric to 0-1 then compute composite
for col in ["betweenness", "in_degree", "pagerank"]:
    max_val = risk_df[col].max()
    risk_df[f"{col}_norm"] = risk_df[col] / max_val if max_val > 0 else 0

risk_df["risk_score"] = (
    0.5 * risk_df["betweenness_norm"] +
    0.3 * risk_df["in_degree_norm"]   +
    0.2 * risk_df["pagerank_norm"]
)

risk_df = risk_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
risk_df["risk_rank"] = risk_df.index + 1

top_entities = list(risk_df[["entity", "node_type", "risk_score"]].itertuples(index=False, name=None))
top_entities = [
    (node, ntype, score) for node, ntype, score in top_entities
    if not str(node).replace("/","").replace("-","").replace(" ","").isdigit()
    and node not in ["Date Not Captured", "Pre-PQ Process", "N/A"]
]
top_entities = top_entities[:10]

# Save full results
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
risk_df[["risk_rank", "entity", "node_type", "risk_score",
         "betweenness", "in_degree", "pagerank"]].to_csv(OUTPUT, index=False)

# Print top 10
print(f"\nTop 10 High-Risk Entities (Composite Score):")
print(f"{'Rank':<5} {'Entity':<45} {'Type':<15} {'Score':<8}")
print("─" * 75)
for rank, (node, ntype, score) in enumerate(top_entities, start=1):
    print(f"{rank:<5} {str(node)[:44]:<45} {ntype:<15} {score:.4f}")

print(f"\nFull risk scores saved → {OUTPUT}")
