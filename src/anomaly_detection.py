"""
Stage 3 — Multi-Dimensional Anomaly Detection (Enhanced)
Maps to: Innate Immunity — fast, broad detection of suspicious signals

Five detection dimensions:
  1. Volume Anomaly     — Z-score on transaction quantity
  2. Frequency Anomaly — unusually high number of transactions on a pair
  3. Temporal Surge    — sudden spike in a manufacturer's monthly volume
  4. Concentration Risk — single distributor supplying >90% of a retailer
  5. Isolation Forest  — ML-based joint-distribution outlier detection (5th signal)

A transaction flagged on 2+ dimensions is classified as a high-confidence anomaly.
Precision/Recall/F1 reported using 3+ Z-score agreement as pseudo-ground-truth.
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import shutil
import tempfile

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT  = os.path.join(ROOT, "data", "processed", "clean_chain.csv")
OUTPUT = os.path.join(ROOT, "output", "anomalies.csv")
CONFIG = os.path.join(ROOT, "config.yaml")

# Load thresholds from config.yaml if present, else use defaults
_thresholds = {}
if HAS_YAML and os.path.exists(CONFIG):
    with open(CONFIG) as f:
        _cfg = yaml.safe_load(f)
    _thresholds = _cfg.get("thresholds", {})

VOL_Z    = _thresholds.get("volume_zscore",         3.0)
FREQ_Z   = _thresholds.get("frequency_zscore",      2.5)
SURGE_Z  = _thresholds.get("temporal_surge_zscore", 2.5)
CONC_PCT = _thresholds.get("concentration_pct",     0.90)

# ── Macro stress adjustment ───────────────────────────────────
# If Stage 9 has run, load stress scores and tighten thresholds
# on high-stress periods (catch more anomalies when environment is bad)
MACRO_STRESS = os.path.join(ROOT, "output", "macro_stress_scores.csv")

def get_macro_multiplier(date_series):
    """Return a threshold multiplier based on avg macro stress for the dataset period."""
    if not os.path.exists(MACRO_STRESS):
        return 1.0
    try:
        stress_df = pd.read_csv(MACRO_STRESS, parse_dates=["date"])
        d_min, d_max = date_series.min(), date_series.max()
        mask  = (stress_df["date"] >= d_min) & (stress_df["date"] <= d_max)
        sub   = stress_df[mask]
        if len(sub) == 0:
            # Dataset dates outside stress range — use overall mean
            avg_stress = stress_df["stress_score"].mean()
        else:
            avg_stress = sub["stress_score"].mean()
        # HIGH stress (≥0.65) → multiplier 0.75 (lower thresholds = more sensitive)
        # LOW  stress (<0.40) → multiplier 1.20 (higher thresholds = fewer false positives)
        if avg_stress >= 0.65:
            return 0.75
        elif avg_stress >= 0.40:
            return 1.00
        else:
            return 1.20
    except Exception:
        return 1.0

print("=" * 55)
print("  STAGE 3 — MULTI-DIMENSIONAL ANOMALY DETECTION")
print("=" * 55)
print(f"  Base thresholds: volume Z>{VOL_Z}  freq Z>{FREQ_Z}  surge Z>{SURGE_Z}  conc>{CONC_PCT:.0%}")

if not os.path.exists(INPUT):
    print(f"[ERROR] {INPUT} not found. Run preprocess.py first.")
    exit(1)

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"])
print(f"Loaded {len(df):,} transactions.")

# Apply macro stress multiplier to thresholds
_mult  = get_macro_multiplier(df["date"])
VOL_Z    = round(VOL_Z   * _mult, 2)
FREQ_Z   = round(FREQ_Z  * _mult, 2)
SURGE_Z  = round(SURGE_Z * _mult, 2)
_stress_label = "HIGH — thresholds tightened" if _mult < 1 else ("LOW — thresholds relaxed" if _mult > 1 else "MEDIUM — thresholds unchanged")
print(f"  Macro stress : {_stress_label}  (multiplier={_mult})")
print(f"  Final thresholds: volume Z>{VOL_Z}  freq Z>{FREQ_Z}  surge Z>{SURGE_Z}\n")

# ─────────────────────────────────────────────
# DIMENSION 1: Volume Anomaly (Z-score)
# Flags transactions where quantity is an extreme outlier
# ─────────────────────────────────────────────
df["z_quantity"] = zscore(df["quantity"])
df["flag_volume"] = df["z_quantity"].abs() > VOL_Z

n = df["flag_volume"].sum()
print(f"[1] Volume Anomaly       : {n} flagged  (|Z-score| > {VOL_Z})")

# ─────────────────────────────────────────────
# DIMENSION 2: Frequency Anomaly
# Flags manufacturer-retailer pairs that transact abnormally often
# ─────────────────────────────────────────────
pair_freq = (
    df.groupby(["manufacturer", "retailer"])
      .size()
      .reset_index(name="pair_freq")
)
df = df.merge(pair_freq, on=["manufacturer", "retailer"], how="left")

if df["pair_freq"].std() > 0:
    df["z_freq"] = zscore(df["pair_freq"])
else:
    df["z_freq"] = 0.0

df["flag_frequency"] = df["z_freq"] > FREQ_Z

n = df["flag_frequency"].sum()
print(f"[2] Frequency Anomaly    : {n} flagged  (pair transaction count Z > {FREQ_Z})")

# ─────────────────────────────────────────────
# DIMENSION 3: Temporal Surge
# Flags when a manufacturer ships unusually large monthly totals
# ─────────────────────────────────────────────
df["year_month"] = df["date"].dt.to_period("M")

monthly = (
    df.groupby(["manufacturer", "year_month"])["quantity"]
      .sum()
      .reset_index(name="monthly_total")
)

def safe_zscore(x):
    if len(x) < 2 or x.std() == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return pd.Series(zscore(x), index=x.index)

monthly["z_surge"] = monthly.groupby("manufacturer")["monthly_total"].transform(safe_zscore)

df = df.merge(
    monthly[["manufacturer", "year_month", "z_surge"]],
    on=["manufacturer", "year_month"],
    how="left"
)
df["z_surge"] = df["z_surge"].fillna(0)
df["flag_surge"] = df["z_surge"].abs() > SURGE_Z

n = df["flag_surge"].sum()
print(f"[3] Temporal Surge       : {n} flagged  (monthly volume Z > {SURGE_Z})")

# ─────────────────────────────────────────────
# DIMENSION 4: Concentration Risk
# Flags retailer-distributor pairs with >90% supply dependency
# ─────────────────────────────────────────────
retailer_total = (
    df.groupby("retailer")["quantity"]
      .sum()
      .reset_index(name="retailer_total")
)

dist_retailer = (
    df.groupby(["distributor", "retailer"])["quantity"]
      .sum()
      .reset_index(name="dr_total")
)

dist_retailer = dist_retailer.merge(retailer_total, on="retailer")
dist_retailer["concentration"] = dist_retailer["dr_total"] / dist_retailer["retailer_total"]

df = df.merge(
    dist_retailer[["distributor", "retailer", "concentration"]],
    on=["distributor", "retailer"],
    how="left"
)
df["concentration"] = df["concentration"].fillna(0)
df["flag_concentration"] = df["concentration"] > CONC_PCT

n = df["flag_concentration"].sum()
print(f"[4] Concentration Risk   : {n} flagged  (single supplier >{CONC_PCT:.0%} of retailer supply)")

# ─────────────────────────────────────────────
# DIMENSION 5: Isolation Forest
# ML-based approach — learns joint distribution of all 4 signal
# features and flags statistical outliers in that combined space.
# Catches anomalies that are subtle in each dimension individually
# but form an outlier pattern when viewed together.
# ─────────────────────────────────────────────
IF_FEATURES = ["z_quantity", "z_freq", "z_surge", "concentration"]
X_if = df[IF_FEATURES].fillna(0).values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_if)

# contamination: ~5% expected anomaly rate based on synthetic dataset design
iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
preds   = iforest.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal
scores  = iforest.score_samples(X_scaled) # lower raw score = more anomalous

df["flag_iforest"]   = (preds == -1)
df["iforest_score"]  = -scores            # negate so higher = more anomalous

n = df["flag_iforest"].sum()
print(f"[5] Isolation Forest     : {n} flagged  (contamination=0.05, n_estimators=100)")

# ─────────────────────────────────────────────
# COMPOSITE SCORING (now 5 signals)
# ─────────────────────────────────────────────

# Save Z-score only count before adding IF (used as pseudo ground truth below)
zscore_flag_cols  = ["flag_volume", "flag_frequency", "flag_surge", "flag_concentration"]
df["zscore_count"] = df[zscore_flag_cols].sum(axis=1)

all_flag_cols     = zscore_flag_cols + ["flag_iforest"]
df["anomaly_score"] = df[all_flag_cols].sum(axis=1)
df["is_anomaly"]    = df["anomaly_score"] >= 2   # 2+ dimensions = high-confidence
df["is_suspect"]    = (df["anomaly_score"] == 1)

print(f"\n──────────────────────────────────────────────────")
print(f"  RESULTS")
print(f"──────────────────────────────────────────────────")
print(f"  High-confidence anomalies (2+ dimensions) : {df['is_anomaly'].sum()}")
print(f"  Suspect transactions    (1 dimension)     : {df['is_suspect'].sum()}")
print(f"  Clean transactions                        : {(df['anomaly_score'] == 0).sum():,}")

# ─────────────────────────────────────────────
# PRECISION / RECALL / F1 EVALUATION
# Pseudo-ground-truth: transactions where 3+ Z-score signals fire
# (high agreement across independent statistical tests = high-confidence true anomaly)
# This lets us evaluate the 2+ threshold and Isolation Forest as independent detectors.
# ─────────────────────────────────────────────
print(f"\n──────────────────────────────────────────────────")
print(f"  PRECISION / RECALL / F1  (pseudo-GT: zscore_count >= 3)")
print(f"──────────────────────────────────────────────────")

y_true = (df["zscore_count"] >= 3).astype(int)
n_gt   = y_true.sum()
print(f"  Ground-truth positives  : {n_gt} transactions ({n_gt/len(df)*100:.2f}%)")

def _prf(y_true, y_pred, label):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    print(f"  {label:<36}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")
    return p, r, f

p1, r1, f1_zs = _prf(y_true, (df["zscore_count"] >= 2).astype(int),  "Z-score only  (2+ signals)")
p2, r2, f1_if = _prf(y_true, df["flag_iforest"].astype(int),          "Isolation Forest alone")
p3, r3, f1_en = _prf(y_true, df["is_anomaly"].astype(int),            "Ensemble (2+ incl. IF)  ← current")

# Save metrics report
METRICS_OUT = os.path.join(ROOT, "output", "anomaly_metrics.txt")
os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
_am = "\n".join([
    "STAGE 3 — ANOMALY DETECTION METRICS",
    "=" * 50,
    "",
    f"Dataset size         : {len(df):,} transactions",
    f"Macro stress mult    : {_mult}  ({_stress_label})",
    f"Final Z thresholds   : vol={VOL_Z}  freq={FREQ_Z}  surge={SURGE_Z}",
    "",
    "Signal counts:",
    f"  [1] Volume Anomaly       : {df['flag_volume'].sum()}",
    f"  [2] Frequency Anomaly    : {df['flag_frequency'].sum()}",
    f"  [3] Temporal Surge       : {df['flag_surge'].sum()}",
    f"  [4] Concentration Risk   : {df['flag_concentration'].sum()}",
    f"  [5] Isolation Forest     : {df['flag_iforest'].sum()}",
    "",
    f"High-confidence anomalies (2+ signals) : {df['is_anomaly'].sum()}",
    f"Suspect transactions    (1 signal)     : {df['is_suspect'].sum()}",
    "",
    "Evaluation  (pseudo-GT: zscore_count >= 3)",
    f"  Ground-truth positives  : {n_gt}",
    "",
    f"  {'Method':<36}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}",
    f"  {'-'*36}  {'-'*9}  {'-'*6}  {'-'*6}",
    f"  {'Z-score only  (2+ signals)':<36}  {p1:>9.3f}  {r1:>6.3f}  {f1_zs:>6.3f}",
    f"  {'Isolation Forest alone':<36}  {p2:>9.3f}  {r2:>6.3f}  {f1_if:>6.3f}",
    f"  {'Ensemble (2+ incl. IF)':<36}  {p3:>9.3f}  {r3:>6.3f}  {f1_en:>6.3f}",
])
try:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        tmp.write(_am)
        _tmp = tmp.name
    shutil.copy2(_tmp, METRICS_OUT)
    os.unlink(_tmp)
    print(f"\nMetrics saved → {METRICS_OUT}")
except Exception as _e:
    print(f"\n[WARN] Could not save anomaly metrics: {_e}")

# ─────────────────────────────────────────────
# SAVE ANOMALIES
# ─────────────────────────────────────────────
anomalies = df[df["is_anomaly"]].copy()
anomalies = anomalies.sort_values("anomaly_score", ascending=False)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
anomalies[[
    "date", "manufacturer", "distributor", "retailer",
    "retailer_state", "quantity", "z_quantity",
    "flag_volume", "flag_frequency", "flag_surge", "flag_concentration",
    "flag_iforest", "iforest_score", "anomaly_score"
]].to_csv(OUTPUT, index=False)

print(f"\nTop anomalies:")
print(anomalies[["manufacturer", "retailer", "quantity", "anomaly_score"]].head(10).to_string(index=False))
print(f"\nSaved → {OUTPUT}")
