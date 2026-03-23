"""
Auto-Onboarding Engine
======================
Drop any CSV file. This script figures out the column mapping on its own.

How it works:
  1. Auto-detects separator and encoding
  2. Profiles every column (data type, cardinality, variance, string length)
  3. Fuzzy-matches column names against a synonym dictionary for each role
  4. Falls back to data profiling when names are ambiguous
  5. Combines both signals into a confidence score
  6. Auto-assigns high-confidence columns silently
  7. Asks the user ONLY for genuinely ambiguous ones (usually 0-2 questions)
  8. Writes config.yaml that the rest of the pipeline reads

Usage (via main.py):
    python3 main.py --onboard /path/to/company_data.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
from difflib import SequenceMatcher

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_OUT = os.path.join(ROOT, "config.yaml")

# ── Synonym dictionary ───────────────────────────────────────
# Every known alias for each pipeline role

SYNONYMS = {
    "date": [
        "date", "order_date", "transaction_date", "ship_date", "shipping_date",
        "order date", "dateorders", "timestamp", "created_at", "purchase_date",
        "invoice_date", "delivery_date", "receipt_time", "orderdate", "shipdate",
        "order_timestamp", "txn_date", "sale_date", "dispatch_date", "ds",
    ],
    "manufacturer": [
        "manufacturer", "vendor", "supplier", "brand", "company", "mfr",
        "producer", "department", "dept", "revised_company_name", "department_name",
        "vendor_name", "supplier_name", "brand_name", "maker", "source_company",
        "seller", "drug_company", "mfg_name",
    ],
    "distributor": [
        "distributor", "wholesaler", "region", "warehouse", "hub", "reporter",
        "intermediary", "market", "order_region", "distribution_center",
        "reporter_name", "order region", "dist", "channel", "fulfillment_center",
        "logistics_center", "dispatch_center", "middleman",
    ],
    "retailer": [
        "retailer", "buyer", "customer", "pharmacy", "store", "outlet",
        "client", "buyer_name", "customer_name", "shop", "purchaser",
        "end_buyer", "receiver", "recipient", "pharmacy_name", "end_customer",
        "customer_city", "buyer_city",
    ],
    "retailer_state": [
        "state", "buyer_state", "customer_state", "province", "region_code",
        "customer state", "retailer_state", "buyer state", "ship_to_state",
        "delivery_state", "customer_province", "st",
    ],
    "quantity": [
        "quantity", "qty", "amount", "sales", "units", "volume", "total",
        "value", "order_item_quantity", "order item quantity", "transaction_quantity",
        "quant", "count", "units_sold", "sales_amount", "order_qty",
        "purchase_qty", "dosage_unit", "drug_quantity", "revenue",
    ],
}

ROLE_LABELS = {
    "date"           : "Transaction Date",
    "manufacturer"   : "Manufacturer / Supplier",
    "distributor"    : "Distributor / Wholesaler",
    "retailer"       : "Retailer / Buyer",
    "retailer_state" : "Retailer State / Region",
    "quantity"       : "Quantity / Volume / Sales",
}

HIGH_CONFIDENCE = 0.60   # auto-assign above this
LOW_CONFIDENCE  = 0.30   # ask user below this


# ── Step 1: File loading ─────────────────────────────────────

def detect_separator(path):
    """Try common separators; pick the one that gives the most columns."""
    best_sep, best_n = ",", 1
    for sep in [",", "|", "\t", ";"]:
        try:
            sample = pd.read_csv(path, sep=sep, nrows=5, encoding="utf-8", on_bad_lines="skip")
            if len(sample.columns) > best_n:
                best_sep, best_n = sep, len(sample.columns)
        except Exception:
            continue
    return best_sep

def load_file(path):
    """Load CSV with auto-detected separator and encoding."""
    sep = detect_separator(path)
    for enc in ["utf-8", "latin-1", "utf-16"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False, on_bad_lines="skip")
            print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns  (sep='{sep}', enc={enc})")
            return df, sep, enc
        except Exception:
            continue
    raise RuntimeError(f"Could not load file: {path}")


# ── Step 2: Column profiling ─────────────────────────────────

def profile_column(series):
    """Return a dict of stats about a column."""
    s = series.dropna()
    n = len(s)
    if n == 0:
        return {"empty": True}

    is_numeric  = pd.api.types.is_numeric_dtype(series)
    is_datetime = False
    if not is_numeric:
        try:
            parsed = pd.to_datetime(s.astype(str).head(50), errors="coerce")
            is_datetime = parsed.notna().mean() > 0.7
        except Exception:
            pass

    n_unique  = s.nunique()
    null_rate = series.isna().mean()

    profile = {
        "is_numeric"  : is_numeric,
        "is_datetime" : is_datetime,
        "n_unique"    : n_unique,
        "null_rate"   : null_rate,
        "total"       : n,
    }

    if is_numeric:
        profile["mean"]   = float(s.mean())
        profile["std"]    = float(s.std())
        profile["cv"]     = float(s.std() / s.mean()) if s.mean() != 0 else 0
        profile["min"]    = float(s.min())
        profile["max"]    = float(s.max())
    else:
        profile["avg_len"] = float(s.astype(str).str.len().mean())

    return profile


# ── Step 3: Name matching ────────────────────────────────────

def name_score(col_name, role):
    """Fuzzy match a column name against all synonyms for a role."""
    col_clean = col_name.lower().replace("_", " ").strip()
    best = 0.0
    for synonym in SYNONYMS[role]:
        ratio = SequenceMatcher(None, col_clean, synonym).ratio()
        if col_clean == synonym:
            return 1.0
        if col_clean in synonym or synonym in col_clean:
            ratio = max(ratio, 0.85)
        best = max(best, ratio)
    return best


# ── Step 4: Profile matching ─────────────────────────────────

def profile_score(profile, role):
    """Score how well a column's data characteristics match a role."""
    if profile.get("empty"):
        return 0.0

    n_unique  = profile["n_unique"]
    is_num    = profile["is_numeric"]
    is_dt     = profile["is_datetime"]
    null_rate = profile["null_rate"]

    if null_rate > 0.5:
        return 0.0

    if role == "date":
        return 0.95 if is_dt else (0.3 if not is_num else 0.0)

    if role == "quantity":
        if not is_num:
            return 0.0
        cv = profile.get("cv", 0)
        mn = profile.get("mean", 0)
        score = 0.5
        if cv > 0.3:    score += 0.2   # high variance is good
        if mn > 1:      score += 0.2   # not binary
        return min(score, 0.9)

    if role == "retailer_state":
        if is_num or is_dt:
            return 0.0
        avg_len = profile.get("avg_len", 99)
        if 1 < avg_len < 25 and 5 < n_unique < 80:
            return 0.75
        return 0.2

    if role == "manufacturer":
        if is_num or is_dt:
            return 0.0
        if 2 <= n_unique <= 100:
            return 0.65
        return 0.2

    if role == "distributor":
        if is_num or is_dt:
            return 0.0
        if 2 <= n_unique <= 250:
            return 0.55
        return 0.2

    if role == "retailer":
        if is_num or is_dt:
            return 0.0
        if n_unique >= 20:
            return 0.60
        return 0.2

    return 0.0


# ── Step 5: Combined scoring & assignment ────────────────────

def score_all(df):
    """Return a roles × columns score matrix."""
    roles   = list(SYNONYMS.keys())
    columns = list(df.columns)
    profiles = {col: profile_column(df[col]) for col in columns}

    scores = {}
    for role in roles:
        scores[role] = {}
        for col in columns:
            ns = name_score(col, role)
            ps = profile_score(profiles[col], role)
            # Name match dominates; profile is a tiebreaker
            combined = 0.70 * ns + 0.30 * ps
            scores[role][col] = round(combined, 3)

    return scores, profiles


def greedy_assign(scores):
    """
    Assign columns to roles greedily (highest score first),
    no column used twice.
    Returns: { role: (column, confidence) }
    """
    roles   = list(scores.keys())
    all_pairs = []
    for role in roles:
        for col, sc in scores[role].items():
            all_pairs.append((sc, role, col))
    all_pairs.sort(reverse=True)

    assigned_roles = {}
    used_cols      = set()

    for sc, role, col in all_pairs:
        if role not in assigned_roles and col not in used_cols:
            assigned_roles[role] = (col, sc)
            used_cols.add(col)

    # Fill any unassigned roles with None
    for role in roles:
        if role not in assigned_roles:
            assigned_roles[role] = (None, 0.0)

    return assigned_roles


# ── Step 6: User confirmation for uncertain columns ──────────

def confirm_mapping(assignment, df):
    """
    For uncertain mappings, ask the user.
    Returns updated assignment dict.
    """
    columns = list(df.columns)
    print()
    needs_input = {r: v for r, v in assignment.items() if v[1] < HIGH_CONFIDENCE}

    if not needs_input:
        print("  All columns mapped with high confidence. No input needed.")
        return assignment

    print(f"  {len(needs_input)} column(s) need confirmation:\n")
    for role, (col, conf) in needs_input.items():
        label = ROLE_LABELS[role]
        print(f"  Role: {label}")
        if col:
            print(f"  Best guess: '{col}'  (confidence: {conf:.0%})")
        else:
            print(f"  No match found.")
        print(f"  Available columns:")
        for i, c in enumerate(columns):
            print(f"    [{i}] {c}")
        print(f"    [s] Skip this role (not in dataset)")

        while True:
            val = input(f"\n  Enter column number for '{label}' (or 's' to skip): ").strip()
            if val.lower() == "s":
                assignment[role] = (None, 1.0)
                break
            try:
                idx = int(val)
                if 0 <= idx < len(columns):
                    assignment[role] = (columns[idx], 1.0)
                    break
                else:
                    print("  Invalid number, try again.")
            except ValueError:
                print("  Enter a number or 's'.")
        print()

    return assignment


# ── Step 7: Retailer combine detection ───────────────────────

def detect_retailer_combine(assignment, df, scores):
    """
    If the retailer column has low uniqueness (e.g. city name only),
    suggest combining it with another column (e.g. state).
    Returns combine_col or None.
    """
    retailer_col, _ = assignment.get("retailer", (None, 0))
    state_col, _    = assignment.get("retailer_state", (None, 0))

    if not retailer_col or not state_col:
        return None

    n_unique = df[retailer_col].nunique()
    if n_unique < 200:
        # Retailer names alone aren't unique enough — combine with state
        combined_unique = (df[retailer_col].astype(str) + ", " + df[state_col].astype(str)).nunique()
        if combined_unique > n_unique:
            return state_col
    return None


# ── Step 8: Write config.yaml ────────────────────────────────

def write_config(path, sep, encoding, assignment, retailer_combine=None):
    col_map = {role: col for role, (col, _) in assignment.items() if col}

    config = {
        "dataset": {
            "path"      : path,
            "separator" : sep,
            "encoding"  : encoding,
        },
        "columns": col_map,
        "settings": {
            "retailer_combine": retailer_combine,   # second col appended to retailer
        },
        "thresholds": {
            "volume_zscore"        : 3.0,
            "frequency_zscore"     : 2.5,
            "temporal_surge_zscore": 2.5,
            "concentration_pct"    : 0.90,
        },
    }

    with open(CONFIG_OUT, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  Config written → {CONFIG_OUT}")


# ── Main ─────────────────────────────────────────────────────

def run(filepath):
    print("=" * 55)
    print("  AUTO-ONBOARDING ENGINE")
    print("=" * 55)

    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    print(f"\nAnalysing: {os.path.basename(filepath)}")
    df, sep, enc = load_file(filepath)

    print("\nProfiling and scoring columns...")
    scores, profiles = score_all(df)
    assignment       = greedy_assign(scores)

    # Show auto-assigned results
    print("\n  Detected column mapping:")
    print(f"  {'Role':<20} {'Column':<35} {'Confidence'}")
    print("  " + "─" * 65)
    for role, (col, conf) in assignment.items():
        label  = ROLE_LABELS[role]
        status = "✓ auto" if conf >= HIGH_CONFIDENCE else ("? check" if conf >= LOW_CONFIDENCE else "✗ low")
        print(f"  {label:<20} {str(col):<35} {conf:.0%}  {status}")

    # Confirm uncertain ones
    assignment = confirm_mapping(assignment, df)

    # Check if retailer should be combined with state
    retailer_combine = detect_retailer_combine(assignment, df, scores)
    if retailer_combine:
        retailer_col = assignment["retailer"][0]
        print(f"\n  Note: '{retailer_col}' has low uniqueness.")
        print(f"  Combining with '{retailer_combine}' for unique retailer IDs.")

    write_config(filepath, sep, enc, assignment, retailer_combine)

    print("\n  Onboarding complete.")
    print("  Running pipeline from Stage 1...\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 src/auto_onboard.py /path/to/data.csv")
        sys.exit(1)
    run(sys.argv[1])
