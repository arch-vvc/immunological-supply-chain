"""
Stage 0 — Dataset Sampling (Run this ONCE before anything else)
Takes the full raw ARCOS dataset (datasetuc.csv, pipe-separated)
and samples 50,000 rows, saves to data/raw/arcos_sampled_50k.csv

Only needed if you have the full ARCOS dataset.
If you already have arcos_sampled_50k.csv, skip this and go to Stage 1.
"""

import pandas as pd
import random
import os

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT  = os.path.join(ROOT, "data", "raw", "datasetuc.csv")         # full ARCOS dataset (pipe-separated)
OUTPUT = os.path.join(ROOT, "data", "raw", "arcos_sampled_50k.csv") # what preprocess.py expects

TARGET_SIZE = 50_000
CHUNK_SIZE  = 200_000

print("=" * 55)
print("  STAGE 0 — DATASET SAMPLING")
print("=" * 55)

if not os.path.exists(INPUT):
    print(f"[ERROR] Full dataset not found at: {INPUT}")
    print("  Place datasetuc.csv (pipe-separated) in data/raw/ and re-run.")
    exit(1)

random.seed(42)
samples = []

print(f"Sampling {TARGET_SIZE:,} rows from full dataset...")

for chunk in pd.read_csv(INPUT, sep="|", chunksize=CHUNK_SIZE, low_memory=False):
    chunk["TRANSACTION_DATE"] = pd.to_datetime(chunk["TRANSACTION_DATE"], errors="coerce")
    chunk = chunk.dropna(subset=["TRANSACTION_DATE", "BUYER_STATE"])
    chunk["year"] = chunk["TRANSACTION_DATE"].dt.year
    chunk = chunk.sample(frac=1)
    keep = int(TARGET_SIZE / 200)
    samples.append(chunk.head(keep))
    if sum(len(s) for s in samples) >= TARGET_SIZE:
        break

sampled = pd.concat(samples)
sampled = sampled.sample(min(TARGET_SIZE, len(sampled)), random_state=42)
sampled = sampled.drop(columns=["year"])

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
sampled.to_csv(OUTPUT, index=False)

print(f"Sampled rows : {len(sampled):,}")
print(f"Saved → {OUTPUT}")
print(f"\nNext step: run python3 main.py (or python3 main.py --from 1)")
