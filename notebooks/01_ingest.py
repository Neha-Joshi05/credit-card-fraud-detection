"""
notebooks/01_ingest.py
──────────────────────
Phase 1 → Step 2: Data Ingestion with dtype enforcement.

What it does:
    - Loads transactions.csv
    - Enforces column dtypes (schema validation)
    - Parses timestamps
    - Saves cleaned Parquet for downstream use

Run:
    python notebooks/01_ingest.py
"""

import pandas as pd
from pathlib import Path

# ── Schema ────────────────────────────────────────────────────────────────────
SCHEMA = {
    "tx_id":                  "string",
    "amount":                 "float64",
    "merchant_cat":           "category",
    "merchant_id_hash":       "string",
    "card_id_hash":           "string",
    "city":                   "category",
    "country":                "category",
    "device_type":            "category",
    "channel":                "category",
    "hour":                   "int64",
    "dayofweek":              "int64",
    "prev_24h_tx_count_card": "float64",
    "prev_24h_amt_card":      "float64",
    "prev_1h_tx_count_card":  "float64",
    "velocity_amt_1h":        "float64",
    "is_international":       "int64",
    "is_night":               "int64",
    "is_fraud":               "int64",
}

# ── Load ──────────────────────────────────────────────────────────────────────
csv_path = Path("data/transactions.csv")

if not csv_path.exists():
    raise FileNotFoundError(
        "❌ data/transactions.csv not found.\n"
        "   Run: python generate_data.py  first."
    )

print(f"📂 Loading {csv_path} …")
df = pd.read_csv(csv_path)

# ── Type enforcement ──────────────────────────────────────────────────────────
print("🔧 Enforcing schema …")
df = df.astype(SCHEMA)
df["ts"] = pd.to_datetime(df["ts"])

# ── Sort chronologically (critical for time-aware split later) ────────────────
df = df.sort_values("ts").reset_index(drop=True)

# ── Basic validation ──────────────────────────────────────────────────────────
assert df["is_fraud"].isin([0, 1]).all(),  "is_fraud must be 0 or 1"
assert df["amount"].ge(0).all(),           "amount must be non-negative"
assert df["ts"].is_monotonic_increasing,   "timestamps must be sorted"

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = Path("data/transactions.parquet")
df.to_parquet(out_path, index=False)

# ── Report ────────────────────────────────────────────────────────────────────
n_total = len(df)
n_fraud = int(df["is_fraud"].sum())

print(f"\n✅ Ingest complete → {out_path}")
print(f"\n   Shape         : {df.shape}")
print(f"   Fraud rate    : {n_fraud/n_total*100:.2f}%  ({n_fraud:,} / {n_total:,})")
print(f"   Date range    : {df['ts'].min().date()} → {df['ts'].max().date()}")
print(f"   Null counts   :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\n   dtypes:\n{df.dtypes}")