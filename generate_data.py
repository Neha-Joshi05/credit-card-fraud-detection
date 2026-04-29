"""
generate_data.py
────────────────
Generates a synthetic credit card transactions dataset for fraud detection.

Run:
    python generate_data.py

Output:
    data/transactions.csv   ← raw synthetic dataset
    data/transactions.parquet ← typed + sorted Parquet version

Fraud rate is intentionally low (~1.5%) to simulate real-world imbalance.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 42
N_TOTAL     = 100_000          # total transactions
FRAUD_RATE  = 0.015            # ~1.5% fraud
OUTPUT_DIR  = Path("data")

np.random.seed(SEED)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _hash_id(prefix: str, n: int, pool: int) -> list[str]:
    """Return n random hashed IDs drawn from a pool of size `pool`."""
    return [f"{prefix}_{np.random.randint(0, pool):06d}" for _ in range(n)]


# ── Core generation ───────────────────────────────────────────────────────────
def generate_transactions(n: int = N_TOTAL, fraud_rate: float = FRAUD_RATE) -> pd.DataFrame:
    n_fraud  = int(n * fraud_rate)
    n_legit  = n - n_fraud
    is_fraud = np.array([0] * n_legit + [1] * n_fraud)

    # ── Timestamps: spread over 90 days ──────────────────────────────────────
    base_time = pd.Timestamp("2024-01-01")
    seconds   = np.random.randint(0, 90 * 86400, size=n)
    timestamps = base_time + pd.to_timedelta(seconds, unit="s")

    # ── Amount ────────────────────────────────────────────────────────────────
    # Legitimate: log-normal centred around ₹500
    # Fraud: bimodal – many small (card-testing) or large (cash-out)
    legit_amt = np.random.lognormal(mean=6.2, sigma=1.1, size=n_legit).clip(10, 150_000)
    fraud_amt = np.where(
        np.random.rand(n_fraud) < 0.4,                      # 40% small-probe
        np.random.uniform(1, 50, size=n_fraud),
        np.random.lognormal(mean=8.5, sigma=0.8, size=n_fraud).clip(500, 500_000),
    )
    amount = np.concatenate([legit_amt, fraud_amt])

    # ── Categorical fields ────────────────────────────────────────────────────
    merchant_cats = ["grocery", "electronics", "travel", "dining",
                     "fuel", "healthcare", "luxury", "online_retail",
                     "entertainment", "utilities"]
    # Fraud skews toward high-risk categories
    legit_cat_prob = [0.20, 0.12, 0.10, 0.15, 0.12, 0.08, 0.05, 0.10, 0.05, 0.03]
    fraud_cat_prob = [0.05, 0.20, 0.15, 0.05, 0.05, 0.03, 0.25, 0.12, 0.05, 0.05]

    merchant_cat = np.concatenate([
        np.random.choice(merchant_cats, size=n_legit, p=legit_cat_prob),
        np.random.choice(merchant_cats, size=n_fraud, p=fraud_cat_prob),
    ])

    countries   = ["IN", "US", "GB", "AE", "SG", "CN", "NG", "BR", "RU", "XX"]
    legit_cntry = [0.60, 0.15, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01]
    fraud_cntry = [0.10, 0.18, 0.10, 0.12, 0.08, 0.10, 0.12, 0.08, 0.08, 0.04]

    country = np.concatenate([
        np.random.choice(countries, size=n_legit, p=legit_cntry),
        np.random.choice(countries, size=n_fraud, p=fraud_cntry),
    ])

    cities = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad",
              "Unknown", "London", "Dubai", "Singapore", "Other"]
    city = np.concatenate([
        np.random.choice(cities[:5] + [cities[-1]], size=n_legit),
        np.random.choice(cities,                    size=n_fraud),
    ])

    device_types = ["mobile", "desktop", "tablet", "pos_terminal", "atm"]
    legit_dev    = [0.45, 0.25, 0.10, 0.15, 0.05]
    fraud_dev    = [0.55, 0.30, 0.10, 0.03, 0.02]
    device_type  = np.concatenate([
        np.random.choice(device_types, size=n_legit, p=legit_dev),
        np.random.choice(device_types, size=n_fraud, p=fraud_dev),
    ])

    channels   = ["online", "pos", "atm", "mobile_app", "ivr"]
    legit_chan  = [0.30, 0.35, 0.10, 0.20, 0.05]
    fraud_chan  = [0.50, 0.20, 0.08, 0.18, 0.04]
    channel     = np.concatenate([
        np.random.choice(channels, size=n_legit, p=legit_chan),
        np.random.choice(channels, size=n_fraud, p=fraud_chan),
    ])

    # ── Velocity / behavioural features ──────────────────────────────────────
    # Legitimate: calm spending; Fraud: burst activity
    prev_24h_tx_count_card = np.concatenate([
        np.random.poisson(lam=3,  size=n_legit).astype(float),
        np.random.poisson(lam=12, size=n_fraud).astype(float),
    ])
    prev_24h_amt_card = np.concatenate([
        np.random.lognormal(6.0, 1.0, size=n_legit),
        np.random.lognormal(8.0, 1.2, size=n_fraud),
    ])
    prev_1h_tx_count_card = np.concatenate([
        np.random.poisson(lam=0.5, size=n_legit).astype(float),
        np.random.poisson(lam=4.0, size=n_fraud).astype(float),
    ])
    velocity_amt_1h = np.concatenate([
        np.random.lognormal(4.5, 1.0, size=n_legit),
        np.random.lognormal(7.5, 1.3, size=n_fraud),
    ])

    # ── Derived time features ─────────────────────────────────────────────────
    hour       = pd.DatetimeIndex(timestamps).hour
    dayofweek  = pd.DatetimeIndex(timestamps).dayofweek

    # Fraud slightly prefers nights (22–5)
    is_night = ((hour >= 22) | (hour <= 5)).astype(int)

    # International flag
    is_international = (country != "IN").astype(int)

    # ── IDs ───────────────────────────────────────────────────────────────────
    tx_id            = [f"TX{i:08d}" for i in range(n)]
    merchant_id_hash = _hash_id("MID", n, pool=2000)
    card_id_hash     = _hash_id("CID", n, pool=8000)

    # ── Assemble ──────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "tx_id":                  tx_id,
        "ts":                     timestamps,
        "amount":                 amount.round(2),
        "merchant_cat":           merchant_cat,
        "merchant_id_hash":       merchant_id_hash,
        "card_id_hash":           card_id_hash,
        "city":                   city,
        "country":                country,
        "device_type":            device_type,
        "channel":                channel,
        "hour":                   hour,
        "dayofweek":              dayofweek,
        "prev_24h_tx_count_card": prev_24h_tx_count_card,
        "prev_24h_amt_card":      prev_24h_amt_card.round(2),
        "prev_1h_tx_count_card":  prev_1h_tx_count_card,
        "velocity_amt_1h":        velocity_amt_1h.round(2),
        "is_international":       is_international,
        "is_night":               is_night,
        "is_fraud":               is_fraud,
    })

    # ── Shuffle and sort by timestamp ─────────────────────────────────────────
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df = df.sort_values("ts").reset_index(drop=True)

    return df


# ── Ingest + type enforcement (mirrors notebooks/01_ingest.py) ─────────────────
SCHEMA = {
    "tx_id":                  "string",
    "ts":                     "string",          # will be parsed to datetime
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


def ingest(df: pd.DataFrame) -> pd.DataFrame:
    """Apply schema enforcement and parse timestamps."""
    df["ts"] = df["ts"].astype(str)
    df = df.astype({k: v for k, v in SCHEMA.items() if k != "ts"})
    df["ts"] = pd.to_datetime(df["ts"])
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔧 Generating synthetic transaction data …")
    df = generate_transactions()

    # Save raw CSV
    csv_path = OUTPUT_DIR / "transactions.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved   → {csv_path}  ({len(df):,} rows)")

    # Save typed Parquet
    df_typed = ingest(df)
    pq_path  = OUTPUT_DIR / "transactions.parquet"
    df_typed.to_parquet(pq_path, index=False)
    print(f"✅ Parquet saved → {pq_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    fraud_count = df["is_fraud"].sum()
    print(f"\n📊 Dataset Summary")
    print(f"   Total rows   : {len(df):,}")
    print(f"   Fraud rows   : {fraud_count:,}  ({fraud_count/len(df)*100:.2f}%)")
    print(f"   Legit rows   : {len(df)-fraud_count:,}")
    ts_parsed = pd.to_datetime(df['ts'])
    print(f"   Date range   : {ts_parsed.min().date()} → {ts_parsed.max().date()}")
    print(f"   Columns      : {list(df.columns)}")
    print(f"\n   Amount stats (all):")
    print(df["amount"].describe().to_string())
    print(f"\n   Amount stats (fraud only):")
    print(df.loc[df.is_fraud==1, "amount"].describe().to_string())