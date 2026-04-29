"""
notebooks/03_features.py
─────────────────────────
Phase 2 → Feature Engineering

What it does:
    - Loads transactions.parquet
    - Engineers new features (ratios, flags, encodings)
    - Performs time-aware train/test split (no leakage)
    - Applies SMOTE to handle class imbalance
    - Saves processed splits to data/

Run:
    python notebooks/03_features.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20    # 80/20 split

# ── Load ──────────────────────────────────────────────────────────────────────
print("📂 Loading transactions.parquet …")
df = pd.read_parquet(DATA_DIR / "transactions.parquet")
df["ts"] = pd.to_datetime(df["ts"])
print(f"   {len(df):,} rows loaded")

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n🔧 Engineering features …")

# ── Amount features ───────────────────────────────────────────────────────────
df["log_amount"]          = np.log1p(df["amount"])
df["amount_vs_24h_avg"]   = df["amount"] / (df["prev_24h_amt_card"] / (df["prev_24h_tx_count_card"] + 1) + 1)
df["amount_vs_1h_vel"]    = df["amount"] / (df["velocity_amt_1h"] + 1)

# ── Velocity ratios ───────────────────────────────────────────────────────────
df["tx_count_ratio_1h_24h"] = df["prev_1h_tx_count_card"] / (df["prev_24h_tx_count_card"] + 1)
df["amt_ratio_1h_24h"]      = df["velocity_amt_1h"]        / (df["prev_24h_amt_card"] + 1)

# ── Risk flags ────────────────────────────────────────────────────────────────
HIGH_RISK_CATS     = ["luxury", "electronics", "travel"]
HIGH_RISK_COUNTRIES = ["NG", "RU", "XX", "CN"]

df["is_high_risk_cat"]     = df["merchant_cat"].isin(HIGH_RISK_CATS).astype(int)
df["is_high_risk_country"] = df["country"].isin(HIGH_RISK_COUNTRIES).astype(int)
df["risk_flag_sum"]        = (
    df["is_high_risk_cat"] +
    df["is_high_risk_country"] +
    df["is_international"] +
    df["is_night"]
)

# ── Time features ─────────────────────────────────────────────────────────────
df["is_weekend"]   = (df["dayofweek"] >= 5).astype(int)
df["hour_sin"]     = np.sin(2 * np.pi * df["hour"] / 24)   # cyclical encoding
df["hour_cos"]     = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"]      = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"]      = np.cos(2 * np.pi * df["dayofweek"] / 7)

# ── Burst transaction flag ─────────────────────────────────────────────────────
df["is_burst"] = ((df["prev_1h_tx_count_card"] >= 5) | (df["tx_count_ratio_1h_24h"] > 0.5)).astype(int)

# ── Label encoding for categoricals ──────────────────────────────────────────
print("🔧 Encoding categoricals …")
le_dict = {}
cat_cols = ["merchant_cat", "device_type", "channel", "country", "city"]
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# Save encoders for serving
joblib.dump(le_dict, DATA_DIR / "label_encoders.joblib")
print(f"   ✅ Label encoders saved → data/label_encoders.joblib")

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE LIST
# ═══════════════════════════════════════════════════════════════════════════════
FEATURES = [
    # Raw numerics
    "amount", "log_amount",
    "prev_24h_tx_count_card", "prev_24h_amt_card",
    "prev_1h_tx_count_card",  "velocity_amt_1h",
    "is_international", "is_night",
    # Engineered
    "amount_vs_24h_avg", "amount_vs_1h_vel",
    "tx_count_ratio_1h_24h", "amt_ratio_1h_24h",
    "is_high_risk_cat", "is_high_risk_country", "risk_flag_sum",
    "is_weekend", "is_burst",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # Encoded categoricals
    "merchant_cat_enc", "device_type_enc", "channel_enc",
    "country_enc", "city_enc",
]

TARGET = "is_fraud"

print(f"\n   Total features : {len(FEATURES)}")
print(f"   Feature list   : {FEATURES}")

# ═══════════════════════════════════════════════════════════════════════════════
# TIME-AWARE TRAIN / TEST SPLIT  (no leakage — test = last 20% by time)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n✂️  Time-aware train/test split …")
df_sorted   = df.sort_values("ts").reset_index(drop=True)
split_idx   = int(len(df_sorted) * (1 - TEST_SIZE))
split_ts    = df_sorted.loc[split_idx, "ts"]

train_df = df_sorted.iloc[:split_idx]
test_df  = df_sorted.iloc[split_idx:]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test  = test_df[FEATURES]
y_test  = test_df[TARGET]

print(f"   Train size  : {len(X_train):,}  | Fraud: {y_train.sum():,}  ({y_train.mean()*100:.2f}%)")
print(f"   Test size   : {len(X_test):,}   | Fraud: {y_test.sum():,}  ({y_test.mean()*100:.2f}%)")
print(f"   Split date  : {split_ts.date()}")

# ═══════════════════════════════════════════════════════════════════════════════
# SMOTE — Oversample minority class in training set only
# ═══════════════════════════════════════════════════════════════════════════════
print("\n⚖️  Applying SMOTE to training set …")
smote = SMOTE(sampling_strategy=0.2, random_state=RANDOM_STATE, n_jobs=-1)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"   Before SMOTE : {len(X_train):,}  fraud={y_train.sum():,}  ({y_train.mean()*100:.2f}%)")
print(f"   After  SMOTE : {len(X_train_res):,}  fraud={y_train_res.sum():,}  ({y_train_res.mean()*100:.2f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE PROCESSED SPLITS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n💾 Saving processed splits …")

pd.DataFrame(X_train_res, columns=FEATURES).to_parquet(DATA_DIR / "X_train.parquet", index=False)
pd.Series(y_train_res, name=TARGET).to_frame().to_parquet(DATA_DIR / "y_train.parquet", index=False)
X_test.to_parquet(DATA_DIR / "X_test.parquet", index=False)
y_test.to_frame().to_parquet(DATA_DIR / "y_test.parquet", index=False)

# Save feature list for model training
joblib.dump(FEATURES, DATA_DIR / "feature_list.joblib")

print(f"   ✅ data/X_train.parquet  ({len(X_train_res):,} rows)")
print(f"   ✅ data/y_train.parquet")
print(f"   ✅ data/X_test.parquet   ({len(X_test):,} rows)")
print(f"   ✅ data/y_test.parquet")
print(f"   ✅ data/feature_list.joblib")

print("\n" + "="*55)
print("✅ Feature engineering complete!")
print("   Next → python notebooks/04_train.py")
print("="*55)