"""
notebooks/02_eda.py
───────────────────
Phase 2 → Exploratory Data Analysis

What it does:
    - Loads transactions.parquet
    - Plots fraud vs legit distributions
    - Analyses amount, time, category, velocity features
    - Saves all charts to outputs/

Run:
    python notebooks/02_eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
FRAUD_COLOR = "#e74c3c"
LEGIT_COLOR = "#2ecc71"
PALETTE     = [LEGIT_COLOR, FRAUD_COLOR]

# ── Load ──────────────────────────────────────────────────────────────────────
pq_path = Path("data/transactions.parquet")
if not pq_path.exists():
    raise FileNotFoundError("❌ Run python notebooks/01_ingest.py first.")

print("📂 Loading data …")
df = pd.read_parquet(pq_path)
df["ts"] = pd.to_datetime(df["ts"])

fraud = df[df.is_fraud == 1]
legit = df[df.is_fraud == 0]
print(f"   {len(df):,} rows | Fraud: {len(fraud):,} ({len(fraud)/len(df)*100:.2f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Class Imbalance
# ═══════════════════════════════════════════════════════════════════════════════
print("\n📊 Chart 1: Class imbalance …")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Class Distribution — Fraud vs Legitimate", fontsize=14, fontweight="bold")

counts = df["is_fraud"].value_counts()
labels = ["Legitimate", "Fraud"]
colors = [LEGIT_COLOR, FRAUD_COLOR]

axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
axes[0].set_title("Transaction Count")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 200, f"{v:,}", ha="center", fontweight="bold")

axes[1].pie(counts.values, labels=labels, colors=colors,
            autopct="%1.2f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("Proportion")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_class_imbalance.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/01_class_imbalance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Amount Distribution
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 2: Amount distribution …")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Transaction Amount Analysis", fontsize=14, fontweight="bold")

# Log-scale histogram overlay
axes[0].hist(np.log1p(legit["amount"]),  bins=60, alpha=0.6, color=LEGIT_COLOR, label="Legit",  density=True)
axes[0].hist(np.log1p(fraud["amount"]),  bins=60, alpha=0.7, color=FRAUD_COLOR, label="Fraud",  density=True)
axes[0].set_title("log(1+Amount) Distribution")
axes[0].set_xlabel("log(1 + Amount)")
axes[0].legend()

# Box plot
bp_data = [legit["amount"].clip(0, 5000), fraud["amount"].clip(0, 5000)]
bp = axes[1].boxplot(bp_data, labels=["Legit", "Fraud"],
                     patch_artist=True, notch=True)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_title("Amount Box Plot (clipped ₹5k)")
axes[1].set_ylabel("Amount")

# Cumulative distribution
for label, group, color in [("Legit", legit, LEGIT_COLOR), ("Fraud", fraud, FRAUD_COLOR)]:
    sorted_amt = np.sort(group["amount"].clip(0, 50_000))
    cdf = np.arange(1, len(sorted_amt)+1) / len(sorted_amt)
    axes[2].plot(sorted_amt, cdf, label=label, color=color)
axes[2].set_title("CDF of Amount (clipped ₹50k)")
axes[2].set_xlabel("Amount")
axes[2].set_ylabel("Cumulative Probability")
axes[2].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_amount_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/02_amount_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Time Patterns
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 3: Time patterns …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Temporal Fraud Patterns", fontsize=14, fontweight="bold")

# Hourly fraud rate
hourly = df.groupby("hour")["is_fraud"].agg(["mean", "sum", "count"]).reset_index()
hourly["fraud_rate_pct"] = hourly["mean"] * 100

ax = axes[0]
bars = ax.bar(hourly["hour"], hourly["fraud_rate_pct"],
              color=plt.cm.RdYlGn_r(hourly["fraud_rate_pct"] / hourly["fraud_rate_pct"].max()),
              edgecolor="white")
ax.set_title("Fraud Rate by Hour of Day")
ax.set_xlabel("Hour (24h)")
ax.set_ylabel("Fraud Rate (%)")
ax.set_xticks(range(0, 24, 2))

# Day of week fraud rate
day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
daily = df.groupby("dayofweek")["is_fraud"].mean() * 100

axes[1].bar(day_labels, daily.values,
            color=[FRAUD_COLOR if v > daily.mean() else LEGIT_COLOR for v in daily.values],
            edgecolor="white")
axes[1].axhline(daily.mean(), linestyle="--", color="orange", linewidth=1.5, label=f"Mean {daily.mean():.2f}%")
axes[1].set_title("Fraud Rate by Day of Week")
axes[1].set_ylabel("Fraud Rate (%)")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_time_patterns.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/03_time_patterns.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Merchant Category Fraud Rate
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 4: Merchant category …")
fig, ax = plt.subplots(figsize=(12, 6))

cat_stats = df.groupby("merchant_cat")["is_fraud"].agg(["mean", "sum", "count"])
cat_stats["fraud_rate_pct"] = cat_stats["mean"] * 100
cat_stats = cat_stats.sort_values("fraud_rate_pct", ascending=True)

colors_cat = [FRAUD_COLOR if v > cat_stats["fraud_rate_pct"].mean() else LEGIT_COLOR
              for v in cat_stats["fraud_rate_pct"]]

bars = ax.barh(cat_stats.index, cat_stats["fraud_rate_pct"],
               color=colors_cat, edgecolor="white")
ax.axvline(cat_stats["fraud_rate_pct"].mean(), linestyle="--", color="orange",
           linewidth=1.5, label=f"Overall avg {cat_stats['fraud_rate_pct'].mean():.2f}%")

for bar, val in zip(bars, cat_stats["fraud_rate_pct"]):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}%", va="center", fontsize=9)

ax.set_title("Fraud Rate by Merchant Category", fontsize=13, fontweight="bold")
ax.set_xlabel("Fraud Rate (%)")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_merchant_category.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/04_merchant_category.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Velocity Features
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 5: Velocity features …")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Velocity / Behavioural Feature Analysis", fontsize=14, fontweight="bold")

velocity_features = [
    ("prev_24h_tx_count_card", "Transactions in Last 24h"),
    ("prev_1h_tx_count_card",  "Transactions in Last 1h"),
    ("prev_24h_amt_card",      "Amount Spent in Last 24h"),
    ("velocity_amt_1h",        "Amount Velocity (1h)"),
]

for ax, (col, title) in zip(axes.flatten(), velocity_features):
    clip_val = df[col].quantile(0.99)
    ax.hist(legit[col].clip(0, clip_val), bins=50, alpha=0.6,
            color=LEGIT_COLOR, label="Legit", density=True)
    ax.hist(fraud[col].clip(0, clip_val), bins=50, alpha=0.7,
            color=FRAUD_COLOR, label="Fraud", density=True)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_velocity_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/05_velocity_features.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 6: Correlation heatmap …")
numeric_cols = [
    "amount", "hour", "dayofweek",
    "prev_24h_tx_count_card", "prev_24h_amt_card",
    "prev_1h_tx_count_card",  "velocity_amt_1h",
    "is_international", "is_night", "is_fraud"
]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 9})
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/06_correlation_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 7 — Country & Channel breakdown
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 7: Country & channel …")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Geographic & Channel Fraud Patterns", fontsize=14, fontweight="bold")

# Country fraud rate (top 10)
country_stats = df.groupby("country")["is_fraud"].agg(["mean","count"])
country_stats["fraud_rate_pct"] = country_stats["mean"] * 100
country_stats = country_stats.sort_values("fraud_rate_pct", ascending=False).head(10)

axes[0].bar(country_stats.index, country_stats["fraud_rate_pct"],
            color=FRAUD_COLOR, alpha=0.8, edgecolor="white")
axes[0].axhline(df["is_fraud"].mean()*100, linestyle="--", color="orange",
                linewidth=1.5, label="Overall avg")
axes[0].set_title("Fraud Rate by Country (Top 10)")
axes[0].set_ylabel("Fraud Rate (%)")
axes[0].legend()

# Channel fraud rate
chan_stats = df.groupby("channel")["is_fraud"].mean().sort_values() * 100
colors_chan = [FRAUD_COLOR if v > chan_stats.mean() else LEGIT_COLOR for v in chan_stats.values]
axes[1].barh(chan_stats.index, chan_stats.values, color=colors_chan, edgecolor="white")
axes[1].axvline(chan_stats.mean(), linestyle="--", color="orange", linewidth=1.5)
axes[1].set_title("Fraud Rate by Channel")
axes[1].set_xlabel("Fraud Rate (%)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_geo_channel.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/07_geo_channel.png")

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("📋 EDA Key Findings")
print("="*55)
print(f"\n  Fraud rate         : {df['is_fraud'].mean()*100:.2f}%")
print(f"  Avg fraud amount   : ₹{fraud['amount'].mean():,.2f}")
print(f"  Avg legit amount   : ₹{legit['amount'].mean():,.2f}")
print(f"  Highest fraud cat  : {cat_stats['fraud_rate_pct'].idxmax()}")
print(f"  Peak fraud hour    : {hourly.loc[hourly['fraud_rate_pct'].idxmax(), 'hour']}:00")
print(f"\n  Charts saved to    : outputs/  (7 files)")
print("\n✅ EDA complete! Next → python notebooks/03_features.py")