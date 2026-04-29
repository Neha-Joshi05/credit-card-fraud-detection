"""
notebooks/05_explain.py
────────────────────────
Phase 3 → Model Explainability with SHAP

What it does:
    - Loads trained XGBoost model
    - Computes SHAP values on test set
    - Generates summary, waterfall, beeswarm, dependence plots
    - Explains individual fraud predictions

Run:
    python notebooks/05_explain.py
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
MODEL_DIR  = Path("models")
OUTPUT_DIR = Path("outputs")

FRAUD_COLOR = "#e74c3c"
LEGIT_COLOR = "#2ecc71"

# ── Load ──────────────────────────────────────────────────────────────────────
print("📂 Loading model and test data …")
model     = joblib.load(MODEL_DIR / "xgb_fraud_model.joblib")
threshold = joblib.load(MODEL_DIR / "optimal_threshold.joblib")
FEATURES  = joblib.load(DATA_DIR  / "feature_list.joblib")

X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")
y_test = pd.read_parquet(DATA_DIR / "y_test.parquet").squeeze()

print(f"   Test set : {X_test.shape}  |  threshold={threshold:.3f}")

# ── Sample for SHAP (use 2000 rows max for speed) ─────────────────────────────
shap_sample = X_test.sample(min(2000, len(X_test)), random_state=42)
print(f"   SHAP sample : {len(shap_sample)} rows")

# ═══════════════════════════════════════════════════════════════════════════════
# SHAP EXPLAINER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n🔍 Computing SHAP values …")
explainer   = shap.TreeExplainer(model)
shap_values = explainer(shap_sample)
print("   ✅ SHAP values computed")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — SHAP Summary (Beeswarm)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n📊 Chart 1: SHAP beeswarm summary …")
fig, ax = plt.subplots(figsize=(10, 9))
shap.summary_plot(shap_values, shap_sample, show=False,
                  plot_size=None, max_display=20)
plt.title("SHAP Feature Impact — Beeswarm", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "14_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/14_shap_beeswarm.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — SHAP Bar Summary (mean |SHAP|)
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 2: SHAP bar summary …")
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, shap_sample, plot_type="bar",
                  show=False, plot_size=None, max_display=20)
plt.title("Mean |SHAP| Feature Importance", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "15_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/15_shap_bar.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — SHAP Dependence plots (top 4 features)
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 3: SHAP dependence plots …")
mean_shap   = np.abs(shap_values.values).mean(axis=0)
top4_idx    = np.argsort(mean_shap)[::-1][:4]
top4_feats  = [FEATURES[i] for i in top4_idx]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("SHAP Dependence Plots — Top 4 Features", fontsize=13, fontweight="bold")

for ax, feat in zip(axes.flatten(), top4_feats):
    shap.dependence_plot(feat, shap_values.values, shap_sample,
                         ax=ax, show=False, dot_size=8, alpha=0.4)
    ax.set_title(feat, fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "16_shap_dependence.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ outputs/16_shap_dependence.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Waterfall plot: Top fraud case
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 4: SHAP waterfall — top fraud case …")

# Get the fraud case with highest model score from sample
y_test_sample = y_test.loc[shap_sample.index]
probs         = model.predict_proba(shap_sample)[:, 1]
fraud_mask    = y_test_sample.values == 1

if fraud_mask.sum() > 0:
    fraud_probs = probs[fraud_mask]
    top_fraud_local_idx = np.argmax(fraud_probs)
    fraud_global_indices = np.where(fraud_mask)[0]
    top_fraud_idx = fraud_global_indices[top_fraud_local_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(shap_values[top_fraud_idx], max_display=15, show=False)
    plt.title(f"SHAP Waterfall — Highest-Confidence Fraud  (score={probs[top_fraud_idx]:.3f})",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "17_shap_waterfall_fraud.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ outputs/17_shap_waterfall_fraud.png")
else:
    print("   ⚠️  No fraud cases in sample — skipping waterfall")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Waterfall: Legit case near threshold (borderline)
# ═══════════════════════════════════════════════════════════════════════════════
print("📊 Chart 5: SHAP waterfall — borderline legit …")

legit_mask = (y_test_sample.values == 0) & (probs > 0.3)
if legit_mask.sum() > 0:
    border_probs = probs[legit_mask]
    border_local = np.argmax(border_probs)
    border_globals = np.where(legit_mask)[0]
    border_idx = border_globals[border_local]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(shap_values[border_idx], max_display=15, show=False)
    plt.title(f"SHAP Waterfall — Borderline Legit  (score={probs[border_idx]:.3f})",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "18_shap_waterfall_legit.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ outputs/18_shap_waterfall_legit.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT: Top 10 features by mean |SHAP|
# ═══════════════════════════════════════════════════════════════════════════════
shap_df = pd.DataFrame({
    "feature":    FEATURES,
    "mean_shap":  mean_shap,
}).sort_values("mean_shap", ascending=False)

print("\n" + "="*55)
print("📋 Top 10 Features by Mean |SHAP|")
print("="*55)
print(shap_df.head(10).to_string(index=False))

print("\n" + "="*55)
print("✅ SHAP explainability complete!")
print("   All charts saved to outputs/ (14–18)")
print("   Next → python serving/api.py  (Phase 4 — FastAPI)")
print("="*55)