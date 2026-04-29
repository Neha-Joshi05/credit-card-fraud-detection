"""
serving/api.py
──────────────
Phase 4 → FastAPI Fraud Detection API

Endpoints:
    GET  /              → health check
    GET  /metrics       → model performance metrics
    POST /predict       → single transaction scoring
    POST /predict/batch → batch transaction scoring (up to 1000)

Run:
    uvicorn serving.api:app --reload --port 8000

Or from project root:
    python serving/api.py
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ── Load model artifacts ──────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODEL_DIR  = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"

model     = joblib.load(MODEL_DIR / "xgb_fraud_model.joblib")
threshold = joblib.load(MODEL_DIR / "optimal_threshold.joblib")
features  = joblib.load(DATA_DIR  / "feature_list.joblib")
metrics   = joblib.load(MODEL_DIR / "eval_metrics.joblib")
le_dict   = joblib.load(DATA_DIR  / "label_encoders.joblib")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Credit Card Fraud Detection API",
    description = "Real-time fraud scoring powered by XGBoost + SHAP",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Request / Response schemas ────────────────────────────────────────────────
class TransactionInput(BaseModel):
    amount:                  float  = Field(..., gt=0,   description="Transaction amount in INR")
    merchant_cat:            str    = Field(...,          description="Merchant category")
    country:                 str    = Field(...,          description="Country code (e.g. IN, US)")
    city:                    str    = Field(...,          description="City name")
    device_type:             str    = Field(...,          description="Device type")
    channel:                 str    = Field(...,          description="Transaction channel")
    hour:                    int    = Field(..., ge=0, le=23)
    dayofweek:               int    = Field(..., ge=0, le=6)
    prev_24h_tx_count_card:  float  = Field(0.0, ge=0)
    prev_24h_amt_card:       float  = Field(0.0, ge=0)
    prev_1h_tx_count_card:   float  = Field(0.0, ge=0)
    velocity_amt_1h:         float  = Field(0.0, ge=0)
    is_international:        int    = Field(0,  ge=0, le=1)
    is_night:                int    = Field(0,  ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "amount": 4500.00,
                "merchant_cat": "electronics",
                "country": "IN",
                "city": "Mumbai",
                "device_type": "mobile",
                "channel": "online",
                "hour": 23,
                "dayofweek": 5,
                "prev_24h_tx_count_card": 8,
                "prev_24h_amt_card": 15000,
                "prev_1h_tx_count_card": 4,
                "velocity_amt_1h": 9000,
                "is_international": 0,
                "is_night": 1,
            }
        }


class PredictionResponse(BaseModel):
    fraud_score:    float
    is_fraud:       bool
    risk_level:     str
    threshold_used: float
    top_risk_factors: list[str]
    timestamp:      str


class BatchRequest(BaseModel):
    transactions: list[TransactionInput] = Field(..., max_items=1000)


class BatchResponse(BaseModel):
    total:         int
    fraud_count:   int
    fraud_rate_pct: float
    results:       list[PredictionResponse]


# ── Feature engineering (mirrors notebooks/03_features.py) ───────────────────
HIGH_RISK_CATS      = {"luxury", "electronics", "travel"}
HIGH_RISK_COUNTRIES = {"NG", "RU", "XX", "CN"}

def engineer_features(tx: TransactionInput) -> pd.DataFrame:
    """Build the 25-feature vector from raw transaction input."""
    row = tx.dict()

    # Engineered
    row["log_amount"]             = np.log1p(row["amount"])
    row["amount_vs_24h_avg"]      = row["amount"] / (row["prev_24h_amt_card"] / (row["prev_24h_tx_count_card"] + 1) + 1)
    row["amount_vs_1h_vel"]       = row["amount"] / (row["velocity_amt_1h"] + 1)
    row["tx_count_ratio_1h_24h"]  = row["prev_1h_tx_count_card"] / (row["prev_24h_tx_count_card"] + 1)
    row["amt_ratio_1h_24h"]       = row["velocity_amt_1h"] / (row["prev_24h_amt_card"] + 1)
    row["is_high_risk_cat"]       = int(row["merchant_cat"] in HIGH_RISK_CATS)
    row["is_high_risk_country"]   = int(row["country"] in HIGH_RISK_COUNTRIES)
    row["risk_flag_sum"]          = (row["is_high_risk_cat"] + row["is_high_risk_country"] +
                                     row["is_international"] + row["is_night"])
    row["is_weekend"]             = int(row["dayofweek"] >= 5)
    row["is_burst"]               = int(row["prev_1h_tx_count_card"] >= 5 or
                                        row["tx_count_ratio_1h_24h"] > 0.5)
    row["hour_sin"]               = np.sin(2 * np.pi * row["hour"] / 24)
    row["hour_cos"]               = np.cos(2 * np.pi * row["hour"] / 24)
    row["dow_sin"]                = np.sin(2 * np.pi * row["dayofweek"] / 7)
    row["dow_cos"]                = np.cos(2 * np.pi * row["dayofweek"] / 7)

    # Label encode categoricals — handle unseen labels gracefully
    for col in ["merchant_cat", "device_type", "channel", "country", "city"]:
        le = le_dict[col]
        val = row[col]
        if val in le.classes_:
            row[col + "_enc"] = int(le.transform([val])[0])
        else:
            row[col + "_enc"] = 0   # fallback for unseen

    df = pd.DataFrame([row])
    return df[features]


def get_risk_level(score: float) -> str:
    if score >= 0.75:  return "CRITICAL"
    if score >= 0.50:  return "HIGH"
    if score >= 0.25:  return "MEDIUM"
    return "LOW"


def get_top_risk_factors(df_row: pd.DataFrame, score: float) -> list[str]:
    """Simple rule-based risk factor explanation (fast, no SHAP overhead)."""
    factors = []
    r = df_row.iloc[0]

    if r["is_night"]:               factors.append("Night-time transaction")
    if r["is_international"]:       factors.append("International transaction")
    if r["is_high_risk_cat"]:       factors.append(f"High-risk merchant category")
    if r["is_high_risk_country"]:   factors.append("High-risk country")
    if r["is_burst"]:               factors.append("Burst transaction activity detected")
    if r["amount_vs_24h_avg"] > 5:  factors.append("Amount unusually high vs 24h average")
    if r["amount_vs_1h_vel"] > 5:   factors.append("Amount unusually high vs 1h velocity")
    if r["prev_1h_tx_count_card"] >= 4: factors.append("High transaction count in last hour")
    if r["risk_flag_sum"] >= 3:     factors.append("Multiple risk flags triggered")

    if not factors:
        factors.append("No specific risk factors — model pattern-based detection")

    return factors[:5]   # top 5


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    return {
        "status":    "online",
        "service":   "Credit Card Fraud Detection API",
        "version":   "1.0.0",
        "threshold": round(threshold, 3),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/metrics", tags=["Model"])
def get_metrics():
    return {
        "model":     "XGBoost (Optuna-tuned)",
        "threshold": round(threshold, 3),
        "metrics": {
            "roc_auc":   round(metrics["roc_auc"],   4),
            "pr_auc":    round(metrics["pr_auc"],    4),
            "precision": round(metrics["precision"], 4),
            "recall":    round(metrics["recall"],    4),
            "f1_score":  round(metrics["f1"],        4),
        },
        "features": len(features),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(tx: TransactionInput):
    try:
        df_row    = engineer_features(tx)
        score     = float(model.predict_proba(df_row)[0, 1])
        is_fraud  = score >= threshold
        factors   = get_top_risk_factors(df_row, score)

        return PredictionResponse(
            fraud_score      = round(score, 4),
            is_fraud         = is_fraud,
            risk_level       = get_risk_level(score),
            threshold_used   = round(threshold, 3),
            top_risk_factors = factors,
            timestamp        = datetime.utcnow().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(req: BatchRequest):
    try:
        results = []
        for tx in req.transactions:
            df_row   = engineer_features(tx)
            score    = float(model.predict_proba(df_row)[0, 1])
            is_fraud = score >= threshold
            factors  = get_top_risk_factors(df_row, score)
            results.append(PredictionResponse(
                fraud_score      = round(score, 4),
                is_fraud         = is_fraud,
                risk_level       = get_risk_level(score),
                threshold_used   = round(threshold, 3),
                top_risk_factors = factors,
                timestamp        = datetime.utcnow().isoformat(),
            ))

        fraud_count = sum(r.is_fraud for r in results)
        return BatchResponse(
            total          = len(results),
            fraud_count    = fraud_count,
            fraud_rate_pct = round(fraud_count / len(results) * 100, 2),
            results        = results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serving.api:app", host="0.0.0.0", port=8000, reload=True)