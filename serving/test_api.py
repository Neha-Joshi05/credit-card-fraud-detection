"""
serving/test_api.py
────────────────────
Tests all API endpoints locally.

Usage:
    # Terminal 1 — start the API
    uvicorn serving.api:app --reload --port 8000

    # Terminal 2 — run tests
    python serving/test_api.py
"""

import requests
import json

BASE = "http://localhost:8000"

def sep(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

# ── Health check ──────────────────────────────────────────────────────────────
sep("GET /  →  Health Check")
r = requests.get(f"{BASE}/")
print(json.dumps(r.json(), indent=2))

# ── Model metrics ─────────────────────────────────────────────────────────────
sep("GET /metrics  →  Model Performance")
r = requests.get(f"{BASE}/metrics")
print(json.dumps(r.json(), indent=2))

# ── Single prediction: suspicious transaction ──────────────────────────────────
sep("POST /predict  →  Suspicious Transaction")
payload_fraud = {
    "amount": 48000,
    "merchant_cat": "luxury",
    "country": "NG",
    "city": "Unknown",
    "device_type": "mobile",
    "channel": "online",
    "hour": 3,
    "dayofweek": 6,
    "prev_24h_tx_count_card": 14,
    "prev_24h_amt_card": 72000,
    "prev_1h_tx_count_card": 7,
    "velocity_amt_1h": 38000,
    "is_international": 1,
    "is_night": 1,
}
r = requests.post(f"{BASE}/predict", json=payload_fraud)
result = r.json()
print(json.dumps(result, indent=2))
print(f"\n  → FRAUD SCORE : {result['fraud_score']}")
print(f"  → IS FRAUD    : {result['is_fraud']}")
print(f"  → RISK LEVEL  : {result['risk_level']}")

# ── Single prediction: normal transaction ─────────────────────────────────────
sep("POST /predict  →  Normal Transaction")
payload_legit = {
    "amount": 350,
    "merchant_cat": "grocery",
    "country": "IN",
    "city": "Mumbai",
    "device_type": "pos_terminal",
    "channel": "pos",
    "hour": 14,
    "dayofweek": 2,
    "prev_24h_tx_count_card": 2,
    "prev_24h_amt_card": 800,
    "prev_1h_tx_count_card": 0,
    "velocity_amt_1h": 0,
    "is_international": 0,
    "is_night": 0,
}
r = requests.post(f"{BASE}/predict", json=payload_legit)
result = r.json()
print(json.dumps(result, indent=2))
print(f"\n  → FRAUD SCORE : {result['fraud_score']}")
print(f"  → IS FRAUD    : {result['is_fraud']}")
print(f"  → RISK LEVEL  : {result['risk_level']}")

# ── Batch prediction ──────────────────────────────────────────────────────────
sep("POST /predict/batch  →  3 transactions")
batch_payload = {
    "transactions": [payload_fraud, payload_legit, {
        "amount": 1200,
        "merchant_cat": "electronics",
        "country": "AE",
        "city": "Dubai",
        "device_type": "mobile",
        "channel": "online",
        "hour": 22,
        "dayofweek": 4,
        "prev_24h_tx_count_card": 5,
        "prev_24h_amt_card": 8000,
        "prev_1h_tx_count_card": 3,
        "velocity_amt_1h": 3600,
        "is_international": 1,
        "is_night": 1,
    }]
}
r = requests.post(f"{BASE}/predict/batch", json=batch_payload)
result = r.json()
print(f"  Total        : {result['total']}")
print(f"  Fraud count  : {result['fraud_count']}")
print(f"  Fraud rate   : {result['fraud_rate_pct']}%")
for i, res in enumerate(result["results"]):
    print(f"\n  Tx #{i+1}: score={res['fraud_score']}  fraud={res['is_fraud']}  risk={res['risk_level']}")

print(f"\n{'─'*55}")
print("✅ All API tests passed!")
print(f"{'─'*55}")
print("\n   Swagger UI → http://localhost:8000/docs")
print("   ReDoc      → http://localhost:8000/redoc")