from pathlib import Path
import sys
import json
from datetime import datetime, timezone
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from db import init_db, save_transaction, get_transactions, get_summary

MODEL_FILE = ROOT / "models" / "fraud_model.pkl"
PREPROCESSOR_FILE = ROOT / "models" / "preprocessing.pkl"
METADATA_FILE = ROOT / "models" / "model_metadata.json"

app = FastAPI(title="Online Payment Fraud Detection API", version="1.0")
model = None
preprocessor = None
metadata = {}


class Payment(BaseModel):
    type: str = Field(..., min_length=1)
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)


def load_model():
    global model, preprocessor, metadata
    if not MODEL_FILE.exists() or not PREPROCESSOR_FILE.exists():
        raise RuntimeError("Model files are missing. Run: python src/train.py")
    model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    if METADATA_FILE.exists():
        metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))


@app.on_event("startup")
def startup():
    init_db()
    load_model()


@app.get("/")
def root():
    return {"service": "Online Payment Fraud Detection", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": metadata.get("model", "unknown")}


@app.post("/predict")
def predict(payment: Payment):
    try:
        frame = pd.DataFrame([payment.model_dump()])
        X = preprocessor.transform(frame)
        probability = float(model.predict_proba(X)[0][1])
        threshold = float(metadata.get("risk_threshold", 0.50))
        prediction = int(probability >= threshold)
        if probability >= 0.70:
            decision = "BLOCK"
        elif probability >= 0.30:
            decision = "REVIEW"
        else:
            decision = "ALLOW"

        transaction = {
            "transaction_id": f"TXN-{datetime.now().strftime('%Y%m%d')}-{uuid4().hex[:8].upper()}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payment.model_dump(),
            "errorBalanceOrig": payment.newbalanceOrig + payment.amount - payment.oldbalanceOrg,
            "errorBalanceDest": payment.oldbalanceDest + payment.amount - payment.newbalanceDest,
            "fraud_probability": probability,
            "prediction": prediction,
            "decision": decision,
        }
        save_transaction(transaction)
        return {**transaction, "model": metadata.get("model", "unknown")}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/transactions")
def transactions(limit: int = 200):
    return get_transactions(max(1, min(limit, 1000))).to_dict(orient="records")


@app.get("/summary")
def summary():
    return get_summary()
