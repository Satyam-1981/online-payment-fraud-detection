from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import db

if not (ROOT / "models" / "fraud_model.pkl").exists():
    pytest.skip("Train the model first with: python src/train.py", allow_module_level=True)

from fastapi.testclient import TestClient
from api import app


@pytest.fixture
def client(tmp_path):
    original_db = db.DB_FILE
    db.DB_FILE = tmp_path / "test_fraud.db"
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        db.DB_FILE = original_db


def payment_payload():
    return {
        "type": "TRANSFER",
        "amount": 5000.0,
        "oldbalanceOrg": 25000.0,
        "newbalanceOrig": 20000.0,
        "oldbalanceDest": 15000.0,
        "newbalanceDest": 20000.0,
    }


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_valid_prediction(client):
    response = client.post("/predict", json=payment_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] in {"ALLOW", "REVIEW", "BLOCK"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["transaction_id"].startswith("TXN-")


def test_prediction_saved_to_database(client):
    response = client.post("/predict", json=payment_payload())
    assert response.status_code == 200
    transaction_id = response.json()["transaction_id"]

    rows = db.get_transactions(limit=10)
    assert not rows.empty
    assert transaction_id in rows["transaction_id"].values
