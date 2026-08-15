from pathlib import Path
import sqlite3
from datetime import datetime, timezone
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_FILE = ROOT / "data" / "fraud_detection.db"


def get_connection():
    DB_FILE.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                amount REAL NOT NULL,
                oldbalance_org REAL NOT NULL,
                newbalance_orig REAL NOT NULL,
                oldbalance_dest REAL NOT NULL,
                newbalance_dest REAL NOT NULL,
                error_balance_orig REAL NOT NULL,
                error_balance_dest REAL NOT NULL,
                fraud_probability REAL NOT NULL,
                prediction INTEGER NOT NULL,
                decision TEXT NOT NULL
            )
        """)


def save_transaction(transaction):
    init_db()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO transactions (
                transaction_id, timestamp, type, amount, oldbalance_org, newbalance_orig,
                oldbalance_dest, newbalance_dest, error_balance_orig, error_balance_dest,
                fraud_probability, prediction, decision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            transaction["transaction_id"], transaction["timestamp"], transaction["type"], transaction["amount"],
            transaction["oldbalanceOrg"], transaction["newbalanceOrig"], transaction["oldbalanceDest"], transaction["newbalanceDest"],
            transaction["errorBalanceOrig"], transaction["errorBalanceDest"], transaction["fraud_probability"],
            transaction["prediction"], transaction["decision"]
        ))


def get_transactions(limit=200):
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM transactions ORDER BY id DESC LIMIT ?", conn, params=(limit,))


def get_summary():
    init_db()
    with get_connection() as conn:
        row = conn.execute("""
            SELECT COUNT(*) AS total, COALESCE(SUM(prediction), 0) AS fraud,
                   COALESCE(AVG(fraud_probability), 0) AS avg_risk
            FROM transactions
        """).fetchone()
    return {"total": row["total"], "fraud": row["fraud"], "avg_risk": row["avg_risk"]}

