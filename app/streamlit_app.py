from pathlib import Path
import json
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
API_URL = "http://127.0.0.1:8000"
OUTPUT_DIR = ROOT / "outputs"
MODEL_METADATA = ROOT / "models" / "model_metadata.json"

st.set_page_config(page_title="FraudGuard", page_icon="🛡️", layout="wide")

if not MODEL_METADATA.exists():
    st.error("Model is not trained yet. Run `python src/train.py` first.")
    st.stop()

metadata = json.loads(MODEL_METADATA.read_text(encoding="utf-8"))

with st.sidebar:
    st.title("🛡️ FraudGuard")
    st.caption("Real-Time Online Payment Fraud Detection")
    page = st.radio("Navigation", ["Dashboard", "Predict", "Transactions", "Model"], label_visibility="collapsed")
    st.divider()
    st.caption(f"API: {API_URL}")
    st.caption(f"Model: {metadata['model']}")


def api_get(path):
    response = requests.get(API_URL + path, timeout=5)
    response.raise_for_status()
    return response.json()


def metric_cards(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


if page == "Dashboard":
    st.title("🛡️ FraudGuard Dashboard")
    st.write("Real-time ML fraud scoring with FastAPI and SQLite transaction storage.")
    try:
        summary = api_get("/summary")
        metric_cards([
            ("Transactions", f"{summary['total']:,}"),
            ("Fraud flagged", f"{summary['fraud']:,}"),
            ("Average risk", f"{summary['avg_risk'] * 100:.2f}%"),
            ("Model", metadata["model"]),
        ])
        txns = api_get("/transactions?limit=20")
        if txns:
            st.subheader("Latest transactions")
            st.dataframe(pd.DataFrame(txns), use_container_width=True, hide_index=True)
        else:
            st.info("No transactions yet. Use Predict to create a real-time prediction.")
    except Exception as error:
        st.error(f"API is not running. Start it with `uvicorn api:app --reload`. Details: {error}")

elif page == "Predict":
    st.title("⚡ Real-Time Payment Prediction")
    st.caption("The payment is sent to the FastAPI model service and saved to SQLite automatically.")
    types = metadata["transaction_types"]
    with st.form("payment"):
        c1, c2 = st.columns(2)
        with c1:
            txn_type = st.selectbox("Transaction type", types)
            amount = st.number_input("Amount", min_value=0.0, value=5000.0, step=100.0)
            old_org = st.number_input("Sender balance before", min_value=0.0, value=25000.0, step=100.0)
            new_org = st.number_input("Sender balance after", min_value=0.0, value=20000.0, step=100.0)
        with c2:
            old_dest = st.number_input("Receiver balance before", min_value=0.0, value=15000.0, step=100.0)
            new_dest = st.number_input("Receiver balance after", min_value=0.0, value=20000.0, step=100.0)
        submit = st.form_submit_button("Analyze Payment", use_container_width=True)

    if submit:
        payload = {
            "type": txn_type, "amount": amount, "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
            "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
        }
        try:
            response = requests.post(API_URL + "/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            probability = result["fraud_probability"]
            if result["decision"] == "BLOCK":
                st.error("🚨 HIGH RISK — BLOCK PAYMENT")
            elif result["decision"] == "REVIEW":
                st.warning("⚠️ MEDIUM RISK — SEND FOR REVIEW")
            else:
                st.success("✅ LOW RISK — ALLOW PAYMENT")
            metric_cards([
                ("Transaction ID", result["transaction_id"]),
                ("Fraud probability", f"{probability * 100:.2f}%"),
                ("Decision", result["decision"]),
                ("Model", result["model"]),
            ])
            st.progress(min(probability, 1.0), text=f"Fraud risk: {probability * 100:.2f}%")
        except Exception as error:
            st.error(f"Prediction failed. Is FastAPI running? {error}")

elif page == "Transactions":
    st.title("📜 Transaction History")
    try:
        data = api_get("/transactions?limit=500")
        if not data:
            st.info("No transactions found.")
        else:
            df = pd.DataFrame(data)
            metric_cards([
                ("Total", f"{len(df):,}"),
                ("Blocked", f"{(df['decision'] == 'BLOCK').sum():,}"),
                ("Review", f"{(df['decision'] == 'REVIEW').sum():,}"),
                ("Allowed", f"{(df['decision'] == 'ALLOW').sum():,}"),
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download CSV", df.to_csv(index=False), "transactions.csv", "text/csv")
    except Exception as error:
        st.error(f"Could not load transactions: {error}")

elif page == "Model":
    st.title("🤖 Model Performance")
    results_file = OUTPUT_DIR / "model_results.csv"
    if results_file.exists():
        results = pd.read_csv(results_file)
        st.dataframe(results.round(4), use_container_width=True, hide_index=True)
        st.subheader("F1 Score")
        st.bar_chart(results.set_index("Model")["F1"])
    metric_cards([
        ("Accuracy", f"{metadata['test_accuracy']:.4f}"),
        ("Precision", f"{metadata['test_precision']:.4f}"),
        ("Recall", f"{metadata['test_recall']:.4f}"),
        ("F1", f"{metadata['test_f1']:.4f}"),
        ("ROC-AUC", f"{metadata['test_roc_auc']:.4f}"),
    ])
    st.info("F1 is emphasized because fraud data is highly imbalanced. Recall is especially important because missed fraud can cause financial loss.")
    cm_file = OUTPUT_DIR / "confusion_matrix.csv"
    if cm_file.exists():
        st.subheader("Confusion Matrix")
        st.dataframe(pd.read_csv(cm_file, index_col=0), use_container_width=True)

