from pathlib import Path
import json
import time
import joblib
import kagglehub
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocessing import TARGET_COL, clean_raw_data, Preprocessor

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
DATASET = "ealaxi/paysim1"
CSV_FILE = "PS_20174392719_1491204439457_log.csv"
RANDOM_STATE = 42
CPU_THREADS = 2


def load_dataset():
    path = Path(kagglehub.dataset_download(DATASET, path=CSV_FILE))
    if path.is_dir():
        path /= CSV_FILE
    return pd.read_csv(path)


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob),
    }


def main():
    start = time.time()
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading PaySim dataset...")
    raw = load_dataset()
    cleaned = clean_raw_data(raw)

    X = cleaned.drop(columns=TARGET_COL)
    y = cleaned[TARGET_COL]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = Preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    legit_count = int((y_train == 0).sum())
    fraud_count = int((y_train == 1).sum())
    fraud_weight = legit_count / max(1, fraud_count)
    print(f"Training class balance: legitimate={legit_count:,}, fraud={fraud_count:,}, fraud_weight={fraud_weight:.2f}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=14, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=CPU_THREADS),
        "XGBoost": XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.08, scale_pos_weight=fraud_weight, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=CPU_THREADS),
    }

    trained = {}
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model_start = time.time()
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        results[name]["Training Time (s)"] = round(time.time() - model_start, 2)
        trained[name] = model

    results_df = pd.DataFrame(results).T.sort_values("F1", ascending=False)
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))
    results_df.index.name = "Model"
    results_df.to_csv(OUTPUT_DIR / "model_results.csv")

    best_name = results_df.index[0]
    best_model = trained[best_name]
    joblib.dump(best_model, MODEL_DIR / "fraud_model.pkl")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessing.pkl")

    matrix = confusion_matrix(y_test, best_model.predict(X_test))
    pd.DataFrame(
        matrix,
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"],
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    best = results_df.loc[best_name]
    metadata = {
        "model": best_name,
        "dataset": "PaySim",
        "dataset_source": "Kaggle: ealaxi/paysim1",
        "dataset_rows": int(len(raw)),
        "fraud_count": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "features": ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "errorBalanceOrig", "errorBalanceDest"],
        "transaction_types": preprocessor.fitted_types_,
        "risk_threshold": 0.50,
        "test_accuracy": float(best["Accuracy"]),
        "test_precision": float(best["Precision"]),
        "test_recall": float(best["Recall"]),
        "test_f1": float(best["F1"]),
        "test_roc_auc": float(best["ROC-AUC"]),
        "training_time_seconds": round(time.time() - start, 2),
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nModel comparison:")
    print(results_df.round(4))
    print(f"\nBest model: {best_name}")


if __name__ == "__main__":
    main()
