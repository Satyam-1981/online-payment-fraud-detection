# Online Payment Fraud Detection

A simple end-to-end real-time payment fraud detection project using PaySim, machine learning, FastAPI, SQLite and Streamlit.

## Features

- PaySim data cleaning and feature engineering
- Logistic Regression, Decision Tree, Random Forest and XGBoost comparison
- Model selection using F1 score
- Precision, Recall, Accuracy and ROC-AUC evaluation
- Real-time fraud probability through REST API
- Risk decision: ALLOW, REVIEW or BLOCK
- SQLite transaction storage
- Streamlit dashboard and transaction history
- Laptop-friendly training with 2 CPU threads
- KNN removed because it is inefficient for this large dataset

## Project structure

```text
Online-Payment-Fraud-Detection/
├── api.py
├── app/
│   └── streamlit_app.py
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── db.py
├── models/          # created after training
├── outputs/         # created after training
├── data/            # SQLite database is created here
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

The first run downloads the PaySim dataset. Training the full dataset can take time on a laptop.

## Start API

```bash
uvicorn api:app --reload
```

API docs: `http://127.0.0.1:8000/docs`

## Start dashboard

```bash
streamlit run app/streamlit_app.py
```

## API endpoints

- `GET /health` - service and model status
- `POST /predict` - real-time payment prediction
- `GET /transactions` - recent stored transactions
- `GET /summary` - transaction statistics

## Architecture

```text
Streamlit UI
     ↓ HTTP
 FastAPI
     ↓
 ML Model
     ↓
Fraud Probability
     ↓
ALLOW / REVIEW / BLOCK
     ↓
  SQLite
```

## Interview points

- Fraud data is imbalanced, so F1 and Recall are more useful than accuracy alone.
- KNN was removed because distance-based prediction is not practical for millions of transactions.
- FastAPI separates model serving from the UI.
- SQLite keeps the project simple while demonstrating persistent storage.
- The system is a portfolio/academic project, not a production payment authorization service.


## Testing

After training the model, run:

```bash
pytest -q
```

The test suite checks:
- API health endpoint
- Valid real-time prediction
- Prediction saved to SQLite

## Class imbalance

PaySim is highly imbalanced, so the training pipeline uses class-aware learning instead of relying on accuracy alone:
- Logistic Regression: `class_weight="balanced"`
- Decision Tree: `class_weight="balanced"`
- Random Forest: `class_weight="balanced"`
- XGBoost: `scale_pos_weight` based on the training class ratio

Models are compared using Accuracy, Precision, Recall, F1 and ROC-AUC. The best model is selected by F1-score.
