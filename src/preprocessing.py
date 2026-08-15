import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET_COL = "isFraud"
RAW_NUMERIC_COLS = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"
]
NUMERIC_COLS = RAW_NUMERIC_COLS + ["errorBalanceOrig", "errorBalanceDest"]
FEATURE_COLS = ["type_encoded"] + NUMERIC_COLS


def clean_raw_data(df):
    df = df.copy().drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=["nameOrig", "nameDest", "step", "isFlaggedFraud"], errors="ignore")

    for col in RAW_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    return df


def add_features(df):
    df = df.copy()
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    return df


class Preprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.fitted_types_ = []

    def fit_transform(self, df):
        df = add_features(df)
        df["type"] = df["type"].astype(str)
        df["type_encoded"] = self.label_encoder.fit_transform(df["type"])
        self.fitted_types_ = list(self.label_encoder.classes_)
        X = df[FEATURE_COLS].copy()
        X[NUMERIC_COLS] = self.scaler.fit_transform(X[NUMERIC_COLS])
        return X

    def transform(self, df):
        df = add_features(df)
        df["type"] = df["type"].astype(str)
        default = self.fitted_types_[0]
        df["type"] = df["type"].where(df["type"].isin(self.fitted_types_), default)
        df["type_encoded"] = self.label_encoder.transform(df["type"])
        X = df[FEATURE_COLS].copy()
        X[NUMERIC_COLS] = self.scaler.transform(X[NUMERIC_COLS])
        return X
