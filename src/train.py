"""
train.py
--------
Train the full fraud detection pipeline:
  1. Load and preprocess data
  2. Train/val/test split + SMOTE
  3. Fit Isolation Forest → add anomaly_score feature
  4. Train LightGBM
  5. Tune threshold
  6. Evaluate on test set
  7. Save all artifacts to models/

Run:
    python src/train.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, f1_score
)
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

from preprocess import (
    clean, get_ohe_columns, apply_ohe,
    NUMERIC_FEATURES, ORDINAL_COLS, ORDINAL_ORDERS, TARGET
)

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "fraud_oracle.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Load & clean ──────────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = clean(df)
    print(f"  Shape after cleaning: {df.shape}")
    print(f"  Fraud rate: {df[TARGET].mean()*100:.1f}%")
    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df):
    print("\nPreprocessing...")

    # Ordinal encoding
    ordinal_categories = [ORDINAL_ORDERS[col] for col in ORDINAL_COLS]
    ord_encoder = OrdinalEncoder(
        categories=ordinal_categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    df[ORDINAL_COLS] = ord_encoder.fit_transform(df[ORDINAL_COLS])

    # OHE — record column names for inference alignment
    ohe_columns = get_ohe_columns(df)
    df_ohe = apply_ohe(df, ohe_columns)

    # Build feature matrix
    X = pd.concat([
        df[NUMERIC_FEATURES].reset_index(drop=True),
        df[ORDINAL_COLS].reset_index(drop=True),
        df_ohe.reset_index(drop=True)
    ], axis=1)
    y = df[TARGET].reset_index(drop=True)

    print(f"  Feature matrix: {X.shape}")
    return X, y, ord_encoder, ohe_columns


# ── Split + Scale + SMOTE ─────────────────────────────────────────────────────

def split_scale_smote(X, y):
    print("\nSplitting, scaling, SMOTE...")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # Scale numeric columns — fit on train only
    scaler = StandardScaler()
    X_train_s = X_train.copy()
    X_val_s   = X_val.copy()
    X_test_s  = X_test.copy()

    X_train_s[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
    X_val_s[NUMERIC_FEATURES]   = scaler.transform(X_val[NUMERIC_FEATURES])
    X_test_s[NUMERIC_FEATURES]  = scaler.transform(X_test[NUMERIC_FEATURES])

    # SMOTE on train only
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_s, y_train)

    print(f"  Train (after SMOTE): {X_train_sm.shape} | Fraud: {y_train_sm.sum()}")
    print(f"  Val               : {X_val_s.shape}   | Fraud: {y_val.sum()}")
    print(f"  Test              : {X_test_s.shape}   | Fraud: {y_test.sum()}")

    return X_train_s, X_train_sm, X_val_s, X_test_s, y_train_sm, y_val, y_test, scaler


# ── Isolation Forest ──────────────────────────────────────────────────────────

def train_isolation_forest(X_train_s, X_train_sm, X_val_s, X_test_s):
    print("\nTraining Isolation Forest...")

    iso = IsolationForest(n_estimators=100, contamination=0.06, random_state=42)
    iso.fit(X_train_s)  # fit on real (pre-SMOTE) training data

    # Add anomaly score as extra feature (negated: higher = more anomalous)
    train_anomaly = -iso.score_samples(X_train_sm)
    val_anomaly   = -iso.score_samples(X_val_s)
    test_anomaly  = -iso.score_samples(X_test_s)

    X_train_final = np.column_stack([X_train_sm, train_anomaly])
    X_val_final   = np.column_stack([X_val_s,   val_anomaly])
    X_test_final  = np.column_stack([X_test_s,  test_anomaly])

    feature_names = list(X_train_sm.columns) + ["anomaly_score"]

    print(f"  Feature matrix with anomaly score: {X_train_final.shape}")
    return X_train_final, X_val_final, X_test_final, feature_names, iso


# ── LightGBM ──────────────────────────────────────────────────────────────────

def train_lightgbm(X_train, y_train, X_val, y_val):
    print("\nTraining LightGBM...")

    model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    val_proba = model.predict_proba(X_val)[:, 1]
    auc_pr  = average_precision_score(y_val, val_proba)
    auc_roc = roc_auc_score(y_val, val_proba)
    print(f"  Val AUC-PR : {auc_pr:.4f}")
    print(f"  Val AUC-ROC: {auc_roc:.4f}")

    return model, val_proba


# ── Threshold tuning ──────────────────────────────────────────────────────────

def tune_threshold(y_val, val_proba):
    print("\nTuning threshold...")

    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores  = [f1_score(y_val, (val_proba >= t).astype(int)) for t in thresholds]
    best_thresh = float(thresholds[np.argmax(f1_scores)])

    low_thresh  = round(max(best_thresh - 0.15, 0.05), 2)
    high_thresh = round(min(best_thresh + 0.15, 0.90), 2)

    print(f"  Optimal binary threshold : {best_thresh:.2f}")
    print(f"  3-class thresholds       : low={low_thresh}, high={high_thresh}")
    return best_thresh, low_thresh, high_thresh


# ── Final evaluation ──────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, threshold):
    print("\nFinal test set evaluation...")

    test_proba = model.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= threshold).astype(int)

    auc_roc = roc_auc_score(y_test, test_proba)
    auc_pr  = average_precision_score(y_test, test_proba)

    print(f"  AUC-ROC : {auc_roc:.4f}")
    print(f"  AUC-PR  : {auc_pr:.4f}")
    print()
    print(classification_report(y_test, test_preds,
                                 target_names=["Not Fraud", "Fraud"]))


# ── Save artifacts ────────────────────────────────────────────────────────────

def save_artifacts(model, iso, scaler, ord_encoder,
                   feature_names, ohe_columns,
                   best_thresh, low_thresh, high_thresh):
    print("\nSaving artifacts...")

    joblib.dump(model,       os.path.join(MODELS_DIR, "fraud_model.pkl"))
    joblib.dump(iso,         os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    joblib.dump(scaler,      os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(ord_encoder, os.path.join(MODELS_DIR, "ordinal_encoder.pkl"))

    config = {
        "low_threshold":    low_thresh,
        "high_threshold":   high_thresh,
        "binary_threshold": best_thresh,
        "best_model":       "LightGBM",
        "numeric_features": NUMERIC_FEATURES,
        "ordinal_cols":     ORDINAL_COLS,
        "nominal_cols":     ["Month", "DayOfWeek", "Make", "AccidentArea",
                             "DayOfWeekClaimed", "MonthClaimed", "Sex",
                             "MaritalStatus", "Fault", "PolicyType",
                             "VehicleCategory", "PoliceReportFiled",
                             "WitnessPresent", "AgentType", "BasePolicy"],
        "ohe_columns":      ohe_columns,
        "feature_names":    feature_names
    }
    with open(os.path.join(MODELS_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    for fname in os.listdir(MODELS_DIR):
        fpath = os.path.join(MODELS_DIR, fname)
        print(f"  {fname:<35} ({os.path.getsize(fpath):,} bytes)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    X, y, ord_encoder, ohe_columns = preprocess(df)
    X_train_s, X_train_sm, X_val_s, X_test_s, y_train_sm, y_val, y_test, scaler = split_scale_smote(X, y)
    X_train_f, X_val_f, X_test_f, feature_names, iso = train_isolation_forest(
        X_train_s, X_train_sm, X_val_s, X_test_s
    )
    model, val_proba = train_lightgbm(X_train_f, y_train_sm, X_val_f, y_val)
    best_thresh, low_thresh, high_thresh = tune_threshold(y_val, val_proba)
    evaluate(model, X_test_f, y_test, best_thresh)
    save_artifacts(model, iso, scaler, ord_encoder,
                   feature_names, ohe_columns,
                   best_thresh, low_thresh, high_thresh)
    print("\nTraining complete.")
