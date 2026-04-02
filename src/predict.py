"""
predict.py
----------
Load trained artifacts and score a single insurance claim.

Returns a dict with:
  - fraud_probability  : float
  - decision           : 'APPROVED' | 'HUMAN_REVIEW' | 'REJECTED'
  - top_shap_features  : list of (feature, shap_value) tuples
  - anomaly_score      : float

Usage (from code):
    from predict import load_artifacts, predict_claim
    artifacts = load_artifacts()
    result = predict_claim(claim_dict, artifacts)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap

from preprocess import build_feature_matrix, NUMERIC_FEATURES, ORDINAL_COLS

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ── Load artifacts ────────────────────────────────────────────────────────────

def load_artifacts() -> dict:
    """Load all saved model artifacts. Call once at app startup."""
    with open(os.path.join(MODELS_DIR, "config.json")) as f:
        config = json.load(f)

    artifacts = {
        "model":       joblib.load(os.path.join(MODELS_DIR, "fraud_model.pkl")),
        "iso_forest":  joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl")),
        "scaler":      joblib.load(os.path.join(MODELS_DIR, "scaler.pkl")),
        "ord_encoder": joblib.load(os.path.join(MODELS_DIR, "ordinal_encoder.pkl")),
        "config":      config,
        "explainer":   None  # lazy-loaded on first use
    }
    return artifacts


def _get_explainer(artifacts: dict):
    """Lazy-load SHAP explainer (slow first call, cached after)."""
    if artifacts["explainer"] is None:
        artifacts["explainer"] = shap.TreeExplainer(artifacts["model"])
    return artifacts["explainer"]


# ── 3-class decision ──────────────────────────────────────────────────────────

def classify(proba: float, low: float, high: float) -> str:
    if proba < low:
        return "APPROVED"
    elif proba < high:
        return "HUMAN_REVIEW"
    else:
        return "REJECTED"


# ── SHAP explanation ──────────────────────────────────────────────────────────

def get_top_shap(artifacts: dict, feature_vector: np.ndarray,
                 feature_names: list, top_n: int = 5) -> list:
    """Return top_n features by absolute SHAP value as (name, value) tuples."""
    explainer = _get_explainer(artifacts)
    raw       = explainer.shap_values(feature_vector.reshape(1, -1))
    # LightGBM returns list of arrays [class0, class1]; take class1 (fraud)
    shap_vals = raw[1][0] if isinstance(raw, list) else raw[0]
    indices   = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    return [(feature_names[i], round(float(shap_vals[i]), 4)) for i in indices]


# ── Main prediction function ──────────────────────────────────────────────────

def predict_claim(claim: dict, artifacts: dict) -> dict:
    """
    Score a single insurance claim.

    Parameters
    ----------
    claim : dict
        Raw claim fields matching the training dataset columns.
    artifacts : dict
        Loaded artifacts from load_artifacts().

    Returns
    -------
    dict with keys:
        fraud_probability, decision, anomaly_score, top_shap_features
    """
    config      = artifacts["config"]
    model       = artifacts["model"]
    iso         = artifacts["iso_forest"]
    scaler      = artifacts["scaler"]
    ord_encoder = artifacts["ord_encoder"]

    # Build feature matrix (clean + ordinal encode + OHE + scale)
    df = pd.DataFrame([claim])
    X  = build_feature_matrix(
        df,
        encoder=ord_encoder,
        scaler=scaler,
        ohe_columns=config["ohe_columns"]
    )

    # Add anomaly score
    anomaly_score  = float(-iso.score_samples(X)[0])
    feature_vector = np.append(X.values[0], anomaly_score)
    feature_names  = config["feature_names"]

    # Fraud probability
    proba = float(model.predict_proba(feature_vector.reshape(1, -1))[0][1])

    # Decision
    decision = classify(proba, config["low_threshold"], config["high_threshold"])

    # SHAP explanation
    top_shap = get_top_shap(artifacts, feature_vector, feature_names)

    return {
        "fraud_probability": round(proba, 4),
        "decision":          decision,
        "anomaly_score":     round(anomaly_score, 4),
        "top_shap_features": top_shap
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sample claim (based on fraud_oracle.csv schema)
    sample_claim = {
        "Month": "Jan", "WeekOfMonth": 3, "DayOfWeek": "Monday",
        "Make": "Honda", "AccidentArea": "Urban", "DayOfWeekClaimed": "Tuesday",
        "MonthClaimed": "Jan", "WeekOfMonthClaimed": 4, "Sex": "Male",
        "MaritalStatus": "Single", "Age": 34, "Fault": "Third Party",
        "PolicyType": "Sport - Collision", "VehicleCategory": "Sport",
        "VehiclePrice": "more than 69000", "Deductible": 400,
        "DriverRating": 4, "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30", "PastNumberOfClaims": "none",
        "AgeOfVehicle": "6 years", "AgeOfPolicyHolder": "31 to 35",
        "PoliceReportFiled": "No", "WitnessPresent": "No",
        "AgentType": "External", "NumberOfSuppliments": "none",
        "AddressChange_Claim": "no change", "NumberOfCars": "1 vehicle",
        "Year": 1994, "BasePolicy": "Collision"
    }

    print("Loading artifacts...")
    artifacts = load_artifacts()

    print("Scoring claim...")
    result = predict_claim(sample_claim, artifacts)

    print(f"\nFraud Probability : {result['fraud_probability']}")
    print(f"Decision          : {result['decision']}")
    print(f"Anomaly Score     : {result['anomaly_score']}")
    print(f"\nTop SHAP Features:")
    for feat, val in result["top_shap_features"]:
        direction = "+fraud" if val > 0 else "-fraud"
        print(f"  {feat:<40} {val:+.4f}  ({direction})")
