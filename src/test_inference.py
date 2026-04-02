"""
test_inference.py
-----------------
Finds real examples from fraud_oracle.csv covering all three decisions:
  APPROVED, HUMAN_REVIEW, REJECTED

Strategy (fast):
  1. Quick-scan rows using ML probability only (no SHAP, no NLP, no GenAI)
  2. Pick one row per bucket closest to bucket midpoint
  3. Run full pipeline (SHAP + NLP + GenAI + audit) only for those 3 rows

Run:
    python test_inference.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import (
    clean, get_ohe_columns, apply_ohe, apply_ordinal_encoding,
    build_feature_matrix, ORDINAL_COLS, NUMERIC_FEATURES, NOMINAL_COLS,
    ORDINAL_ORDERS, TARGET
)
from predict import load_artifacts, predict_claim
from nlp     import run_nlp_pipeline
from genai   import generate_summary
from audit   import log_decision

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "fraud_oracle.csv")

# ── Load artifacts ─────────────────────────────────────────────────────────────

print("Loading ML artifacts...")
artifacts = load_artifacts()
config    = artifacts["config"]
model     = artifacts["model"]
iso       = artifacts["iso_forest"]
scaler    = artifacts["scaler"]
ord_enc   = artifacts["ord_encoder"]
low_t     = config["low_threshold"]
high_t    = config["high_threshold"]
ohe_cols  = config["ohe_columns"]

print(f"Thresholds: APPROVED < {low_t}  |  HUMAN_REVIEW {low_t}-{high_t}  |  REJECTED >= {high_t}")

# ── Quick score function (no SHAP) ─────────────────────────────────────────────

def quick_score(claim: dict) -> tuple:
    """Return (fraud_probability, decision) without SHAP — fast scan."""
    df_c = pd.DataFrame([claim])
    X    = build_feature_matrix(df_c, ord_enc, scaler, ohe_cols)
    anom = float(-iso.score_samples(X)[0])
    fv   = np.append(X.values[0], anom)
    prob = float(model.predict_proba(fv.reshape(1, -1))[0][1])
    if prob < low_t:
        decision = "APPROVED"
    elif prob < high_t:
        decision = "HUMAN_REVIEW"
    else:
        decision = "REJECTED"
    return prob, decision

# ── Load data ──────────────────────────────────────────────────────────────────

print("\nLoading data...")
df_raw = pd.read_csv(DATA_PATH)
df     = clean(df_raw)
CLAIM_COLS = NUMERIC_FEATURES + ORDINAL_COLS + NOMINAL_COLS

# ── Scan rows, collect candidates per bucket ───────────────────────────────────

print("Scanning rows to find candidates per bucket...")
print("(scanning up to 500 rows — should be fast)\n")

# Store: bucket -> list of (row_idx, prob, true_label)
candidates = {"APPROVED": [], "HUMAN_REVIEW": [], "REJECTED": []}

for idx in range(min(500, len(df))):
    row   = df.iloc[idx]
    claim = {col: row[col] for col in CLAIM_COLS if col in row.index}
    try:
        prob, decision = quick_score(claim)
        true_label = int(df_raw.iloc[idx][TARGET])
        candidates[decision].append((idx, prob, true_label, claim))
    except Exception:
        continue

    if idx % 50 == 0:
        counts = {k: len(v) for k, v in candidates.items()}
        print(f"  Scanned {idx+1:>3} rows | "
              f"APPROVED: {counts['APPROVED']}  "
              f"HUMAN_REVIEW: {counts['HUMAN_REVIEW']}  "
              f"REJECTED: {counts['REJECTED']}")

    # Stop early if we have enough candidates in each bucket
    if all(len(v) >= 3 for v in candidates.values()):
        print(f"  All buckets filled at row {idx}.")
        break

print()

# ── Pick most representative from each bucket ──────────────────────────────────

midpoints = {
    "APPROVED":     low_t / 2,
    "HUMAN_REVIEW": (low_t + high_t) / 2,
    "REJECTED":     min(high_t + 0.30, 0.95)
}

selected = {}
for bucket, cands in candidates.items():
    if not cands:
        print(f"  WARNING: No candidates found for {bucket}")
        continue
    mid = midpoints[bucket]
    best = min(cands, key=lambda x: abs(x[1] - mid))
    selected[bucket] = best
    label_str = "FRAUD" if best[2] == 1 else "Not Fraud"
    print(f"  Selected for {bucket:<15}: row {best[0]}  prob={best[1]:.4f}  true_label={label_str}")

# ── Full pipeline for each selected row ────────────────────────────────────────

SEP = "=" * 65

def run_full_report(bucket: str, row_idx: int, claim: dict, true_label: int):
    label_str = "FRAUD" if true_label == 1 else "Not Fraud"

    print(f"\n{SEP}")
    print(f"  DECISION: {bucket}   |   True Label: {label_str}")
    print(SEP)

    ml_result    = predict_claim(claim, artifacts)      # full run with SHAP
    nlp_result   = run_nlp_pipeline(claim)
    genai_result = generate_summary(ml_result, nlp_result, claim)
    claim_id     = log_decision(claim, ml_result, nlp_result, genai_result)

    prob     = ml_result["fraud_probability"]
    decision = ml_result["decision"]
    anomaly  = ml_result["anomaly_score"]
    shap     = ml_result["top_shap_features"]

    print(f"  Claim ID          : {claim_id}")
    print(f"  Fraud Probability : {prob:.4f}  ({prob*100:.1f}%)")
    print(f"  Decision          : {decision}")
    print(f"  Anomaly Score     : {anomaly:.4f}")
    print()

    print("  Key Claim Facts:")
    key_fields = [
        ("Age", "Age"), ("Sex", "Sex"), ("Fault", "Fault"),
        ("PolicyType", "Policy Type"), ("VehicleCategory", "Vehicle Cat"),
        ("VehiclePrice", "Vehicle Price"), ("PoliceReportFiled", "Police Report"),
        ("WitnessPresent", "Witness"), ("AgentType", "Agent Type"),
        ("PastNumberOfClaims", "Past Claims"), ("BasePolicy", "Base Policy"),
        ("AddressChange_Claim", "Addr Change"), ("DriverRating", "Driver Rating"),
    ]
    for col, label in key_fields:
        print(f"    {label:<18}: {claim.get(col, 'N/A')}")

    print()
    print("  Top SHAP Features:")
    for feat, val in shap:
        direction = "+fraud" if val > 0 else "-fraud"
        print(f"    {feat:<40} {val:+.4f}  ({direction})")

    print()
    print(f"  NLP Risk Score    : {nlp_result['nlp_risk_score']}")
    print(f"  NLP Valid         : {nlp_result['is_valid']}")
    print()
    print(f"  GenAI Source      : {genai_result['source']} ({genai_result['model']})")
    print("  GenAI Summary (first 5 lines):")
    count = 0
    for line in genai_result["summary"].strip().split("\n"):
        if line.strip():
            print(f"    {line.strip()}")
            count += 1
            if count >= 5:
                break

    print(f"\n  Logged with ID: {claim_id}")


print(f"\nRunning full pipeline for 3 representative claims...")

for bucket in ["APPROVED", "HUMAN_REVIEW", "REJECTED"]:
    if bucket in selected:
        row_idx, prob, true_label, claim = selected[bucket]
        run_full_report(bucket, row_idx, claim, true_label)
    else:
        print(f"\n  {bucket}: no candidate found — try increasing scan range.")

print(f"\n{SEP}")
print("All claims logged to logs/audit_log.csv")
print("Open Streamlit app -> Audit Dashboard to see all entries.")
