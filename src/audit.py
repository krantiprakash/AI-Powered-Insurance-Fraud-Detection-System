"""
audit.py
--------
Audit logging for every claim decision made by the system.

Every time a claim is scored, one record is appended to the audit CSV
(default: logs/audit_log.csv — override via AUDIT_LOG_PATH or INSURANCE_AUDIT_LOG).

Each record contains:
  - Timestamp
  - Claim snapshot (key fields)
  - ML results (fraud probability, decision, anomaly score)
  - NLP results (risk score, flags, keywords)
  - GenAI summary (truncated)
  - Reviewer action (if human reviewed)
  - Final outcome

Usage (from code):
    from audit import log_decision, log_reviewer_action, load_audit_log
    log_decision(claim, ml_result, nlp_result, genai_result)
"""

import os
import csv
import json
import time
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR  = os.path.join(BASE_DIR, "logs")

# Manual override: path relative to project root (e.g. "data/ui_sample_audit_log.csv"), or "" to ignore.
# Takes precedence over INSURANCE_AUDIT_LOG when non-empty.
AUDIT_LOG_PATH = "data/ui_sample_audit_log.csv"

# Optional: INSURANCE_AUDIT_LOG=/path/to/file.csv — use when logs/audit_log.csv is locked (e.g. open in Excel).
_env_log = os.environ.get("INSURANCE_AUDIT_LOG", "").strip()
if AUDIT_LOG_PATH.strip():
    p = AUDIT_LOG_PATH.strip()
    LOG_FILE = p if os.path.isabs(p) else os.path.join(BASE_DIR, p)
elif _env_log:
    LOG_FILE = _env_log if os.path.isabs(_env_log) else os.path.join(BASE_DIR, _env_log)
else:
    LOG_FILE = os.path.join(LOGS_DIR, "audit_log.csv")

os.makedirs(LOGS_DIR, exist_ok=True)
_log_dir = os.path.dirname(os.path.abspath(LOG_FILE))
if _log_dir:
    os.makedirs(_log_dir, exist_ok=True)


def _append_row_with_retry(path: str, record: dict, *, attempts: int = 10, delay_sec: float = 0.2) -> None:
    """Append one CSV row; retry on Windows PermissionError when the file is temporarily locked."""
    last_err: OSError | None = None
    for _ in range(attempts):
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writerow(record)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(delay_sec)
    msg = (
        f"Cannot write to audit log: {path}\n"
        "Another program likely has this file open for exclusive access.\n"
        "On Windows, close Excel, LibreOffice, or any preview that opened audit_log.csv, "
        "then try again.\n"
        "Or set INSURANCE_AUDIT_LOG to a different path (e.g. logs/audit_log_dev.csv)."
    )
    raise PermissionError(msg) from last_err

# ── CSV columns ───────────────────────────────────────────────────────────────

COLUMNS = [
    "timestamp",
    "claim_id",
    # Key claim fields
    "age", "sex", "make", "vehicle_category", "policy_type",
    "fault", "accident_area", "police_report", "witness_present",
    "agent_type", "past_claims", "address_change",
    "vehicle_price", "deductible", "driver_rating", "year",
    # ML results
    "fraud_probability", "decision", "anomaly_score",
    "top_shap_feature_1", "top_shap_feature_2", "top_shap_feature_3",
    # NLP results
    "nlp_risk_score", "nlp_valid", "nlp_flag_count",
    "nlp_flags", "nlp_keywords",
    # GenAI
    "genai_source", "genai_summary_snippet",
    # Human review (filled in later via log_reviewer_action)
    "reviewer_action", "reviewer_notes", "final_outcome",
    "reviewed_at"
]


def _ensure_header():
    """Write CSV header if the file does not exist yet."""
    if not os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
        except PermissionError as e:
            raise PermissionError(
                f"Cannot create audit log at {LOG_FILE}. "
                "Check folder permissions, or set INSURANCE_AUDIT_LOG to a writable path."
            ) from e


def _make_claim_id(claim: dict, timestamp: str) -> str:
    """Generate a unique claim ID from key fields + timestamp."""
    age  = claim.get("Age", "0")
    make = claim.get("Make", "X").replace(" ", "")[:3].upper()
    ts   = timestamp.replace("-", "").replace(":", "").replace(" ", "")[:12]
    return f"CLM-{make}{age}-{ts}"


# ── Log a new decision ────────────────────────────────────────────────────────

def log_decision(claim: dict, ml_result: dict,
                 nlp_result: dict, genai_result: dict) -> str:
    """
    Append one record to the audit log for a new claim decision.

    Returns
    -------
    str — the generated claim_id for this record
    """
    _ensure_header()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    claim_id  = _make_claim_id(claim, timestamp)

    # SHAP top 3 feature names
    shap = ml_result.get("top_shap_features", [])
    shap_1 = f"{shap[0][0]}={shap[0][1]:+.4f}" if len(shap) > 0 else ""
    shap_2 = f"{shap[1][0]}={shap[1][1]:+.4f}" if len(shap) > 1 else ""
    shap_3 = f"{shap[2][0]}={shap[2][1]:+.4f}" if len(shap) > 2 else ""

    # NLP fields
    consistency = nlp_result.get("consistency", {})
    risk_kw     = nlp_result.get("risk_keywords", {})
    flags_str   = " | ".join(consistency.get("flags", []))
    keywords_str = ", ".join(risk_kw.get("detected_keywords", []))

    # GenAI summary — keep first 200 chars to avoid huge CSV cells
    summary_full    = genai_result.get("summary", "")
    summary_snippet = summary_full[:200].replace("\n", " ").strip()

    record = {
        "timestamp":            timestamp,
        "claim_id":             claim_id,
        # Claim fields
        "age":                  claim.get("Age", ""),
        "sex":                  claim.get("Sex", ""),
        "make":                 claim.get("Make", ""),
        "vehicle_category":     claim.get("VehicleCategory", ""),
        "policy_type":          claim.get("PolicyType", ""),
        "fault":                claim.get("Fault", ""),
        "accident_area":        claim.get("AccidentArea", ""),
        "police_report":        claim.get("PoliceReportFiled", ""),
        "witness_present":      claim.get("WitnessPresent", ""),
        "agent_type":           claim.get("AgentType", ""),
        "past_claims":          claim.get("PastNumberOfClaims", ""),
        "address_change":       claim.get("AddressChange_Claim", ""),
        "vehicle_price":        claim.get("VehiclePrice", ""),
        "deductible":           claim.get("Deductible", ""),
        "driver_rating":        claim.get("DriverRating", ""),
        "year":                 claim.get("Year", ""),
        # ML results
        "fraud_probability":    ml_result.get("fraud_probability", ""),
        "decision":             ml_result.get("decision", ""),
        "anomaly_score":        ml_result.get("anomaly_score", ""),
        "top_shap_feature_1":   shap_1,
        "top_shap_feature_2":   shap_2,
        "top_shap_feature_3":   shap_3,
        # NLP results
        "nlp_risk_score":       nlp_result.get("nlp_risk_score", ""),
        "nlp_valid":            nlp_result.get("is_valid", ""),
        "nlp_flag_count":       consistency.get("flag_count", 0),
        "nlp_flags":            flags_str,
        "nlp_keywords":         keywords_str,
        # GenAI
        "genai_source":         genai_result.get("source", ""),
        "genai_summary_snippet": summary_snippet,
        # Human review — blank until reviewed
        "reviewer_action":      "",
        "reviewer_notes":       "",
        "final_outcome":        ml_result.get("decision", ""),
        "reviewed_at":          ""
    }

    _append_row_with_retry(LOG_FILE, record)

    return claim_id


# ── Update with reviewer action ───────────────────────────────────────────────

def log_reviewer_action(claim_id: str, action: str,
                        notes: str = "", final_outcome: str = "") -> bool:
    """
    Update an existing audit record with the human reviewer's decision.

    Parameters
    ----------
    claim_id       : str — claim_id returned by log_decision()
    action         : str — e.g. 'APPROVED', 'REJECTED', 'ESCALATED'
    notes          : str — free-text reviewer notes
    final_outcome  : str — final status after review

    Returns
    -------
    bool — True if record was found and updated, False otherwise
    """
    if not os.path.exists(LOG_FILE):
        return False

    updated     = False
    reviewed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows        = []

    with open(LOG_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["claim_id"] == claim_id:
                row["reviewer_action"] = action
                row["reviewer_notes"]  = notes
                row["final_outcome"]   = final_outcome or action
                row["reviewed_at"]     = reviewed_at
                updated = True
            rows.append(row)

    if updated:
        last_err: OSError | None = None
        for _ in range(10):
            try:
                with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)
                return True
            except PermissionError as e:
                last_err = e
                time.sleep(0.2)
        raise PermissionError(
            f"Cannot save reviewer update to {LOG_FILE}. Close any app that has this CSV open."
        ) from last_err

    return updated


# ── Load audit log ────────────────────────────────────────────────────────────

def load_audit_log() -> list:
    """
    Load all audit records as a list of dicts.
    Returns empty list if log file does not exist.
    """
    if not os.path.exists(LOG_FILE):
        return []

    with open(LOG_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ── Summary stats ─────────────────────────────────────────────────────────────

def audit_summary() -> dict:
    """
    Return high-level stats from the audit log.
    Useful for the dashboard in app.py.
    """
    records = load_audit_log()
    if not records:
        return {
            "total_claims":    0,
            "approved":        0,
            "human_review":    0,
            "rejected":        0,
            "reviewed_by_human": 0,
            "avg_fraud_prob":  0.0
        }

    total      = len(records)
    approved   = sum(1 for r in records if r["decision"] == "APPROVED")
    review     = sum(1 for r in records if r["decision"] == "HUMAN_REVIEW")
    rejected   = sum(1 for r in records if r["decision"] == "REJECTED")
    reviewed   = sum(1 for r in records if r["reviewer_action"] != "")

    probs = []
    for r in records:
        try:
            probs.append(float(r["fraud_probability"]))
        except (ValueError, TypeError):
            pass

    avg_prob = round(sum(probs) / len(probs), 4) if probs else 0.0

    return {
        "total_claims":      total,
        "approved":          approved,
        "human_review":      review,
        "rejected":          rejected,
        "reviewed_by_human": reviewed,
        "avg_fraud_prob":    avg_prob
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from predict import load_artifacts, predict_claim
    from nlp     import run_nlp_pipeline
    from genai   import generate_summary

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

    print("Running full pipeline...")
    artifacts  = load_artifacts()
    ml_result  = predict_claim(sample_claim, artifacts)
    nlp_result = run_nlp_pipeline(sample_claim)
    genai_result = generate_summary(ml_result, nlp_result, sample_claim)

    print("Logging decision to audit log...")
    claim_id = log_decision(sample_claim, ml_result, nlp_result, genai_result)
    print(f"  Logged with claim_id: {claim_id}")

    print("\nSimulating reviewer action...")
    success = log_reviewer_action(
        claim_id,
        action="REJECTED",
        notes="High fraud probability confirmed. No police report or witness.",
        final_outcome="REJECTED"
    )
    print(f"  Reviewer action logged: {success}")

    print("\nAudit summary:")
    summary = audit_summary()
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")

    print(f"\nLog file: {LOG_FILE}")
