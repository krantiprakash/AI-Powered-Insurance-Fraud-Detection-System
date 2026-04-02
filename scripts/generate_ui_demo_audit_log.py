"""
Generate data/ui_sample_audit_log.csv — synthetic audit rows for all ML decisions
(APPROVED, HUMAN_REVIEW, REJECTED) matching src/audit.py COLUMNS for Streamlit UI demos.

Usage:
    python scripts/generate_ui_demo_audit_log.py

Point the app at this file (optional):
    set INSURANCE_AUDIT_LOG=data/ui_sample_audit_log.csv
"""

import csv
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
from audit import COLUMNS  # noqa: E402

OUT_PATH = os.path.join(BASE, "data", "ui_sample_audit_log.csv")

ROWS = [
    # --- APPROVED (low fraud probability) ---
    {
        "timestamp": "2026-04-01 09:12:00",
        "claim_id": "CLM-DEMO-APPROVED-001",
        "age": "38", "sex": "Male", "make": "Ford", "vehicle_category": "Sedan",
        "policy_type": "Sedan - Liability", "fault": "Policy Holder",
        "accident_area": "Urban", "police_report": "Yes", "witness_present": "Yes",
        "agent_type": "Internal", "past_claims": "none", "address_change": "no change",
        "vehicle_price": "20000 to 29000", "deductible": "400", "driver_rating": "4", "year": "1995",
        "fraud_probability": "0.0312", "decision": "APPROVED", "anomaly_score": "0.42",
        "top_shap_feature_1": "PastNumberOfClaims_none=-1.2",
        "top_shap_feature_2": "PoliceReportFiled_Yes=-0.8",
        "top_shap_feature_3": "WitnessPresent_Yes=-0.4",
        "nlp_risk_score": "0.12", "nlp_valid": "True", "nlp_flag_count": "0",
        "nlp_flags": "", "nlp_keywords": "",
        "genai_source": "demo", "genai_summary_snippet": "Low fraud risk; documentation consistent. Recommend approval.",
        "reviewer_action": "", "reviewer_notes": "", "final_outcome": "APPROVED", "reviewed_at": "",
    },
    {
        "timestamp": "2026-04-01 10:05:22",
        "claim_id": "CLM-DEMO-APPROVED-002",
        "age": "52", "sex": "Female", "make": "Toyota", "vehicle_category": "Utility",
        "policy_type": "Utility - All Perils", "fault": "Third Party",
        "accident_area": "Rural", "police_report": "Yes", "witness_present": "No",
        "agent_type": "External", "past_claims": "1", "address_change": "1 year",
        "vehicle_price": "30000 to 39000", "deductible": "500", "driver_rating": "3", "year": "1996",
        "fraud_probability": "0.0891", "decision": "APPROVED", "anomaly_score": "0.38",
        "top_shap_feature_1": "DriverRating=+0.2",
        "top_shap_feature_2": "VehiclePrice=-0.3",
        "top_shap_feature_3": "AccidentArea_Rural=-0.1",
        "nlp_risk_score": "0.08", "nlp_valid": "True", "nlp_flag_count": "0",
        "nlp_flags": "", "nlp_keywords": "",
        "genai_source": "demo", "genai_summary_snippet": "Claim aligns with policy history; no red flags in narrative.",
        "reviewer_action": "", "reviewer_notes": "", "final_outcome": "APPROVED", "reviewed_at": "",
    },
    {
        "timestamp": "2026-04-01 11:30:15",
        "claim_id": "CLM-DEMO-APPROVED-003",
        "age": "29", "sex": "Male", "make": "Honda", "vehicle_category": "Sedan",
        "policy_type": "Sedan - Collision", "fault": "Policy Holder",
        "accident_area": "Urban", "police_report": "Yes", "witness_present": "Yes",
        "agent_type": "Internal", "past_claims": "none", "address_change": "no change",
        "vehicle_price": "less than 20000", "deductible": "300", "driver_rating": "4", "year": "1994",
        "fraud_probability": "0.156", "decision": "APPROVED", "anomaly_score": "0.41",
        "top_shap_feature_1": "Deductible=-0.2",
        "top_shap_feature_2": "Fault_Policy Holder=-0.15",
        "top_shap_feature_3": "AgentType_Internal=-0.1",
        "nlp_risk_score": "0.15", "nlp_valid": "True", "nlp_flag_count": "0",
        "nlp_flags": "", "nlp_keywords": "",
        "genai_source": "demo", "genai_summary_snippet": "Straightforward fender-bender; supporting details adequate.",
        "reviewer_action": "APPROVED", "reviewer_notes": "Spot check OK.", "final_outcome": "APPROVED",
        "reviewed_at": "2026-04-01 14:00:00",
    },
    # --- HUMAN_REVIEW (borderline) ---
    {
        "timestamp": "2026-04-01 12:44:03",
        "claim_id": "CLM-DEMO-REVIEW-001",
        "age": "44", "sex": "Female", "make": "Mazda", "vehicle_category": "Sedan",
        "policy_type": "Sedan - All Perils", "fault": "Third Party",
        "accident_area": "Urban", "police_report": "No", "witness_present": "No",
        "agent_type": "External", "past_claims": "2 to 4", "address_change": "under 6 months",
        "vehicle_price": "40000 to 59000", "deductible": "400", "driver_rating": "2", "year": "1995",
        "fraud_probability": "0.412", "decision": "HUMAN_REVIEW", "anomaly_score": "0.52",
        "top_shap_feature_1": "Fault_Third Party=+0.5",
        "top_shap_feature_2": "PastNumberOfClaims=+0.4",
        "top_shap_feature_3": "AddressChange_Claim=+0.3",
        "nlp_risk_score": "0.45", "nlp_valid": "True", "nlp_flag_count": "2",
        "nlp_flags": "timing inconsistency | sparse detail",
        "nlp_keywords": "cash, urgent",
        "genai_source": "demo", "genai_summary_snippet": "Borderline score; recommend specialist review of timeline vs policy dates.",
        "reviewer_action": "", "reviewer_notes": "", "final_outcome": "HUMAN_REVIEW", "reviewed_at": "",
    },
    {
        "timestamp": "2026-04-01 13:10:40",
        "claim_id": "CLM-DEMO-REVIEW-002",
        "age": "61", "sex": "Male", "make": "Pontiac", "vehicle_category": "Sport",
        "policy_type": "Sport - Collision", "fault": "Policy Holder",
        "accident_area": "Urban", "police_report": "No", "witness_present": "Yes",
        "agent_type": "External", "past_claims": "1", "address_change": "2 to 3 years",
        "vehicle_price": "60000 to 69000", "deductible": "700", "driver_rating": "3", "year": "1994",
        "fraud_probability": "0.385", "decision": "HUMAN_REVIEW", "anomaly_score": "0.48",
        "top_shap_feature_1": "VehiclePrice=+0.35",
        "top_shap_feature_2": "PoliceReportFiled_No=+0.25",
        "top_shap_feature_3": "anomaly_score=+0.2",
        "nlp_risk_score": "0.38", "nlp_valid": "True", "nlp_flag_count": "1",
        "nlp_flags": "witness statement vs damage mismatch",
        "nlp_keywords": "",
        "genai_source": "demo", "genai_summary_snippet": "Moderate risk; witness present but no police report — manual verification suggested.",
        "reviewer_action": "APPROVED", "reviewer_notes": "Body shop photos corroborate witness.", "final_outcome": "APPROVED",
        "reviewed_at": "2026-04-01 15:22:00",
    },
    {
        "timestamp": "2026-04-01 14:02:18",
        "claim_id": "CLM-DEMO-REVIEW-003",
        "age": "33", "sex": "Female", "make": "Nissan", "vehicle_category": "Sedan",
        "policy_type": "Sedan - Liability", "fault": "Third Party",
        "accident_area": "Rural", "police_report": "No", "witness_present": "No",
        "agent_type": "External", "past_claims": "more than 4", "address_change": "4 to 8 years",
        "vehicle_price": "20000 to 29000", "deductible": "400", "driver_rating": "1", "year": "1996",
        "fraud_probability": "0.551", "decision": "HUMAN_REVIEW", "anomaly_score": "0.55",
        "top_shap_feature_1": "PastNumberOfClaims=+0.6",
        "top_shap_feature_2": "DriverRating=+0.45",
        "top_shap_feature_3": "PoliceReportFiled_No=+0.3",
        "nlp_risk_score": "0.62", "nlp_valid": "False", "nlp_flag_count": "3",
        "nlp_flags": "duplicate phrasing | vague location | date gap",
        "nlp_keywords": "lawyer, settlement",
        "genai_source": "demo", "genai_summary_snippet": "Elevated risk; narrative quality poor and claim history heavy — fraud unit queue.",
        "reviewer_action": "REJECTED", "reviewer_notes": "Unable to verify loss location; claimant unresponsive.", "final_outcome": "REJECTED",
        "reviewed_at": "2026-04-01 16:45:00",
    },
    {
        "timestamp": "2026-04-01 15:18:55",
        "claim_id": "CLM-DEMO-REVIEW-004",
        "age": "47", "sex": "Male", "make": "Chevrolet", "vehicle_category": "Utility",
        "policy_type": "Utility - Collision", "fault": "Third Party",
        "accident_area": "Urban", "police_report": "Yes", "witness_present": "No",
        "agent_type": "Internal", "past_claims": "2 to 4", "address_change": "no change",
        "vehicle_price": "40000 to 59000", "deductible": "500", "driver_rating": "2", "year": "1995",
        "fraud_probability": "0.498", "decision": "HUMAN_REVIEW", "anomaly_score": "0.50",
        "top_shap_feature_1": "Days_Policy_Claim=+0.4",
        "top_shap_feature_2": "Fault_Third Party=+0.35",
        "top_shap_feature_3": "WitnessPresent_No=+0.2",
        "nlp_risk_score": "0.41", "nlp_valid": "True", "nlp_flag_count": "1",
        "nlp_flags": "repair estimate pending",
        "nlp_keywords": "",
        "genai_source": "demo", "genai_summary_snippet": "Borderline; escalate if estimate exceeds threshold.",
        "reviewer_action": "ESCALATED", "reviewer_notes": "Referred to SIU for pattern review.", "final_outcome": "ESCALATED",
        "reviewed_at": "2026-04-01 17:10:00",
    },
    # --- REJECTED (high fraud probability) ---
    {
        "timestamp": "2026-04-01 16:00:00",
        "claim_id": "CLM-DEMO-REJECT-001",
        "age": "22", "sex": "Male", "make": "BMW", "vehicle_category": "Sport",
        "policy_type": "Sport - Collision", "fault": "Third Party",
        "accident_area": "Urban", "police_report": "No", "witness_present": "No",
        "agent_type": "External", "past_claims": "more than 4", "address_change": "under 6 months",
        "vehicle_price": "more than 69000", "deductible": "400", "driver_rating": "1", "year": "1994",
        "fraud_probability": "0.941", "decision": "REJECTED", "anomaly_score": "0.62",
        "top_shap_feature_1": "anomaly_score=+1.1",
        "top_shap_feature_2": "Fault_Third Party=+0.9",
        "top_shap_feature_3": "PastNumberOfClaims=+0.7",
        "nlp_risk_score": "0.88", "nlp_valid": "False", "nlp_flag_count": "4",
        "nlp_flags": "staged loss language | inconsistent vehicle",
        "nlp_keywords": "total loss, wire transfer",
        "genai_source": "demo", "genai_summary_snippet": "High fraud probability; multiple structural red flags. Recommend denial.",
        "reviewer_action": "", "reviewer_notes": "", "final_outcome": "REJECTED", "reviewed_at": "",
    },
    {
        "timestamp": "2026-04-01 16:22:11",
        "claim_id": "CLM-DEMO-REJECT-002",
        "age": "35", "sex": "Female", "make": "Accura", "vehicle_category": "Sport",
        "policy_type": "Sport - All Perils", "fault": "Third Party",
        "accident_area": "Urban", "police_report": "No", "witness_present": "No",
        "agent_type": "External", "past_claims": "2 to 4", "address_change": "under 6 months",
        "vehicle_price": "more than 69000", "deductible": "400", "driver_rating": "2", "year": "1994",
        "fraud_probability": "0.972", "decision": "REJECTED", "anomaly_score": "0.58",
        "top_shap_feature_1": "Fault_Third Party=+0.95",
        "top_shap_feature_2": "VehiclePrice=+0.5",
        "top_shap_feature_3": "AddressChange_Claim=+0.45",
        "nlp_risk_score": "0.91", "nlp_valid": "True", "nlp_flag_count": "2",
        "nlp_flags": "duplicate VIN mention | template text",
        "nlp_keywords": "gift, cousin",
        "genai_source": "demo", "genai_summary_snippet": "Model strongly indicates fraud; anomaly score elevated.",
        "reviewer_action": "REJECTED", "reviewer_notes": "Confirmed with underwriting; deny.", "final_outcome": "REJECTED",
        "reviewed_at": "2026-04-01 18:00:00",
    },
    {
        "timestamp": "2026-04-01 17:05:33",
        "claim_id": "CLM-DEMO-REJECT-003",
        "age": "58", "sex": "Male", "make": "Dodge", "vehicle_category": "Sedan",
        "policy_type": "Sedan - Collision", "fault": "Third Party",
        "accident_area": "Rural", "police_report": "No", "witness_present": "No",
        "agent_type": "External", "past_claims": "more than 4", "address_change": "1 year",
        "vehicle_price": "60000 to 69000", "deductible": "700", "driver_rating": "1", "year": "1995",
        "fraud_probability": "0.887", "decision": "REJECTED", "anomaly_score": "0.59",
        "top_shap_feature_1": "DriverRating=+0.85",
        "top_shap_feature_2": "PoliceReportFiled_No=+0.5",
        "top_shap_feature_3": "PastNumberOfClaims=+0.55",
        "nlp_risk_score": "0.79", "nlp_valid": "False", "nlp_flag_count": "3",
        "nlp_flags": "", "nlp_keywords": "injury, chiropractor",
        "genai_source": "demo", "genai_summary_snippet": "High-risk profile and soft-tissue narrative without corroboration.",
        "reviewer_action": "", "reviewer_notes": "", "final_outcome": "REJECTED", "reviewed_at": "",
    },
    {
        "timestamp": "2026-04-01 18:40:00",
        "claim_id": "CLM-DEMO-REJECT-004",
        "age": "41", "sex": "Female", "make": "Jaguar", "vehicle_category": "Sport",
        "policy_type": "Sport - Liability", "fault": "Third Party",
        "accident_area": "Urban", "police_report": "No", "witness_present": "No",
        "agent_type": "External", "past_claims": "1", "address_change": "no change",
        "vehicle_price": "more than 69000", "deductible": "400", "driver_rating": "1", "year": "1996",
        "fraud_probability": "0.903", "decision": "REJECTED", "anomaly_score": "0.61",
        "top_shap_feature_1": "VehiclePrice=+0.7",
        "top_shap_feature_2": "Fault_Third Party=+0.65",
        "top_shap_feature_3": "anomaly_score=+0.55",
        "nlp_risk_score": "0.85", "nlp_valid": "True", "nlp_flag_count": "1",
        "nlp_flags": "luxury vehicle mismatch vs income proxy",
        "nlp_keywords": "",
        "genai_source": "demo", "genai_summary_snippet": "Automatic rejection band; manual appeal path available.",
        "reviewer_action": "", "reviewer_notes": "", "final_outcome": "REJECTED", "reviewed_at": "",
    },
]


def main() -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    for row in ROWS:
        missing = set(COLUMNS) - set(row.keys())
        extra = set(row.keys()) - set(COLUMNS)
        if missing or extra:
            raise SystemExit(f"Row schema mismatch: missing={missing} extra={extra}")
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(ROWS)
    print(f"Wrote {len(ROWS)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
