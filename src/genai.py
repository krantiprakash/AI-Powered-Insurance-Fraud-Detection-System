"""
genai.py
--------
Groq LLaMA 3 integration for generating human-readable claim summaries.

Takes combined output from predict.py (ML results) + nlp.py (NLP results)
and produces a structured natural-language report for human reviewers.

Fallback: If GROQ_API_KEY is not set or the API call fails, a rule-based
summary is returned so the app works without an API key.

Usage (from code):
    from genai import generate_summary
    report = generate_summary(ml_result, nlp_result, claim)
"""

import os
from dotenv import load_dotenv

# Load .env from project root (one level above src/)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BASE_DIR, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.1-8b-instant"


# ── Prompt Builder ────────────────────────────────────────────────────────────

def _build_prompt(ml_result: dict, nlp_result: dict, claim: dict) -> str:
    """
    Construct a structured prompt for LLaMA 3 from ML + NLP outputs.
    """
    fraud_prob   = ml_result.get("fraud_probability", "N/A")
    decision     = ml_result.get("decision", "N/A")
    anomaly      = ml_result.get("anomaly_score", "N/A")
    shap_features = ml_result.get("top_shap_features", [])

    nlp_score    = nlp_result.get("nlp_risk_score", 0)
    flags        = nlp_result.get("consistency", {}).get("flags", [])
    keywords     = nlp_result.get("risk_keywords", {}).get("detected_keywords", [])
    valid        = nlp_result.get("is_valid", True)
    issues       = nlp_result.get("validation_issues", [])
    claim_text   = nlp_result.get("claim_text", "")

    # Format SHAP features
    shap_lines = ""
    for feat, val in shap_features:
        direction = "increases" if val > 0 else "decreases"
        shap_lines += f"  - {feat}: {val:+.4f} ({direction} fraud probability)\n"

    # Format flags and keywords
    flags_str   = "\n".join(f"  - {f}" for f in flags)   if flags   else "  None"
    keywords_str = "\n".join(f"  - {k}" for k in keywords) if keywords else "  None"
    issues_str   = "\n".join(f"  - {i}" for i in issues)   if issues  else "  None"

    # Key claim facts
    age         = claim.get("Age", "?")
    sex         = claim.get("Sex", "?")
    make        = claim.get("Make", "?")
    vehicle_cat = claim.get("VehicleCategory", "?")
    policy_type = claim.get("PolicyType", "?")
    fault       = claim.get("Fault", "?")
    area        = claim.get("AccidentArea", "?")
    police      = claim.get("PoliceReportFiled", "?")
    witness     = claim.get("WitnessPresent", "?")
    agent       = claim.get("AgentType", "?")
    past_claims = claim.get("PastNumberOfClaims", "?")
    addr_change = claim.get("AddressChange_Claim", "?")

    prompt = f"""You are an insurance fraud analyst AI. Analyze the following claim and produce a concise, structured fraud assessment report for a human reviewer.

## CLAIM DETAILS
- Claimant: {age}-year-old {sex}
- Vehicle: {make} {vehicle_cat} | Policy: {policy_type}
- Accident Area: {area} | Fault: {fault}
- Police Report Filed: {police} | Witness Present: {witness}
- Agent Type: {agent}
- Past Number of Claims: {past_claims}
- Address Change Before Claim: {addr_change}

## CLAIM DESCRIPTION
{claim_text}

## ML MODEL RESULTS
- Fraud Probability: {fraud_prob} ({int(float(fraud_prob)*100)}%)
- Decision: {decision}
- Anomaly Score: {anomaly} (higher = more anomalous)

## TOP FRAUD INDICATORS (SHAP)
{shap_lines if shap_lines else "  Not available"}

## NLP ANALYSIS
- Structural Validity: {"VALID" if valid else "INVALID"}
- Validation Issues:
{issues_str}
- Consistency Flags (text vs form contradictions):
{flags_str}
- Risk Keywords Detected:
{keywords_str}
- NLP Risk Score: {nlp_score} / 1.0

## YOUR TASK
Write a professional fraud assessment report with these sections:
1. **Risk Summary** (1-2 sentences: overall risk level and decision)
2. **Key Red Flags** (bullet list of the most suspicious signals)
3. **Mitigating Factors** (bullet list of anything that reduces suspicion)
4. **Reviewer Recommendation** (one clear action: Approve / Escalate / Reject, and why)

Be concise. Use plain English. Do not repeat all the numbers — focus on the story the data tells."""

    return prompt


# ── Groq API Call ─────────────────────────────────────────────────────────────

def _call_groq(prompt: str) -> str:
    """Send prompt to Groq LLaMA 3 and return the response text."""
    from groq import Groq

    client   = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role":    "system",
                "content": (
                    "You are a concise insurance fraud analyst. "
                    "Write clear, professional reports for human reviewers. "
                    "Avoid unnecessary repetition. Be direct."
                )
            },
            {
                "role":    "user",
                "content": prompt
            }
        ],
        temperature=0.3,   # low temperature for factual, consistent output
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


# ── Rule-based Fallback ───────────────────────────────────────────────────────

def _fallback_summary(ml_result: dict, nlp_result: dict, claim: dict) -> str:
    """
    Generate a structured summary without calling the API.
    Used when GROQ_API_KEY is missing or the API call fails.
    """
    fraud_prob  = ml_result.get("fraud_probability", 0)
    decision    = ml_result.get("decision", "UNKNOWN")
    anomaly     = ml_result.get("anomaly_score", 0)
    shap        = ml_result.get("top_shap_features", [])
    nlp_score   = nlp_result.get("nlp_risk_score", 0)
    flags       = nlp_result.get("consistency", {}).get("flags", [])
    keywords    = nlp_result.get("risk_keywords", {}).get("detected_keywords", [])

    # Risk level label
    if fraud_prob >= 0.6:
        risk_level = "HIGH"
    elif fraud_prob >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Decision label
    decision_map = {
        "APPROVED":     "This claim can be approved.",
        "HUMAN_REVIEW": "This claim requires human review before a decision.",
        "REJECTED":     "This claim has been flagged for rejection."
    }
    decision_text = decision_map.get(decision, "Decision pending.")

    # Top SHAP feature
    top_feat = shap[0][0].replace("_", " ") if shap else "anomaly score"

    lines = [
        "## Fraud Assessment Report (Auto-generated)",
        "",
        f"**Risk Summary**",
        f"This claim presents a {risk_level} fraud risk with a fraud probability of "
        f"{fraud_prob:.1%}. {decision_text}",
        "",
        "**Key Red Flags**",
    ]

    if fraud_prob > 0.5:
        lines.append(f"- High ML fraud probability ({fraud_prob:.1%})")
    if anomaly > 0.45:
        lines.append(f"- Elevated anomaly score ({anomaly:.4f}) — unusual claim pattern")
    if top_feat:
        lines.append(f"- Strongest fraud indicator: {top_feat}")
    if flags:
        for f in flags:
            lines.append(f"- Consistency issue: {f}")
    if keywords:
        lines.append(f"- Risk keywords in description: {', '.join(keywords)}")
    if claim.get("PoliceReportFiled") == "No":
        lines.append("- No police report filed")
    if claim.get("WitnessPresent") == "No":
        lines.append("- No witness present")
    if claim.get("AgentType") == "External":
        lines.append("- External agent handling the claim")

    lines += [
        "",
        "**Mitigating Factors**",
    ]

    mitigating = []
    if fraud_prob < 0.3:
        mitigating.append("Low fraud probability from ML model")
    if nlp_score == 0:
        mitigating.append("No NLP red flags detected")
    if claim.get("PoliceReportFiled") == "Yes":
        mitigating.append("Police report was filed")
    if claim.get("WitnessPresent") == "Yes":
        mitigating.append("Witness was present")
    if not mitigating:
        mitigating.append("No strong mitigating factors identified")

    for m in mitigating:
        lines.append(f"- {m}")

    lines += [
        "",
        "**Reviewer Recommendation**",
    ]

    if decision == "APPROVED":
        lines.append("Approve this claim. ML and NLP signals are within acceptable range.")
    elif decision == "HUMAN_REVIEW":
        lines.append(
            "Escalate to a senior reviewer. The claim falls in a borderline zone — "
            "manual verification of key fields is recommended."
        )
    else:
        lines.append(
            "Reject this claim. Multiple high-risk signals detected. "
            "Notify the claimant and document the decision in the audit log."
        )

    return "\n".join(lines)


# ── Main Function ─────────────────────────────────────────────────────────────

def generate_summary(ml_result: dict, nlp_result: dict, claim: dict) -> dict:
    """
    Generate a human-readable fraud assessment report.

    Parameters
    ----------
    ml_result  : dict — output from predict.predict_claim()
    nlp_result : dict — output from nlp.run_nlp_pipeline()
    claim      : dict — raw claim fields

    Returns
    -------
    dict:
        summary      : str   — full report text
        source       : str   — 'groq' | 'fallback'
        model        : str   — model name used
        error        : str | None
    """
    if not GROQ_API_KEY:
        summary = _fallback_summary(ml_result, nlp_result, claim)
        return {
            "summary": summary,
            "source":  "fallback",
            "model":   "rule-based",
            "error":   "GROQ_API_KEY not set"
        }

    try:
        prompt  = _build_prompt(ml_result, nlp_result, claim)
        summary = _call_groq(prompt)
        return {
            "summary": summary,
            "source":  "groq",
            "model":   GROQ_MODEL,
            "error":   None
        }

    except Exception as e:
        summary = _fallback_summary(ml_result, nlp_result, claim)
        return {
            "summary": summary,
            "source":  "fallback",
            "model":   "rule-based",
            "error":   str(e)
        }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from predict import load_artifacts, predict_claim
    from nlp import run_nlp_pipeline

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

    print("Loading ML artifacts...")
    artifacts  = load_artifacts()

    print("Running ML prediction...")
    ml_result  = predict_claim(sample_claim, artifacts)

    print("Running NLP pipeline...")
    nlp_result = run_nlp_pipeline(sample_claim)

    print("Generating GenAI summary...")
    result     = generate_summary(ml_result, nlp_result, sample_claim)

    print(f"\nSource : {result['source']} ({result['model']})")
    if result["error"]:
        print(f"Error  : {result['error']}")
    print(f"\n{'='*60}")
    print(result["summary"])
    print("="*60)
