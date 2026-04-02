"""
nlp.py
------
NLP pipeline for insurance claim analysis.

Responsibilities:
  1. Claim validation  — structural checks before scoring
  2. Entity extraction — spaCy NER on free-text claim description
  3. Consistency check — compare extracted entities with structured fields
  4. Risk-keyword scan — detect red-flag phrases in text

Usage (from code):
    from nlp import validate_claim, analyze_claim_text, run_nlp_pipeline
    result = run_nlp_pipeline(claim_dict)
"""

import re
import spacy

# ── Load spaCy model (once at module import) ──────────────────────────────────

_nlp = spacy.load("en_core_web_sm")

# ── Constants ─────────────────────────────────────────────────────────────────

REQUIRED_FIELDS = [
    "Month", "DayOfWeek", "Make", "AccidentArea", "Sex",
    "MaritalStatus", "Age", "Fault", "PolicyType", "VehicleCategory",
    "VehiclePrice", "Deductible", "DriverRating", "Days_Policy_Accident",
    "Days_Policy_Claim", "AgeOfVehicle", "AgeOfPolicyHolder",
    "PoliceReportFiled", "WitnessPresent", "AgentType",
    "NumberOfCars", "Year", "BasePolicy"
]

MIN_DESCRIPTION_WORDS = 5

# Keywords that raise suspicion — intended for user-provided free-text descriptions
RISK_KEYWORDS = [
    "dui", "drunk", "alcohol", "intoxicated",
    "staged", "fabricated", "false", "fake",
    "cash only", "off the books", "no receipt",
    "multiple claims", "filed before",
    "suspicious", "inconsistent", "contradict",
]

# ── 1. Claim Validation ───────────────────────────────────────────────────────

def validate_claim(claim: dict) -> dict:
    """
    Check structural validity of a claim before ML scoring.

    Returns
    -------
    dict:
        is_valid      : bool
        issues        : list[str]  — human-readable list of problems
    """
    issues = []

    # Missing required fields
    for field in REQUIRED_FIELDS:
        if field not in claim or claim[field] is None or str(claim[field]).strip() == "":
            issues.append(f"Missing required field: '{field}'")

    # Age sanity check
    age = claim.get("Age")
    if age is not None:
        try:
            age = float(age)
            if age <= 0:
                issues.append("Age must be greater than 0")
            elif age > 100:
                issues.append(f"Age value {age} seems implausible (> 100)")
        except (ValueError, TypeError):
            issues.append("Age must be a numeric value")

    # Year sanity check
    year = claim.get("Year")
    if year is not None:
        try:
            year = int(year)
            if year < 1990 or year > 2030:
                issues.append(f"Year value {year} is outside expected range (1990-2030)")
        except (ValueError, TypeError):
            issues.append("Year must be a numeric value")

    return {
        "is_valid": len(issues) == 0,
        "issues":   issues
    }


# ── 2. Synthetic text generation ──────────────────────────────────────────────

def generate_claim_text(claim: dict) -> str:
    """
    Convert a structured claim dict into a natural-language description.
    Used when no free-text description is provided.

    This is the bridge that lets spaCy/NLP work on structured data.
    """
    age          = claim.get("Age", "unknown")
    sex          = claim.get("Sex", "unknown").lower()
    fault        = claim.get("Fault", "unknown")
    policy_type  = claim.get("PolicyType", "unknown")
    vehicle_cat  = claim.get("VehicleCategory", "unknown").lower()
    make         = claim.get("Make", "unknown")
    area         = claim.get("AccidentArea", "unknown").lower()
    police       = claim.get("PoliceReportFiled", "No")
    witness      = claim.get("WitnessPresent", "No")
    veh_price    = claim.get("VehiclePrice", "unknown")
    past_claims  = claim.get("PastNumberOfClaims", "none")
    addr_change  = claim.get("AddressChange_Claim", "no change")
    agent_type   = claim.get("AgentType", "unknown")
    deductible   = claim.get("Deductible", "unknown")
    month        = claim.get("Month", "unknown")
    day_of_week  = claim.get("DayOfWeek", "unknown")

    police_str  = "A police report was filed."      if police  == "Yes" else "No police report was filed."
    witness_str = "A witness was present."           if witness == "Yes" else "No witness was present."
    addr_str    = (f"The policyholder changed address {addr_change} before the claim."
                   if addr_change != "no change" else "No recent address change.")

    text = (
        f"A {age}-year-old {sex} filed a {policy_type} insurance claim on a {day_of_week} in {month}. "
        f"The incident occurred in a {area} area. The vehicle involved was a {make} {vehicle_cat} "
        f"valued at {veh_price}. Fault was attributed to: {fault}. "
        f"Deductible amount: {deductible}. "
        f"{police_str} {witness_str} "
        f"The agent type handling this claim is {agent_type}. "
        f"Past number of claims: {past_claims}. "
        f"{addr_str}"
    )
    return text


# ── 3. Entity Extraction ──────────────────────────────────────────────────────

def extract_entities(text: str) -> dict:
    """
    Run spaCy NER on claim text and return extracted entities by type.

    Returns
    -------
    dict:
        money_entities   : list[str]  — monetary values found
        date_entities    : list[str]  — dates/times found
        location_entities: list[str]  — GPE/LOC entities found
        org_entities     : list[str]  — organisations found
        person_entities  : list[str]  — person names found
        all_entities     : list[dict] — full entity list with labels
    """
    doc = _nlp(text)

    entities = {
        "money_entities":    [],
        "date_entities":     [],
        "location_entities": [],
        "org_entities":      [],
        "person_entities":   [],
        "all_entities":      []
    }

    for ent in doc.ents:
        record = {"text": ent.text, "label": ent.label_}
        entities["all_entities"].append(record)

        if ent.label_ == "MONEY":
            entities["money_entities"].append(ent.text)
        elif ent.label_ in ("DATE", "TIME"):
            entities["date_entities"].append(ent.text)
        elif ent.label_ in ("GPE", "LOC"):
            entities["location_entities"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["org_entities"].append(ent.text)
        elif ent.label_ == "PERSON":
            entities["person_entities"].append(ent.text)

    return entities


# ── 4. Consistency Check ──────────────────────────────────────────────────────

def check_consistency(claim: dict, text: str) -> dict:
    """
    Compare free-text claim description against structured claim fields.
    Only meaningful for user-provided descriptions (not synthetic text).

    Flags inconsistencies that could signal fraud.

    Returns
    -------
    dict:
        flags       : list[str]   — list of inconsistency descriptions
        flag_count  : int
    """
    flags = []
    text_lower = text.lower()

    # Police report: flag only if text POSITIVELY says report was filed but field says No
    # Use word-boundary check to avoid matching "no police report was filed"
    police_filed = claim.get("PoliceReportFiled", "No")
    if police_filed == "No":
        # Match "police report filed" or "filed a police report" but NOT when preceded by "no"
        if re.search(r"(?<!no )police report (was )?filed", text_lower):
            flags.append("Text implies police report filed but structured field says No")

    # Witness: flag only if text POSITIVELY says witness was present but field says No
    witness_present = claim.get("WitnessPresent", "No")
    if witness_present == "No":
        if re.search(r"(?<!no )witness (was )?present", text_lower):
            flags.append("Text implies witness present but structured field says No")

    # Fault mismatch
    fault = claim.get("Fault", "")
    if fault == "Policy Holder" and "third party" in text_lower:
        flags.append("Text mentions third party but fault field is Policy Holder")
    elif fault == "Third Party" and re.search(r"\bpolicy holder\b", text_lower) and "third party" not in text_lower:
        flags.append("Text suggests policy holder fault but fault field is Third Party")

    return {
        "flags":      flags,
        "flag_count": len(flags)
    }


# ── 5. Risk Keyword Scan ──────────────────────────────────────────────────────

def scan_risk_keywords(text: str) -> dict:
    """
    Scan claim text for known fraud risk keywords.

    Returns
    -------
    dict:
        detected_keywords : list[str]
        risk_keyword_count: int
        has_risk_keywords : bool
    """
    text_lower = text.lower()
    detected   = [kw for kw in RISK_KEYWORDS if kw in text_lower]

    return {
        "detected_keywords":  detected,
        "risk_keyword_count": len(detected),
        "has_risk_keywords":  len(detected) > 0
    }


# ── 6. Description Validation ─────────────────────────────────────────────────

def validate_description(text: str) -> dict:
    """
    Basic text quality checks on claim description.

    Returns
    -------
    dict:
        word_count    : int
        is_sufficient : bool   — meets MIN_DESCRIPTION_WORDS threshold
        issue         : str | None
    """
    if not text or not text.strip():
        return {
            "word_count":    0,
            "is_sufficient": False,
            "issue":         "Description is empty"
        }

    word_count = len(text.split())
    is_sufficient = word_count >= MIN_DESCRIPTION_WORDS

    return {
        "word_count":    word_count,
        "is_sufficient": is_sufficient,
        "issue":         None if is_sufficient else f"Description too short ({word_count} words, min {MIN_DESCRIPTION_WORDS})"
    }


# ── 7. Main NLP Pipeline ──────────────────────────────────────────────────────

def run_nlp_pipeline(claim: dict, description: str = None) -> dict:
    """
    Run the full NLP pipeline on a claim.

    Parameters
    ----------
    claim       : dict  — structured claim fields
    description : str   — optional free-text description; if None, auto-generated

    Returns
    -------
    dict with keys:
        is_valid          : bool
        validation_issues : list[str]
        claim_text        : str
        description_check : dict
        entities          : dict
        consistency       : dict
        risk_keywords     : dict
        nlp_risk_score    : float  (0-1, higher = more suspicious)
        nlp_summary       : str    (one-line text summary for GenAI stage)
    """
    # Step 1: Structural validation
    validation = validate_claim(claim)

    # Step 2: Prepare text (use provided or generate synthetic)
    if description and description.strip():
        claim_text = description.strip()
    else:
        claim_text = generate_claim_text(claim)

    # Step 3: Description quality check
    desc_check = validate_description(claim_text)

    # Step 4: Entity extraction
    entities = extract_entities(claim_text)

    # Step 5: Consistency check (only meaningful for user-provided text)
    is_synthetic = description is None or not description.strip()
    consistency  = check_consistency(claim, claim_text) if not is_synthetic else {"flags": [], "flag_count": 0}

    # Step 6: Risk keyword scan (only on user-provided text; synthetic text has no red flags)
    risk_kw = scan_risk_keywords(claim_text) if not is_synthetic else {
        "detected_keywords": [], "risk_keyword_count": 0, "has_risk_keywords": False
    }

    # Step 7: Compute NLP risk score (0-1)
    # Factors: missing fields, consistency flags, risk keywords
    missing_count = len(validation["issues"])
    flag_count    = consistency["flag_count"]
    kw_count      = risk_kw["risk_keyword_count"]

    # Weighted formula — capped at 1.0
    nlp_risk_score = min(
        (missing_count * 0.05) + (flag_count * 0.20) + (kw_count * 0.15),
        1.0
    )

    # Step 8: One-line NLP summary for GenAI stage
    police   = claim.get("PoliceReportFiled", "No")
    witness  = claim.get("WitnessPresent", "No")
    fault    = claim.get("Fault", "unknown")
    age      = claim.get("Age", "?")
    make     = claim.get("Make", "?")
    p_type   = claim.get("PolicyType", "?")
    area     = claim.get("AccidentArea", "?")

    nlp_summary = (
        f"Claimant age {age}, {fault} fault, {p_type} policy, {make} vehicle, "
        f"{area} accident area. Police report: {police}. Witness: {witness}. "
        f"NLP risk score: {nlp_risk_score:.2f}. "
        f"Consistency flags: {flag_count}. "
        f"Risk keywords detected: {kw_count}."
    )

    return {
        "is_valid":          validation["is_valid"],
        "validation_issues": validation["issues"],
        "claim_text":        claim_text,
        "description_check": desc_check,
        "entities":          entities,
        "consistency":       consistency,
        "risk_keywords":     risk_kw,
        "nlp_risk_score":    round(nlp_risk_score, 4),
        "nlp_summary":       nlp_summary
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
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

    print("Running NLP pipeline...")
    result = run_nlp_pipeline(sample_claim)

    print(f"\nClaim Valid       : {result['is_valid']}")
    print(f"Validation Issues : {result['validation_issues']}")
    print(f"\nGenerated Text:\n  {result['claim_text']}")
    print(f"\nEntities Extracted:")
    for k, v in result["entities"].items():
        if k != "all_entities" and v:
            print(f"  {k:<22}: {v}")
    print(f"\nConsistency Flags : {result['consistency']['flags']}")
    print(f"Risk Keywords     : {result['risk_keywords']['detected_keywords']}")
    print(f"NLP Risk Score    : {result['nlp_risk_score']}")
    print(f"\nNLP Summary for GenAI:\n  {result['nlp_summary']}")
