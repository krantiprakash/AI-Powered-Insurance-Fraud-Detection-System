"""
app.py
------
Streamlit UI for the Insurance Fraud Detection System.

Pages:
  1. Submit Claim    — fill the claim form and score it
  2. Results         — fraud probability, decision, SHAP, NLP, GenAI summary
  3. Human Review    — reviewer submits final decision for borderline claims
  4. Audit Dashboard — summary stats + full audit log table

Run:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Add src/ to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from predict import load_artifacts, predict_claim
from nlp     import run_nlp_pipeline
from genai   import generate_summary
from audit   import log_decision, log_reviewer_action, load_audit_log, audit_summary

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load ML artifacts once ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ML models...")
def get_artifacts():
    return load_artifacts()

artifacts = get_artifacts()

# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.title("🔍 Fraud Detection")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Submit Claim", "Results", "Human Review", "Audit Dashboard"]
)
st.sidebar.markdown("---")
st.sidebar.caption("AI-Powered Insurance Fraud Detection System")

# ── Helper: decision badge ────────────────────────────────────────────────────

def decision_badge(decision: str) -> str:
    colors = {
        "APPROVED":     ("✅", "#1e7e34", "#d4edda"),
        "HUMAN_REVIEW": ("⚠️",  "#856404", "#fff3cd"),
        "REJECTED":     ("❌", "#721c24", "#f8d7da"),
    }
    icon, text_color, bg_color = colors.get(decision, ("❓", "#333", "#eee"))
    return (
        f'<div style="background:{bg_color};color:{text_color};'
        f'padding:16px;border-radius:10px;text-align:center;font-size:1.4em;font-weight:bold;">'
        f'{icon} {decision}</div>'
    )

# ── Helper: gauge chart ───────────────────────────────────────────────────────

def fraud_gauge(probability: float) -> go.Figure:
    pct = round(probability * 100, 1)
    if probability < 0.3:
        color = "#28a745"
    elif probability < 0.6:
        color = "#ffc107"
    else:
        color = "#dc3545"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": "Fraud Probability", "font": {"size": 16}},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1},
            "bar":   {"color": color},
            "steps": [
                {"range": [0,  30], "color": "#d4edda"},
                {"range": [30, 60], "color": "#fff3cd"},
                {"range": [60, 100],"color": "#f8d7da"},
            ],
            "threshold": {
                "line":  {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": pct
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
    return fig

# ── Claim form fields ─────────────────────────────────────────────────────────

MONTHS       = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
DAYS         = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MAKES        = ["Honda","Toyota","Pontiac","Ford","Mazda","Chevrolet","VW","Dodge",
                 "Saturn","Mercury","Oldsmobile","Saab","Nissan","Jaguar","Accura",
                 "Suburu","Ferrari","Mecedes","BMW"]
VEHICLE_CATS = ["Sedan","Sport","Utility"]
POLICY_TYPES = ["Sedan - All Perils","Sedan - Collision","Sedan - Liability",
                 "Sport - All Perils","Sport - Collision","Sport - Liability",
                 "Utility - All Perils","Utility - Collision","Utility - Liability"]
VEH_PRICES   = ["less than 20000","20000 to 29000","30000 to 39000",
                 "40000 to 59000","60000 to 69000","more than 69000"]
DAYS_ACC     = ["none","1 to 7","8 to 15","15 to 30","more than 30"]
DAYS_CLM     = ["none","8 to 15","15 to 30","more than 30"]
PAST_CLAIMS  = ["none","1","2 to 4","more than 4"]
AGE_VEH      = ["new","2 years","3 years","4 years","5 years","6 years","7 years","more than 7"]
AGE_HOLDER   = ["16 to 17","18 to 20","21 to 25","26 to 30","31 to 35",
                 "36 to 40","41 to 50","51 to 65","over 65"]
NUM_SUPPL    = ["none","1 to 2","3 to 5","more than 5"]
ADDR_CHANGE  = ["no change","under 6 months","1 year","2 to 3 years","4 to 8 years"]
NUM_CARS     = ["1 vehicle","2 vehicles","3 to 4","5 to 8","more than 8"]
BASE_POLICY  = ["Collision","Liability","All Perils"]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Submit Claim
# ─────────────────────────────────────────────────────────────────────────────

if page == "Submit Claim":
    st.title("📋 Submit Insurance Claim")
    st.markdown("Fill in the claim details below and click **Analyze Claim** to score it.")
    st.markdown("---")

    with st.form("claim_form"):
        # ── Claimant Info ──────────────────────────────────────────────────
        st.subheader("👤 Claimant Information")
        c1, c2, c3 = st.columns(3)
        age            = c1.number_input("Age",              min_value=16, max_value=100, value=34)
        sex            = c2.selectbox("Sex",                 ["Male","Female"])
        marital_status = c3.selectbox("Marital Status",      ["Single","Married","Divorced","Widow"])

        # ── Incident Details ───────────────────────────────────────────────
        st.subheader("🚗 Incident Details")
        c1, c2, c3 = st.columns(3)
        month         = c1.selectbox("Month of Incident",   MONTHS, index=0)
        day_of_week   = c2.selectbox("Day of Week",         DAYS, index=0)
        week_of_month = c3.number_input("Week of Month",    min_value=1, max_value=5, value=3)

        c1, c2, c3 = st.columns(3)
        accident_area = c1.selectbox("Accident Area",       ["Urban","Rural"])
        fault         = c2.selectbox("Fault",               ["Policy Holder","Third Party"])
        police_report = c3.selectbox("Police Report Filed", ["No","Yes"])

        c1, c2 = st.columns(2)
        witness_present = c1.selectbox("Witness Present",   ["No","Yes"])
        agent_type      = c2.selectbox("Agent Type",        ["External","Internal"])

        # ── Claim Timing ───────────────────────────────────────────────────
        st.subheader("📅 Claim Timing")
        c1, c2, c3 = st.columns(3)
        month_claimed        = c1.selectbox("Month Claimed",         MONTHS, index=0)
        day_of_week_claimed  = c2.selectbox("Day of Week Claimed",   DAYS, index=1)
        week_of_month_claimed= c3.number_input("Week of Month Claimed", min_value=1, max_value=5, value=4)

        c1, c2 = st.columns(2)
        days_policy_accident = c1.selectbox("Days Policy → Accident", DAYS_ACC, index=4)
        days_policy_claim    = c2.selectbox("Days Policy → Claim",    DAYS_CLM, index=3)

        # ── Vehicle Information ────────────────────────────────────────────
        st.subheader("🚙 Vehicle Information")
        c1, c2, c3 = st.columns(3)
        make             = c1.selectbox("Make",             MAKES, index=0)
        vehicle_category = c2.selectbox("Vehicle Category", VEHICLE_CATS, index=1)
        vehicle_price    = c3.selectbox("Vehicle Price",    VEH_PRICES, index=5)

        c1, c2 = st.columns(2)
        age_of_vehicle = c1.selectbox("Age of Vehicle",     AGE_VEH, index=5)
        year           = c2.selectbox("Year",               [1994, 1995, 1996], index=0)

        # ── Policy Information ─────────────────────────────────────────────
        st.subheader("📄 Policy Information")
        c1, c2, c3 = st.columns(3)
        policy_type        = c1.selectbox("Policy Type",          POLICY_TYPES, index=1)
        base_policy        = c2.selectbox("Base Policy",          BASE_POLICY, index=0)
        age_of_policy_holder = c3.selectbox("Age of Policy Holder", AGE_HOLDER, index=4)

        c1, c2, c3 = st.columns(3)
        deductible        = c1.selectbox("Deductible ($)",        [300, 400, 500, 700], index=1)
        driver_rating     = c2.selectbox("Driver Rating",         [1, 2, 3, 4], index=3)
        past_claims       = c3.selectbox("Past Number of Claims", PAST_CLAIMS, index=0)

        c1, c2, c3 = st.columns(3)
        num_suppliments   = c1.selectbox("Number of Supplements", NUM_SUPPL, index=0)
        address_change    = c2.selectbox("Address Change (Claim)", ADDR_CHANGE, index=0)
        number_of_cars    = c3.selectbox("Number of Cars",        NUM_CARS, index=0)

        # ── Optional description ───────────────────────────────────────────
        st.subheader("📝 Claim Description (Optional)")
        description = st.text_area(
            "Free-text description of the incident",
            placeholder="Describe the incident in your own words. Leave blank to use auto-generated description.",
            height=100
        )

        submitted = st.form_submit_button("🔍 Analyze Claim", use_container_width=True)

    if submitted:
        claim = {
            "Month": month, "WeekOfMonth": week_of_month,
            "DayOfWeek": day_of_week, "Make": make,
            "AccidentArea": accident_area,
            "DayOfWeekClaimed": day_of_week_claimed,
            "MonthClaimed": month_claimed,
            "WeekOfMonthClaimed": week_of_month_claimed,
            "Sex": sex, "MaritalStatus": marital_status,
            "Age": age, "Fault": fault,
            "PolicyType": policy_type, "VehicleCategory": vehicle_category,
            "VehiclePrice": vehicle_price, "Deductible": deductible,
            "DriverRating": driver_rating,
            "Days_Policy_Accident": days_policy_accident,
            "Days_Policy_Claim": days_policy_claim,
            "PastNumberOfClaims": past_claims,
            "AgeOfVehicle": age_of_vehicle,
            "AgeOfPolicyHolder": age_of_policy_holder,
            "PoliceReportFiled": police_report,
            "WitnessPresent": witness_present,
            "AgentType": agent_type,
            "NumberOfSuppliments": num_suppliments,
            "AddressChange_Claim": address_change,
            "NumberOfCars": number_of_cars,
            "Year": year, "BasePolicy": base_policy
        }

        with st.spinner("Running ML model, NLP pipeline, and GenAI summary..."):
            ml_result    = predict_claim(claim, artifacts)
            nlp_result   = run_nlp_pipeline(claim, description=description if description.strip() else None)
            genai_result = generate_summary(ml_result, nlp_result, claim)
            claim_id     = log_decision(claim, ml_result, nlp_result, genai_result)

        # Store in session state for Results page
        st.session_state["claim"]        = claim
        st.session_state["ml_result"]    = ml_result
        st.session_state["nlp_result"]   = nlp_result
        st.session_state["genai_result"] = genai_result
        st.session_state["claim_id"]     = claim_id

        st.success(f"Claim analyzed. ID: `{claim_id}` — navigate to **Results** to view the full report.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Results
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Results":
    st.title("📊 Claim Analysis Results")

    if "ml_result" not in st.session_state:
        st.info("No claim has been analyzed yet. Go to **Submit Claim** first.")
        st.stop()

    ml_result    = st.session_state["ml_result"]
    nlp_result   = st.session_state["nlp_result"]
    genai_result = st.session_state["genai_result"]
    claim        = st.session_state["claim"]
    claim_id     = st.session_state["claim_id"]

    st.markdown(f"**Claim ID:** `{claim_id}`")
    st.markdown("---")

    # ── Row 1: Decision badge + Gauge ──────────────────────────────────────
    col_badge, col_gauge, col_scores = st.columns([2, 2, 2])

    with col_badge:
        st.markdown("### Decision")
        st.markdown(decision_badge(ml_result["decision"]), unsafe_allow_html=True)
        st.markdown(f"<br><b>Anomaly Score:</b> {ml_result['anomaly_score']}", unsafe_allow_html=True)

    with col_gauge:
        st.plotly_chart(fraud_gauge(ml_result["fraud_probability"]), use_container_width=True)

    with col_scores:
        st.markdown("### Risk Scores")
        st.metric("ML Fraud Probability",  f"{ml_result['fraud_probability']:.1%}")
        st.metric("NLP Risk Score",        f"{nlp_result['nlp_risk_score']:.2f} / 1.0")
        st.metric("Anomaly Score",         f"{ml_result['anomaly_score']:.4f}")

    st.markdown("---")

    # ── Row 2: SHAP + NLP ─────────────────────────────────────────────────
    col_shap, col_nlp = st.columns(2)

    with col_shap:
        st.subheader("🔑 Top Fraud Indicators (SHAP)")
        shap_features = ml_result.get("top_shap_features", [])
        if shap_features:
            names  = [f[0].replace("_", " ") for f in shap_features]
            values = [f[1] for f in shap_features]
            colors = ["#dc3545" if v > 0 else "#28a745" for v in values]

            fig = go.Figure(go.Bar(
                x=values, y=names,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.4f}" for v in values],
                textposition="outside"
            ))
            fig.update_layout(
                height=300,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title="SHAP Value",
                yaxis={"autorange": "reversed"}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Red = increases fraud probability | Green = decreases it")
        else:
            st.info("SHAP values not available.")

    with col_nlp:
        st.subheader("🧠 NLP Analysis")
        valid = nlp_result.get("is_valid", True)
        st.markdown(f"**Claim Validity:** {'✅ Valid' if valid else '❌ Invalid'}")

        issues = nlp_result.get("validation_issues", [])
        if issues:
            st.markdown("**Validation Issues:**")
            for i in issues:
                st.markdown(f"- ⚠️ {i}")

        flags = nlp_result.get("consistency", {}).get("flags", [])
        if flags:
            st.markdown("**Consistency Flags:**")
            for f in flags:
                st.markdown(f"- 🚩 {f}")
        else:
            st.markdown("**Consistency Flags:** None ✅")

        keywords = nlp_result.get("risk_keywords", {}).get("detected_keywords", [])
        if keywords:
            st.markdown(f"**Risk Keywords:** `{'`, `'.join(keywords)}`")
        else:
            st.markdown("**Risk Keywords:** None ✅")

        st.markdown("**Extracted Entities (spaCy NER):**")
        entities = nlp_result.get("entities", {})
        for etype in ["date_entities", "org_entities", "money_entities", "location_entities"]:
            vals = entities.get(etype, [])
            if vals:
                label = etype.replace("_entities", "").title()
                st.markdown(f"- **{label}:** {', '.join(vals)}")

    st.markdown("---")

    # ── Row 3: Claim text ──────────────────────────────────────────────────
    with st.expander("📝 Claim Text Used for NLP Analysis"):
        st.write(nlp_result.get("claim_text", ""))

    # ── Row 4: GenAI Report ────────────────────────────────────────────────
    st.subheader("🤖 AI-Generated Fraud Assessment Report")
    source = genai_result.get("source", "fallback")
    model  = genai_result.get("model", "")
    st.caption(f"Generated by: **{model}** ({source})")
    if genai_result.get("error"):
        st.caption(f"Note: {genai_result['error']}")
    st.markdown(genai_result.get("summary", "No summary available."))

    st.markdown("---")
    if ml_result["decision"] == "HUMAN_REVIEW":
        st.warning("⚠️ This claim requires human review. Go to the **Human Review** page to submit your decision.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Human Review
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Human Review":
    st.title("👤 Human Review")
    st.markdown("Review borderline claims and submit your final decision.")
    st.markdown("---")

    if "claim_id" not in st.session_state:
        st.info("No claim has been analyzed yet. Go to **Submit Claim** first.")
        st.stop()

    claim_id  = st.session_state["claim_id"]
    ml_result = st.session_state["ml_result"]
    claim     = st.session_state["claim"]

    st.markdown(f"**Claim ID:** `{claim_id}`")

    # Quick summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Probability", f"{ml_result['fraud_probability']:.1%}")
    c2.metric("ML Decision",       ml_result["decision"])
    c3.metric("Anomaly Score",     f"{ml_result['anomaly_score']:.4f}")

    st.markdown("---")
    st.subheader("📋 Key Claim Facts")

    facts = {
        "Age":              claim.get("Age"),
        "Sex":              claim.get("Sex"),
        "Fault":            claim.get("Fault"),
        "Policy Type":      claim.get("PolicyType"),
        "Vehicle":          f"{claim.get('Make')} {claim.get('VehicleCategory')}",
        "Accident Area":    claim.get("AccidentArea"),
        "Police Report":    claim.get("PoliceReportFiled"),
        "Witness Present":  claim.get("WitnessPresent"),
        "Agent Type":       claim.get("AgentType"),
        "Past Claims":      claim.get("PastNumberOfClaims"),
        "Address Change":   claim.get("AddressChange_Claim"),
    }
    facts_df = pd.DataFrame(facts.items(), columns=["Field", "Value"])
    st.dataframe(facts_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("✍️ Submit Your Decision")

    with st.form("review_form"):
        reviewer_action = st.selectbox(
            "Final Decision",
            ["APPROVED", "REJECTED", "ESCALATED"],
            index=1
        )
        reviewer_notes = st.text_area(
            "Notes (required)",
            placeholder="Explain your reasoning for this decision...",
            height=120
        )
        submit_review = st.form_submit_button("✅ Submit Review", use_container_width=True)

    if submit_review:
        if not reviewer_notes.strip():
            st.error("Please provide notes before submitting.")
        else:
            success = log_reviewer_action(
                claim_id,
                action=reviewer_action,
                notes=reviewer_notes,
                final_outcome=reviewer_action
            )
            if success:
                st.success(f"Review submitted. Final outcome: **{reviewer_action}**")
                st.balloons()
            else:
                st.error("Could not find this claim in the audit log. Please re-submit the claim.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Audit Dashboard
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Audit Dashboard":
    st.title("📁 Audit Dashboard")
    st.markdown("Overview of all claims processed by the system.")
    st.markdown("---")

    summary = audit_summary()

    if summary["total_claims"] == 0:
        st.info("No claims have been processed yet. Submit a claim first.")
        st.stop()

    # ── KPI row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Claims",         summary["total_claims"])
    c2.metric("✅ Approved",           summary["approved"])
    c3.metric("⚠️ Human Review",       summary["human_review"])
    c4.metric("❌ Rejected",           summary["rejected"])
    c5.metric("Avg Fraud Probability", f"{summary['avg_fraud_prob']:.1%}")

    st.markdown("---")

    # ── Decision distribution pie ──────────────────────────────────────────
    col_pie, col_table = st.columns([1, 2])

    with col_pie:
        st.subheader("Decision Distribution")
        labels = ["Approved", "Human Review", "Rejected"]
        values = [summary["approved"], summary["human_review"], summary["rejected"]]
        colors = ["#28a745", "#ffc107", "#dc3545"]

        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            marker_colors=colors,
            hole=0.4,
            textinfo="label+percent"
        ))
        fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.subheader("Recent Claims")
        records = load_audit_log()
        if records:
            display_cols = [
                "timestamp", "claim_id", "age", "make", "fault",
                "fraud_probability", "decision", "nlp_risk_score",
                "reviewer_action", "final_outcome"
            ]
            df = pd.DataFrame(records)
            df = df[[c for c in display_cols if c in df.columns]]
            df = df.sort_values("timestamp", ascending=False).head(20)

            def color_decision(val):
                if val == "APPROVED":    return "background-color: #d4edda"
                if val == "HUMAN_REVIEW":return "background-color: #fff3cd"
                if val == "REJECTED":    return "background-color: #f8d7da"
                return ""

            styled = df.style.map(color_decision, subset=["decision"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Full log download ──────────────────────────────────────────────────
    st.subheader("📥 Download Full Audit Log")
    records = load_audit_log()
    if records:
        full_df  = pd.DataFrame(records)
        csv_data = full_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download audit_log.csv",
            data=csv_data,
            file_name="audit_log.csv",
            mime="text/csv",
            use_container_width=True
        )
