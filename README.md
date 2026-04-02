# Insurance Fraud Detection System

## Results in brief (percentages)

| Item | Summary |
|------|---------|
| **Class balance** | About **94%** not-fraud, **6%** fraud in `fraud_oracle.csv` (~15.4k rows) |
| **AUC-ROC (validation)** | About **85%** of random (fraud, not-fraud) pairs ranked correctly (scale 0–100%) |
| **AUC-ROC (test)** | About **82%** |
| **AUC-PR (validation)** | About **24%** average precision for the fraud class (baseline ≈ fraud rate **6%**) |
| **AUC-PR (test)** | About **22%** |
| **3-class routing** | Probabilities &lt; **5%** → APPROVED; **5%–29%** → HUMAN_REVIEW; ≥ **29%** → REJECTED |

---
Optional: open **`pipeline_architecture.html`** in a browser for a visual diagram
---

## 1. Project overview

This is an **end-to-end insurance fraud detection system** for structured vehicle claims. It combines:

- **Machine learning** — LightGBM classifier plus an **Isolation Forest** anomaly score as an extra input feature  
- **Explainability** — SHAP (top drivers of the fraud score)  
- **NLP** — spaCy NER, optional synthetic claim text from fields, validation, consistency checks, risk-keyword scan  
- **Generative AI** — Groq **LLaMA 3.1** for reviewer-style summaries (with a **rule-based fallback** if no API key)  
- **Human review** — Streamlit page for borderline cases  
- **Audit trail** — append-only **`logs/audit_log.csv`**

### Dataset (`fraud_oracle.csv`)

- **What:** Tabular **vehicle insurance** claims.
- **Target:** **`FraudFound_P`** — `0` = not fraud, `1` = fraud. Roughly **94%** not fraud and **6%** fraud, so evaluation focuses on **AUC-PR** and **fraud precision/recall**, not raw accuracy alone.
- **No narrative text in CSV:** There is no long claim-description column. **`nlp.py`** can **synthesize** a short paragraph from structured fields, the Streamlit form also allows an **optional** free-text description for NLP checks.

---

## 2. Playground notebook (`playground.ipynb`) & why these models

> **Note:** Experiments live in **`playground.ipynb`**

### 2.1 What the playground notebook is for

**`playground.ipynb`** is the **experiment workspace** — run it cell-by-cell to:

- Load **`fraud_oracle.csv`**, clean it (e.g. Age = 0, `'0'` in date-like categories), and explore distributions  
- Define **ordinal vs nominal** encoding and build the final feature matrix  
- **Train/val/test** split (stratified), **SMOTE** on train only  
- Compare models and pick what ships in `train.py`  
- **Tune thresholds** (F1 on validation, then 3-class bands for APPROVED / HUMAN_REVIEW / REJECTED)  
- **Evaluate** on held-out test (AUC-ROC, AUC-PR, confusion matrix, classification report)  
- **SHAP** analysis (which features drive fraud in this dataset)  
- Optionally save artifacts (the repo’s **`src/train.py`** mirrors the winning pipeline from here)

Use the notebook when you want to **change features, try new models, or reproduce figures** without touching the Streamlit app until you are happy with results.

### 2.2 Why we chose these models

| Model / component | Role | Why we use it |
|-------------------|------|----------------|
| **Logistic Regression** | Baseline | Fast, interpretable coefficients, easy to explain to non-experts. On this dataset it underperformed tree models on **AUC-PR** (the main metric under imbalance), so it stays a **comparison baseline** in the notebook, not the deployed classifier. |
| **XGBoost** | Strong alternative | Excellent on mixed tabular data (numeric + many categoricals after OHE). In our runs it was **very close** to LightGBM on validation AUC-PR. |
| **LightGBM** | **Final classifier** | **Best validation AUC-PR** among models we compared (slightly ahead of XGBoost here), fast training on high-dimensional sparse OHE features, and **SHAP** integrates cleanly for explanations in the app. |
| **Isolation Forest** | **Feature, not a second vote** | Fraud includes **rare, unusual** patterns that labels alone may not fully capture. IF is **unsupervised**: it scores how “normal” a claim looks in feature space. We **fit IF only on training data**, then feed **`-score_samples` → `anomaly_score`** as **one extra column** into LightGBM. In SHAP, **`anomaly_score` was among the top drivers**, so it genuinely helped the supervised model rather than duplicating it. |
| **SMOTE + threshold tuning** | Handling imbalance | Raw accuracy favors the majority class; **AUC-PR** and **recall on fraud** matter more. SMOTE balances the **training** set; **thresholds** on validation set operating point for F1 / review load. |

**Summary:** We **experimented in `playground.ipynb`**, compared **LR → XGBoost → LightGBM** with an **Isolation Forest** anomaly signal, and **select LightGBM + IF feature** in **`src/train.py`** because that combination gave the best trade-off on **AUC-PR** and explainability for this dataset.

---

## 3. Tools & technology

| Area | Technology |
|------|------------|
| Language | Python 3.10+ recommended |
| Data | **pandas**, **NumPy** |
| Preprocessing | **scikit-learn** (StandardScaler, OrdinalEncoder), custom OHE alignment in **`src/preprocess.py`** |
| Imbalance | **imbalanced-learn** (SMOTE on training set only) |
| ML models | **LightGBM** (classifier), **Isolation Forest** (unsupervised anomaly → `anomaly_score` feature) |
| Explainability | **SHAP** (TreeExplainer) |
| NLP | **spaCy** + **`en_core_web_sm`** |
| LLM | **Groq** API, model **`llama-3.1-8b-instant`** (see `src/genai.py`) |
| UI | **Streamlit**, **Plotly** (gauge, bar chart, pie chart) |
| Config / secrets | **python-dotenv**, project root **`.env`** for `GROQ_API_KEY` |
| Exploration | **`playground.ipynb`** (EDA, model comparison, SHAP — see §2) |

---

## 4. How to use it

### 4.1 Prerequisites

- **`fraud_oracle.csv`** in the **project root**.  


### 4.2 Virtual environment (`myenv`)

```bash
cd "path/to/project work"
python -m venv myenv
source myenv/Scripts/activate
```

### 4.3 Install dependencies

For the full app stack (includes spaCy English model wheel and Groq client):

```bash
pip install -r src/requirements.txt
```

### 4.4 Train models

From project root (with `myenv` active):

```bash
python src/train.py
```

### 4.5 Groq API key

Create **`.env`** in the project root:

```env
GROQ_API_KEY=your_key_here
```
If the key is missing, **`genai.py`** still returns a **fallback** text summary.

### 4.6 Run the web app

From project root:

```bash
streamlit run app.py
```

Use the sidebar:

| Page | Purpose |
|------|---------|
| **Submit Claim** | Form → ML + NLP + GenAI → audit log → session state |
| **Results** | Fraud probability, decision, SHAP, NLP block, AI report |
| **Human Review** | Notes + final action for the last analyzed claim (session) |
| **Audit Dashboard** | Key Performance Indicator(KPIs), chart, recent rows, CSV download |

---

## 5. Results (model performance)

**Dataset:** `fraud_oracle.csv` — ~15,420 rows, target **`FraudFound_P`** (~94% not fraud, ~6% fraud; severe imbalance).

**Training setup (high level):**

- Stratified split: **70% train / 15% validation / 15% test**  
- **SMOTE** applied on **training** data only  
- **Isolation Forest** fit on **pre-SMOTE** training features; **negative** `score_samples` → **`anomaly_score`** appended for LightGBM  
- **Thresholds** tuned on validation; **3-class** routing stored in **`models/config.json`**

**Reported metrics:**

| Metric | Validation | Test set |
|--------|-------------------------|--------------------------|
| AUC-ROC | 85% (0.85) | 82% (0.82) |
| AUC-PR | 24% (0.24) | 22% (0.22) |
| Fraud recall| — | 50% |
| Fraud precision | — | 20% |
| Fraud F1 | — | 28% |
| Overall accuracy (0.14 threshold) | — | 85% |

**3-class routing** (from `config.json`, probabilities on 0–100% scale):  

- **`APPROVED`** if fraud probability **&lt; 5%**  
- **`HUMAN_REVIEW`** if **5% ≤ probability &lt; 29%**  
- **`REJECTED`** if probability **≥ 29%**  

---


## 6. Project layout

```
project work/
├── app.py                      # Streamlit UI
├── pipeline_architecture.html  # Static pipeline diagram (open in browser)
├── fraud_oracle.csv            # Data
├── playground.ipynb            # Experiments & EDA (see §2)
├── README.md
├── requirements.txt            # Large export (optional / notebook)
├── myenv/                      # Your virtual env (local; do not commit)
├── .env                        # GROQ_API_KEY (local; do not commit)
├── src/
│   ├── requirements.txt        # Recommended install list for the app
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── nlp.py
│   ├── genai.py
│   ├── audit.py
│   └── test_inference.py
├── models/
│   ├── config.json
│   ├── fraud_model.pkl
│   ├── isolation_forest.pkl
│   ├── scaler.pkl
│   └── ordinal_encoder.pkl
└── logs/
    └── audit_log.csv           # Created at runtime
```
