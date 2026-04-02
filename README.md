# Insurance Fraud Detection System

## Results in brief (percentages)

| Item | Summary |
|------|---------|
| **Class balance** | About **94%** not-fraud, **6%** fraud in `fraud_oracle.csv` (~15.4k rows) |
| **AUC-ROC (validation)** | About **85%** of random (fraud, not-fraud) pairs ranked correctly (scale 0–100%) |
| **AUC-ROC (test)** | About **82%** |
| **AUC-PR (validation)** | About **24%** average precision for the fraud class (baseline ≈ fraud rate **6%**) |
| **AUC-PR (test)** | About **22%** |
| **Accuracy (test, default 0.5 threshold)** | About **85%** overall — *misleading here* because **94%** of rows are not-fraud |
| **Fraud recall (test, tuned threshold ~0.14)** | The model flags about **50%** of true fraud cases as fraud |
| **Fraud precision (same setting)** | About **20%** of predicted-fraud cases are truly fraud |
| **Fraud F1 (test)** | About **28%** (harmonic mean of precision and recall for fraud) |
| **3-class routing** | Probabilities &lt; **5%** → APPROVED; **5%–29%** → HUMAN_REVIEW; ≥ **29%** → REJECTED |

**AUC-PR** and **fraud recall/precision** matter more than headline **accuracy**.*

---

## How to read this README

If you are new to the project, read in this order:

1. **Project overview** — what it is and why it exists  
2. **Playground notebook & model choices** — experiments (`playground.ipynb`) and why LightGBM + Isolation Forest  
3. **Tools & technology** — what we built with  
4. **How to run it** — environment, install, train, app  
5. **Results** — what the model achieved (numbers you can cite)  
6. **Pipeline architecture** — short step-by-step of the full flow  

Optional: open **`pipeline_architecture.html`** in a browser for a visual diagram, and add **screenshots** from Streamlit (Submit Claim, Results, Audit Dashboard) in your report or slides.

---

## 1. Project overview

This is an **end-to-end insurance fraud detection demo** for structured vehicle claims (schema aligned with **`fraud_oracle.csv`**). It combines:

- **Machine learning** — LightGBM classifier plus an **Isolation Forest** anomaly score as an extra input feature  
- **Explainability** — SHAP (top drivers of the fraud score)  
- **NLP** — spaCy NER, optional synthetic claim text from fields, validation, consistency checks, risk-keyword scan  
- **Generative AI** — Groq **LLaMA 3.1** for reviewer-style summaries (with a **rule-based fallback** if no API key)  
- **Human review** — Streamlit page for borderline cases  
- **Audit trail** — append-only **`logs/audit_log.csv`**

The main entry point for users is **`streamlit run app.py`**. Training is a separate script: **`python src/train.py`**.

### Dataset (`fraud_oracle.csv`)

- **What:** Tabular **vehicle insurance** claims (classic Oracle-style fraud dataset, often shared on Kaggle under similar names).
- **Where:** Put **`fraud_oracle.csv`** in the **project root** (next to `app.py`). `src/train.py` and `src/test_inference.py` read it from there.
- **Shape:** ~**15,420** rows; after dropping `PolicyNumber` and `RepNumber`, modeling uses **~31** columns including the target.
- **Target:** **`FraudFound_P`** — `0` = not fraud, `1` = fraud. Roughly **94%** not fraud and **6%** fraud, so evaluation focuses on **AUC-PR** and **fraud precision/recall**, not raw accuracy alone.
- **No narrative text in CSV:** There is no long claim-description column. **`nlp.py`** can **synthesize** a short paragraph from structured fields; the Streamlit form also allows an **optional** free-text description for NLP checks.
- **Redistribution:** Check the **original data license** before publishing the CSV with your repository.

---

## 2. Playground notebook (`playground.ipynb`) & why these models

> **Note:** Experiments live in **`playground.ipynb`** (Jupyter notebook), not a `playground.py` file. The notebook is for exploration; production training and inference are in **`src/train.py`** and **`src/predict.py`**.

### 2.1 What the playground notebook is for

**`playground.ipynb`** is the **experiment workspace** — run it cell-by-cell (or section-by-section) to:

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

**Summary:** We **experimented in `playground.ipynb`**, compared **LR → XGBoost → LightGBM** with an **Isolation Forest** anomaly signal, and **locked in LightGBM + IF feature** in **`src/train.py`** because that combination gave the best trade-off on **AUC-PR** and explainability for this dataset.

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

- **`fraud_oracle.csv`** in the **project root** (same folder as `app.py`).  
- **`models/`** populated with trained artifacts, **or** run training first (Section 4.4).

### 4.2 Virtual environment (`myenv`)

Create and use a virtual environment named **`myenv`** (Windows example):

```powershell
cd "path\to\project work"
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

On macOS / Linux:

```bash
cd "path/to/project work"
python -m venv myenv
source myenv/bin/activate
```

Keep the environment activated for all `pip` and `python` commands below.

### 4.3 Install dependencies

For the full app stack (includes spaCy English model wheel and Groq client):

```powershell
pip install -r src/requirements.txt
```

If **`streamlit`** is missing after install:

```powershell
pip install streamlit plotly
```

Download spaCy model if not installed via requirements:

```powershell
python -m spacy download en_core_web_sm
```

### 4.4 Train models (first time or after changing training code)

From project root (with `myenv` active):

```powershell
python src/train.py
```

This writes **`models/fraud_model.pkl`**, **`isolation_forest.pkl`**, **`scaler.pkl`**, **`ordinal_encoder.pkl`**, and **`models/config.json`** (thresholds + feature metadata).

### 4.5 Optional: Groq API key

Create **`.env`** in the project root:

```env
GROQ_API_KEY=your_key_here
```

Never commit real keys. If the key is missing, **`genai.py`** still returns a **fallback** text summary.

### 4.6 Run the web app

From project root:

```powershell
streamlit run app.py
```

Use the sidebar:

| Page | Purpose |
|------|---------|
| **Submit Claim** | Form → ML + NLP + GenAI → audit log → session state |
| **Results** | Fraud probability, decision, SHAP, NLP block, AI report |
| **Human Review** | Notes + final action for the last analyzed claim (session) |
| **Audit Dashboard** | KPIs, chart, recent rows, CSV download |

### 4.7 Optional CLI checks

```powershell
python src/predict.py          # single sample claim
python src/test_inference.py   # three real rows → APPROVED / HUMAN_REVIEW / REJECTED examples + audit rows
python src/nlp.py
python src/genai.py
python src/audit.py
```

---

## 5. Results (model performance)

**Dataset:** `fraud_oracle.csv` — ~15,420 rows, target **`FraudFound_P`** (~94% not fraud, ~6% fraud; severe imbalance).

**Training setup (high level):**

- Stratified split: **70% train / 15% validation / 15% test**  
- **SMOTE** applied on **training** data only  
- **Isolation Forest** fit on **pre-SMOTE** training features; **negative** `score_samples` → **`anomaly_score`** appended for LightGBM  
- **Thresholds** tuned on validation; **3-class** routing stored in **`models/config.json`**

**Reported metrics** (approximate; re-run `train.py` for exact values). AUC values below are the same as **85%** / **82%** etc. when read as *proportion of the unit square* under the curve (0–100% scale).

| Metric | Validation (approx.) | Held-out test (approx.) |
|--------|-------------------------|--------------------------|
| AUC-ROC | ~85% (0.85) | ~82% (0.82) |
| AUC-PR | ~24% (0.24) | ~22% (0.22) |
| Fraud recall (tuned threshold) | — | ~50% |
| Fraud precision | — | ~20% |
| Fraud F1 | — | ~28% |
| Overall accuracy (0.5 threshold) | — | ~85% *(dominated by majority class)* |

**3-class routing** (from `config.json`, probabilities on 0–100% scale):  

- **`APPROVED`** if fraud probability **&lt; 5%**  
- **`HUMAN_REVIEW`** if **5% ≤ probability &lt; 29%**  
- **`REJECTED`** if probability **≥ 29%**  

**SHAP (importance):** **`anomaly_score`** and categorical drivers such as **Fault (Third Party)**, **BasePolicy**, **PolicyType**, etc., dominated mean |SHAP| in experiments.

**Visuals you can attach**

- Streamlit **Results** page (gauge + decision badge)  
- Streamlit **Audit Dashboard** (pie + table)  
- Notebook plots: confusion matrix, precision–recall curve, SHAP summary (from `playground.ipynb`)  

---

## 6. Pipeline architecture (brief, step by step)

Each step maps to code under **`src/`** and artifacts under **`models/`**.

| Step | What happens |
|------|----------------|
| **0 — Input** | User or script supplies **one claim** as a **dictionary** of fields (same schema as the CSV, minus label). Optional free-text description for NLP. |
| **1 — Preprocess** | **`preprocess.py`**: clean (e.g. Age=0 → median, `'0'` → Unknown), **ordinal** encode ordered fields, **one-hot** nominals aligned to training column list, **scale** numeric columns with saved scaler. |
| **2 — Anomaly feature** | **`isolation_forest.pkl`**: `anomaly_score = −score_samples(X)`; concatenated as the **last** feature before the classifier. |
| **3 — Classifier** | **`fraud_model.pkl`** (LightGBM): outputs **fraud probability** [0, 1]. |
| **4 — Decision** | **`predict.py`**: map probability to **APPROVED / HUMAN_REVIEW / REJECTED** using **`low_threshold`** and **`high_threshold`** from **`config.json`**. |
| **5 — Explain** | **SHAP** on the same feature vector; top features returned for UI / audit. |
| **6 — NLP** | **`nlp.py`**: validate fields, build **synthetic text** if no user text, **spaCy NER**, optional **consistency / keyword** checks on user text, **NLP risk score** + short summary string. |
| **7 — GenAI** | **`genai.py`**: prompt = ML + NLP + key facts → **Groq LLaMA 3.1** report, or **fallback** if API unavailable. |
| **8 — Audit** | **`audit.py`**: append row to **`logs/audit_log.csv`**; human review updates same **`claim_id`** when used from the app. |

**Note:** **`preprocess.py`** is a **library** (imported by **`train.py`** and **`predict.py`**), not a script you run by itself for the pipeline.

---

## 7. Project layout

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

---

## 8. Configuration reference

| Item | Location |
|------|-----------|
| Thresholds, OHE columns, `feature_names` | `models/config.json` |
| Groq model id | `src/genai.py` → `GROQ_MODEL` |
| Audit CSV path | `src/audit.py` → `logs/audit_log.csv` |

---

## 9. Limitations

- **Demonstration / educational** use — not a production fraud operations system.  
- **Session-based UI** — Results / Human Review refer to the **last analyzed claim** in the current browser session unless you extend persistence.  
- **Class imbalance** — SMOTE + threshold tuning; interpret precision/recall together.  
- **Data** — Confirm license/terms for **`fraud_oracle.csv`** and **Groq** when redistributing or demoing.

---

## License / data

Confirm licensing for **`fraud_oracle.csv`** and **Groq** API terms when publishing or competing.
