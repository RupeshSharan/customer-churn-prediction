# 📊 Customer Churn Studio

### Predict. Explain. Retain.

An end-to-end machine learning system that predicts customer churn for **Telecom** and **Banking** industries — and explains *why* each customer might leave, in plain English.

> *"This customer is at high risk of leaving because of their Contract type, high Monthly Charges, and no Tech Support. The company should reach out immediately."*

---

## 🎯 What Makes This Different

Most churn prediction projects stop at **"who will leave."** This one answers **"why they will leave"** and **"what to do about it."**

| Feature | Description |
|---------|-------------|
| 🧠 **Ensemble ML** | XGBoost + Random Forest (Optuna-tuned, 40 trials) |
| ⚖️ **Class Balancing** | SMOTE oversampling + physically balanced dataset (Type 2) |
| 💬 **Plain-English Explanations** | SHAP-powered, context-aware sentences anyone can read |
| 📊 **Interactive Dashboard** | Streamlit app with single-customer & batch-CSV modes |
| 📄 **PDF Reports** | Auto-generated business reports with risk distribution |

---

## 🖥️ Dashboard Preview

The Streamlit app adapts its messaging based on risk level:

- **🟢 Low Risk** → *"✅ Why This Customer Is Likely to Stay"*
- **🟡 Medium Risk** → *"⚠️ This Customer Needs Attention"*
- **🔴 High Risk** → *"🚨 Why This Customer May Leave"*

Each prediction includes:
- Churn probability percentage
- Risk level badge
- Natural-language explanation
- Visual bar chart (Keeping ← → Leaving)
- Actionable breakdown of each contributing factor

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Models** | XGBoost, Random Forest, Scikit-learn |
| **Optimization** | Optuna (Bayesian hyperparameter tuning) |
| **Explainability** | SHAP (TreeExplainer — fast & exact) |
| **Balancing** | SMOTE (imbalanced-learn) + Undersampling |
| **App** | Streamlit |
| **Reports** | ReportLab (PDF generation) |
| **Data** | Pandas, NumPy |
| **Language** | Python 3.12 |

---

## 📁 Project Structure

```
customer_churn/
├── streamlit_churn_app.py          # 🖥️  Streamlit dashboard
├── churn_explainer.py              # 💬  SHAP explanation engine
├── train_telco_optimized.py        # 🧠  Telco model training (SMOTE)
├── train_telco_type2.py            # 🧠  Telco Type 2 training (undersampled)
├── train_bank_optimized.py         # 🧠  Bank model training
├── requirements.txt                # 📦  Dependencies
├── data/
│   ├── telco_churn_processed_for_modeling.csv
│   └── Customer-Churn-Records-bank.csv
├── models/
│   ├── telco_ensemble_model.pkl    # Telco ensemble (XGB + RF)
│   ├── telco_xgb_model.pkl         # Telco XGBoost (for SHAP)
│   ├── telco2_ensemble_model.pkl   # Telco Type 2 ensemble
│   ├── telco2_xgb_model.pkl        # Telco Type 2 XGB (for SHAP)
│   ├── bank_ensemble_model.pkl     # Bank ensemble
│   ├── bank_xgb_model.pkl          # Bank XGBoost (for SHAP)
│   └── *_scaler.pkl, *_encoders.pkl, *_feature_order.pkl
├── Book1.twb                       # 📊  Tableau workbook
├── bank.twb                        # 📊  Tableau workbook
└── telco_customer_churn_viz.twbx   # 📊  Tableau packaged workbook
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Models (optional — pre-trained models are included)

```bash
# Telco (SMOTE-balanced, all 7043 rows)
python train_telco_optimized.py

# Telco Type 2 (undersampled to 3738 balanced rows)
python train_telco_type2.py

# Bank
python train_bank_optimized.py
```

### 3. Launch the Dashboard

```bash
streamlit run streamlit_churn_app.py
```

---

## 📊 Datasets

### Telco Customer Churn
- **Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers × 21 features
- **Target:** `Churn` (Yes/No — 27% churn rate)
- **Key Features:** Contract type, Monthly Charges, Internet Service, Tenure, Tech Support

### Bank Customer Churn
- **Source:** [Kaggle — Bank Customer Churn Records](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)
- **Size:** 10,000 customers × 18 features
- **Target:** `Exited` (0/1 — 20% churn rate)
- **Key Features:** Age, Geography, Complaints, Balance, Active Membership

---

## 🧠 Model Architecture

### Training Pipeline

```
Raw Data → Feature Engineering → Encoding → Scaling → Balancing → Optuna Tuning → Ensemble Training
```

### Feature Engineering

**Telco:**
- `avg_charges_per_month` — spending rate
- `tenure_x_monthly` — loyalty × spending interaction
- `charge_ratio` — proportion of monthly vs. total charges
- `high_charge_short_tenure` — new customers paying a lot

**Bank:**
- `BalanceSalaryRatio` — financial health indicator
- `LoyaltyScore` — tenure × products
- `ProductsPerTenure` — product adoption rate
- `BalanceEngagement` — has savings + actively uses account

### Three Model Variants

| Model | Dataset | Balancing | Training Size | Use Case |
|-------|---------|-----------|---------------|----------|
| **Telco** | Full (7,043) | SMOTE | 8,278 (synthetic) | General prediction |
| **Telco Type 2** | Balanced (3,738) | Undersampled | 2,990 | Equal class representation |
| **Bank** | Full (10,000) | SMOTE | ~16,000 (synthetic) | Banking churn |

---

## 💬 How Explanations Work

The system uses **SHAP (SHapley Additive exPlanations)** with `TreeExplainer` on the standalone XGBoost model to decompose each prediction into per-feature contributions.

The raw SHAP values are translated into **context-aware plain English**:

### Low Risk (< 33%)
> *"This customer is likely to stay because of their average monthly spending, Online Security service, and Tech Support access. However, keep an eye on their Contract type — if these are not managed well, the customer could become unhappy over time."*

### High Risk (> 66%)
> *"This customer is at high risk of leaving because of their Contract type, high Monthly Charges, and no Tech Support. The company should reach out to this customer immediately and address these concerns to prevent them from leaving."*

---

## 📈 Model Performance (Test Set)

All models are optimized with **Optuna** (40 trials, 5-fold stratified cross-validation).
Metrics are computed on a **held-out 20% test set** that the model never saw during training.

### Telco Models

| Metric | Telco (SMOTE) | Telco Type 2 (Balanced) | Improvement |
|--------|:------------:|:----------------------:|:-----------:|
| **ROC-AUC** | 0.8374 | **0.8830** | +5.4% |
| **Accuracy** | 77.86% | **80.35%** | +2.5% |
| **Precision** | 57.87% | **79.63%** | +21.8% |
| **Recall** | 60.96% | **81.55%** | +20.6% |
| **F1 Score** | 59.38% | **80.58%** | +21.2% |

> **Key Insight:** The balanced (undersampled) Telco Type 2 model significantly outperforms the SMOTE-based model, especially in **Precision** and **Recall**. By removing excess majority-class samples rather than synthetically generating minority samples, the model learns cleaner decision boundaries.

### Bank Model

| Metric | Score |
|--------|:-----:|
| **ROC-AUC** | **0.9996** |
| **Accuracy** | **99.85%** |
| **Precision** | **99.75%** |
| **Recall** | **99.51%** |
| **F1 Score** | **99.63%** |

> The Bank model achieves near-perfect classification thanks to strong signal features like **Complaints** and **Active Membership**, combined with the SMOTE-balanced ensemble approach.

---

## 🗺️ Future Roadmap

- [ ] **Dockerization** — containerize for cloud deployment
- [ ] **FastAPI backend** — serve predictions via REST API
- [ ] **Drift detection** — monitor data pattern changes over time
- [ ] **A/B experiment tracking** — measure retention campaign effectiveness

---

## 👨‍💻 Author

**Rupesh Sharan**
*CSE (AI/ML) Student | Graduating 2027*

Building systems that bridge the gap between complex algorithms and business value.

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
