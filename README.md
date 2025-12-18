# 📊 Customer Churn Prediction Studio

### End-to-End Machine Learning for Banking & Telecom

> **A production-ready solution that doesn't just predict churn—it explains it.**
> This project demonstrates a full-stack ML pipeline for the **Telecom and Banking domains**, featuring deep learning, automated PDF reporting, and Explainable AI (XAI) for business decision support.

---

## 🧠 The Business Problem

Customer churn is a silent revenue killer. In subscription models, acquiring a new customer is **5-25x more expensive** than retaining an existing one.

**The Goal:** Move beyond simple binary classification to provide:

1. **Early Warning:** Identify high-risk customers before they leave.
2. **Root Cause Analysis:** Use SHAP to explain *why* a specific customer is at risk (e.g., "Contract Type" vs. "Support Calls").
3. **Actionable Intelligence:** Generate risk-stratified reports for marketing teams.

---

## 🚀 Key Features

### 1. Dual-Domain Support 🏦 📡

Unlike standard projects that focus on one dataset, this system handles two distinct business domains with separate preprocessing pipelines:

* **Telecom:** Focuses on service usage, contract types, and payment methods.
* **Banking:** Focuses on credit scores, account balances, and product engagement.

### 2. Business-Driven Feature Engineering ⚙️

Features were engineered to capture behavioral signals, not just raw data:

* **`BalanceSalaryRatio`:** Estimates financial stability (Bank).
* **`TenureStrategy`:** Segments customers into New, Established, and Loyal bins (Telco).
* **`LoyaltyScore`:** A composite metric derived from activity and tenure.

### 3. Explainable AI (XAI) 🔍

Black-box models are hard to trust. I integrated **SHAP (SHapley Additive exPlanations)** to provide:

* **Waterfall Plots:** Visualizing exactly which features pushed a customer's risk score up or down.
* **Global Importance:** Identifying the top churn drivers across the entire customer base.

### 4. Automated Reporting 📄

The app generates a downloadable **Executive PDF Report** summarizing:

* Total Churn Risk
* Risk Segmentation (High/Medium/Low)
* Top Drivers of Churn
* *Built using ReportLab.*

---

## 🛠️ Tech Stack

| Component | Tools Used |
| --- | --- |
| **Core Logic** | Python 3.12, Pandas, NumPy |
| **Machine Learning** | TensorFlow (Keras), Scikit-Learn |
| **Model Architecture** | Neural Networks (Dense), Entity Embeddings (for Categorical Data) |
| **Explainability** | SHAP (KernelExplainer) |
| **Web App** | Streamlit |
| **Reporting** | ReportLab |

---

## 🤖 Model Performance

*Optimization Focus: Recall (Minimizing False Negatives)*

| Domain | Metric | Score |
| --- | --- | --- |
| **Telecom** | ROC-AUC | **~0.84** |
| **Banking** | Accuracy | **~80%+** |

---

## 📸 Application Screenshots

*(Add your screenshots here)*

* **Dashboard Home:** *[Image Placeholder]*
* **SHAP Waterfall Plot:** *[Image Placeholder]*
* **PDF Report Preview:** *[Image Placeholder]*

---

## 👨‍💻 Author

**Rupesh Sharan**
*CSE (AI/ML) Undergraduate | Aspiring Machine Learning Engineer*

[LinkedIn](https://www.google.com/search?q=https://linkedin.com/in/rupesh-sharan-chavan-452a98289) | [GitHub](https://www.google.com/search?q=https://github.com/RupeshSharan)
