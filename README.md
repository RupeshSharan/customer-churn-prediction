# 📊 Customer Churn Prediction Studio

**End-to-End Machine Learning Project | Business-Ready | Explainable AI**

> A production-ready churn prediction system for **Telecom and Banking domains**, featuring advanced feature engineering, deep learning models, explainable AI (SHAP), and an interactive Streamlit dashboard for real-world decision support.

---

## 🧠 Business Problem

Customer churn directly impacts:

* Revenue
* Customer lifetime value
* Marketing costs

**Goal:**
Predict **which customers are likely to churn**, explain *why*, and enable **early intervention strategies**.

---

## 📊 Datasets Used

### 🔹 Telecom Churn Dataset

* Customer demographics
* Subscription details
* Service usage
* Billing & tenure

### 🔹 Bank Customer Churn Dataset

* Credit score
* Balance & salary
* Product usage
* Complaints & satisfaction score

---

## ⚙️ Feature Engineering (Business-Driven)

### Telecom

* `avg_charges_per_month`
* `num_services`
* `is_new_customer`
* `tenure_bin` (early churn detection)

### Banking

* `BalanceSalaryRatio`
* `LoyaltyScore`
* Age segmentation
* Product interaction features

Each feature was engineered to **reflect real business intuition**, not just improve accuracy.

---

## 🤖 Models Used

### 🏦 Banking Churn Model

* Neural Network with **Embedding layers**
* Handles high-cardinality categorical features
* Optimized using EarlyStopping & ModelCheckpoint

### 📡 Telecom Churn Model

* Deep Neural Network
* StandardScaler pipeline
* Class imbalance handling

📌 Both models are saved and reused in production.

---

## 📈 Model Explainability (SHAP)

This project uses **SHAP (SHapley Additive Explanations)** to ensure transparency:

* 🔍 Waterfall plots for **single customer explanations**
* 📊 Feature importance visualization
* 🧠 Business-friendly interpretation


---

## 🖥️ Streamlit Application

### Features:

* 📁 Batch CSV predictions
* 🧍 Single-customer churn prediction
* 🎯 Risk classification (Low / Medium / High)
* 🔎 SHAP explanation on demand
* 📄 One-click PDF business report export

---

## 📄 Business PDF Report

Automatically generated executive report including:

* Average churn risk
* Risk distribution
* Top churn drivers (SHAP)
* Customer volume summary

Perfect for:

* Management reviews
* Strategy meetings
* Client presentations

---

## 🧪 Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **TensorFlow / Keras**
* **SHAP**
* **Streamlit**
* **ReportLab**
* **Git & GitHub**

---

## 📌 Key ML Engineering Practices Demonstrated

✔ Feature consistency between training & inference
✔ Robust handling of missing / unseen categories
✔ Scalable preprocessing pipelines
✔ Model explainability
✔ Production-oriented UI design

---

## 🎯 Results & Impact

* Identifies high-risk customers **before churn**
* Explains *why* customers churn
* Enables targeted retention strategies
* Reduces decision latency with automation

---

## 👨‍💻 Author

**Rupesh Sharan**
🎓 CSE (AI/ML) Student
💡 Aspiring Data Scientist / ML Engineer
