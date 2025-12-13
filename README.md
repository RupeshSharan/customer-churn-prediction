# Customer Churn Prediction Studio 🚀

An end-to-end machine learning system for predicting customer churn across **Banking** and **Telecom** domains, featuring robust preprocessing, deep learning models, business-ready reporting, and an interactive Streamlit UI.

---

## 📌 Project Highlights

- Dual-domain churn prediction (Bank + Telco)
- Deep Learning models with embeddings (Bank) and dense networks (Telco)
- Robust handling of missing & unseen features at inference time
- Business-friendly risk segmentation (Low / Medium / High)
- Interactive Streamlit dashboard
- Automated PDF business report generation
- Model explainability using SHAP (Telco model)

---

## 🧠 Models Used

### 🔹 Bank Churn Model
- Neural Network with **Embedding layers** for categorical variables
- Engineered features:
  - Balance–Salary Ratio
  - Loyalty Score
- Optimized using EarlyStopping
- Saved in native Keras format

### 🔹 Telco Churn Model
- Fully dense neural network
- Extensive feature engineering:
  - Average charges per month
  - Number of active services
  - New customer flag
  - Tenure binning
- Scaled numerical features
- SHAP explainability supported

---

## 🗂️ Project Structure

