# 📊 Customer Churn Studio

### Turning Data into Retention Strategies

**Welcome to Customer Churn Studio!**

In the world of subscription business, losing a customer (churn) is costly. I built this project to move beyond simple prediction and focus on **actionable insights**. It’s not enough to know *who* will leave; businesses need to know *why*.

This is an end-to-end machine learning application that predicts churn risk for **Telecom** and **Banking** customers. It wraps complex Deep Learning models in a user-friendly Streamlit dashboard, complete with Explainable AI (SHAP) to help stakeholders trust the model's decisions.

---

## 💡 The "Why" Behind This Project

I wanted to build a system that mimics a real-world production environment. A raw Jupyter Notebook is great for research, but it doesn't help a marketing manager.

**My goals were to:**

1. **Solve a Real Problem:** Help businesses identify at-risk customers *before* they leave.
2. **Prioritize Explainability:** Use SHAP to break open the "black box" of Deep Learning so non-technical users can understand the drivers (e.g., "This customer is leaving because their contract is Month-to-Month").
3. **Go Full-Cycle:** From raw CSVs to a deployed app that generates PDF reports.

---

## ⚙️ How It Works (The Solution)

I designed the architecture to handle two distinct domains—**Banking and Telecom**—demonstrating that the pipeline is flexible and scalable.

### 1. The Models 🧠

I moved away from standard decision trees and implemented **Deep Learning** models using **TensorFlow/Keras**.

* **Telecom Model:** Focuses on service usage patterns and contract details.
* **Banking Model:** Utilizes **Entity Embeddings** for categorical features (like Geography and Card Type) to capture complex relationships better than standard One-Hot Encoding.

### 2. The Dashboard (Streamlit) 🖥️

The front end allows two modes of operation:

* **Single Customer Check:** Input details for one customer to get an instant risk score and a "Why?" explanation.
* **Batch Processing:** Upload a CSV of thousands of customers. The system processes them and generates a downloadable report with risk segments (High, Medium, Low).

### 3. Explainable AI (XAI) 🔍

Trust is key. I integrated **SHAP (SHapley Additive exPlanations)** to generate waterfall plots. This shows exactly which feature pushed the probability up or down for any specific customer.

---

## 🛠️ Tech Stack & Tools

* **Core Logic:** Python 3.12, Pandas, NumPy
* **Machine Learning:** TensorFlow, Keras, Scikit-learn
* **Visualization:** Matplotlib, SHAP
* **App Framework:** Streamlit
* **Reporting:** ReportLab (for generating professional PDFs)

---

## 📊 Key Features Engineered

Feature engineering is where the magic happens. I didn't just dump the raw data into the model; I created meaningful signals:

* **For Telecom:**
* `avg_charges_per_month`: Normalizing charges to find price sensitivity.
* `tenure_bin`: Grouping customers by loyalty stages (New vs. Long-term).


* **For Banking:**
* `BalanceSalaryRatio`: A derived metric to understand financial health relative to income.
* `LoyaltyScore`: A composite metric based on tenure and activity.



---

## 📈 Performance & Impact

I optimized the models for **Recall**, because in churn prediction, missing a churning customer (False Negative) is much worse than flagging a loyal one (False Positive).

* **Telecom Model:** ~84% ROC-AUC
* **Banking Model:** >80% Accuracy

---

## 🚀 Future Roadmap

This project is fully functional, but I'm always looking to improve it. Next on my list:

* **Dockerization:** Containerizing the app for easier cloud deployment.
* **API Endpoint:** Building a FastAPI backend to serve predictions to other systems.
* **Live Monitoring:** Adding a drift detection module to see if data patterns change over time.

---

## 👨‍💻 About Me

**Rupesh Sharan**
*CSE (AI/ML) Student | Graduating 2027*

I am an aspiring Machine Learning Engineer passionate about building systems that bridge the gap between complex algorithms and business value.
