import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
import shap
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime

# ---------------- CONFIG ---------------- #

st.set_page_config(page_title="Customer Churn Studio", layout="wide")

BASE_DIR = Path(r"C:\Users\rupes\Downloads\customer_churn")
MODELS_DIR = BASE_DIR / "models"

# ---------------- LOAD ASSETS ---------------- #

@st.cache_resource
def load_assets():
    return {
        "bank_model": load_model(MODELS_DIR / "bank_churn_model_emb.keras"),
        "bank_scaler": joblib.load(MODELS_DIR / "bank_scaler.pkl"),
        "bank_encoders": joblib.load(MODELS_DIR / "bank_label_encoders.pkl"),
        "telco_model": load_model(MODELS_DIR / "telco_churn_model.keras"),
        "telco_scaler": joblib.load(MODELS_DIR / "telco_scaler.pkl"),
        "telco_encoders": joblib.load(MODELS_DIR / "telco_label_encoders.pkl"),
        "telco_features": joblib.load(MODELS_DIR / "telco_feature_order.pkl"),
    }

assets = load_assets()

# ---------------- BANK PREPROCESS ---------------- #

def preprocess_bank(df):
    df = df.copy()

    NUM_COLS = [
        "CreditScore","Age","Tenure","Balance","NumOfProducts",
        "EstimatedSalary","Satisfaction Score","Point Earned"
    ]

    # Ensure numeric columns exist
    for c in NUM_COLS:
        if c not in df:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Engineered features
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["LoyaltyScore"] = df["Tenure"] * df["NumOfProducts"]

    BANK_NUM_COLS = NUM_COLS + ["BalanceSalaryRatio", "LoyaltyScore"]
    X_num = assets["bank_scaler"].transform(df[BANK_NUM_COLS])

    # Categorical embeddings
    X_cat = []
    for col, le in assets["bank_encoders"].items():
        if col not in df:
            df[col] = ""
        mapping = {cls: i + 1 for i, cls in enumerate(le.classes_)}
        encoded = df[col].astype(str).map(lambda x: mapping.get(x, 0)).astype(int)
        X_cat.append(encoded.values.reshape(-1, 1))

    return [X_num] + X_cat

# ---------------- TELCO PREPROCESS ---------------- #

def preprocess_telco(df):
    df = df.copy()

    # Ensure all features exist
    for col in assets["telco_features"]:
        if col not in df:
            df[col] = 0

    # Numeric safety
    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Engineered features
    df["avg_charges_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["num_services"] = (
        (df.get("PhoneService", "No") == "Yes").astype(int) +
        (df.get("InternetService", "No") != "No").astype(int)
    )
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)

    df["tenure_bin"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72, 1e9],
        labels=[0,1,2,3,4]
    ).astype(int)

    # Encode categoricals safely
    for col, le in assets["telco_encoders"].items():
        mapping = {cls: i for i, cls in enumerate(le.classes_)}
        df[col] = df[col].astype(str).map(lambda x: mapping.get(x, 0)).astype(int)

    X = df[assets["telco_features"]].astype(float)
    return assets["telco_scaler"].transform(X)

# ---------------- PDF REPORT ---------------- #

def generate_pdf_report(df, dataset):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    y = 800

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "Customer Churn Business Report")
    y -= 40

    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Dataset: {dataset}")
    y -= 20
    c.drawString(50, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    y -= 40
    c.drawString(50, y, f"Average churn probability: {df['churn_probability'].mean():.2%}")

    y -= 30
    for k, v in df["risk"].value_counts().items():
        c.drawString(70, y, f"{k}: {v}")
        y -= 18

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ---------------- UI ---------------- #

st.title("📊 Customer Churn Studio")

dataset = st.selectbox("Dataset", ["Telco", "Bank"])
file = st.file_uploader("Upload CSV", type="csv")
threshold = st.slider("Churn Threshold", 0.0, 1.0, 0.5)

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    if st.button("Run Prediction"):
        try:
            if dataset == "Bank":
                inputs = preprocess_bank(df)
                probs = assets["bank_model"].predict(inputs).ravel()
            else:
                Xs = preprocess_telco(df)
                probs = assets["telco_model"].predict(Xs).ravel()

            df["churn_probability"] = probs
            df["risk"] = pd.cut(probs, [0, .33, .66, 1], labels=["Low","Medium","High"])

            st.success("Prediction successful")
            st.metric("Average churn", f"{probs.mean():.3f}")
            st.dataframe(df.head(20))

            # -------- SHAP (TELCO ONLY) -------- #
            if st.checkbox("Show SHAP Explanation"):
                if dataset == "Telco":
                    explainer = shap.Explainer(assets["telco_model"], Xs)
                    shap_values = explainer(Xs[:1])
                    fig, ax = plt.subplots()
                    shap.plots.bar(shap_values[0], show=False)
                    st.pyplot(fig)
                else:
                    st.info("SHAP not supported for embedding-based bank model.")

            # -------- PDF -------- #
            if st.button("📄 Download Business PDF"):
                pdf = generate_pdf_report(df, dataset)
                st.download_button(
                    "Download PDF",
                    pdf,
                    file_name="churn_report.pdf",
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
