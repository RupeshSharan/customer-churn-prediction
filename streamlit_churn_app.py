import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# ================= CONFIG ================= #

st.set_page_config(
    page_title="Customer Churn Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================= SIDEBAR STYLING ================= #

st.markdown("""
<style>
/* Soft radiant orange sidebar background */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFDAB9 0%, #FFB880 40%, #F0A06A 100%);
}

/* All sidebar text: dark color for readability */
[data-testid="stSidebar"] * {
    color: #1a1a1a !important;
}

/* Sidebar header */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #1a1a1a !important;
    text-shadow: none;
}

/* Sidebar labels */
[data-testid="stSidebar"] label {
    color: #1a1a1a !important;
    font-weight: 600 !important;
}

/* Select boxes and inputs in sidebar */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: rgba(255,255,255,0.85) !important;
    border-radius: 6px;
}

/* Slider thumb color */
[data-testid="stSidebar"] [role="slider"] {
    background-color: #fff !important;
    border: 2px solid #1a1a1a !important;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(r"C:\Users\rupes\Downloads\customer_churn")
MODELS_DIR = BASE_DIR / "models"

# ================= LOAD ASSETS ================= #

@st.cache_resource
def load_assets():
    """Load optimized models and preprocessing artifacts."""
    assets = {}

    # ---- Telco ----
    try:
        assets["telco_ensemble"] = joblib.load(MODELS_DIR / "telco_ensemble_model.pkl")
        assets["telco_xgb"] = joblib.load(MODELS_DIR / "telco_xgb_model.pkl")
        assets["telco_scaler"] = joblib.load(MODELS_DIR / "telco_scaler.pkl")
        assets["telco_encoders"] = joblib.load(MODELS_DIR / "telco_label_encoders.pkl")
        assets["telco_features"] = joblib.load(MODELS_DIR / "telco_feature_order.pkl")
        assets["telco_ready"] = True
    except Exception as e:
        st.warning(f"Telco models not found. Run train_telco_optimized.py first. ({e})")
        assets["telco_ready"] = False

    # ---- Telco Type 2 (balanced) ----
    try:
        assets["telco2_ensemble"] = joblib.load(MODELS_DIR / "telco2_ensemble_model.pkl")
        assets["telco2_xgb"] = joblib.load(MODELS_DIR / "telco2_xgb_model.pkl")
        assets["telco2_scaler"] = joblib.load(MODELS_DIR / "telco2_scaler.pkl")
        assets["telco2_encoders"] = joblib.load(MODELS_DIR / "telco2_label_encoders.pkl")
        assets["telco2_features"] = joblib.load(MODELS_DIR / "telco2_feature_order.pkl")
        assets["telco2_ready"] = True
    except Exception as e:
        assets["telco2_ready"] = False

    # ---- Bank ----
    try:
        assets["bank_ensemble"] = joblib.load(MODELS_DIR / "bank_ensemble_model.pkl")
        assets["bank_xgb"] = joblib.load(MODELS_DIR / "bank_xgb_model.pkl")
        assets["bank_scaler"] = joblib.load(MODELS_DIR / "bank_scaler.pkl")
        assets["bank_encoders"] = joblib.load(MODELS_DIR / "bank_label_encoders.pkl")
        assets["bank_features"] = joblib.load(MODELS_DIR / "bank_feature_order.pkl")
        assets["bank_ready"] = True
    except Exception as e:
        st.warning(f"Bank models not found. Run train_bank_optimized.py first. ({e})")
        assets["bank_ready"] = False

    return assets


assets = load_assets()

# ================= IMPORT EXPLAINER ================= #

import sys
sys.path.insert(0, str(BASE_DIR))
from churn_explainer import explain_churn, preprocess_telco, preprocess_bank

# ================= HELPER: RISK COLOR ================= #

def risk_badge(risk):
    colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    return f"{colors.get(risk, '⚪')} **{risk}**"

# ================= HELPER: SHAP BAR CHART ================= #

def plot_shap_bar(top_reasons, title="What Matters Most for This Customer"):
    """Create a horizontal bar chart showing retention vs risk factors."""
    labels = [r["label"] for r in reversed(top_reasons)]
    values = [r["shap_value"] for r in reversed(top_reasons)]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.5)))
    ax.barh(labels, values, color=colors, edgecolor="none", height=0.6)
    ax.set_xlabel("← Keeping the Customer          Risk of Leaving →")
    ax.set_title(title)
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")

    # Hide numeric tick labels on x-axis (not meaningful to non-technical users)
    ax.set_xticklabels([])

    plt.tight_layout()
    return fig

# ================= PDF REPORT ================= #

def generate_pdf_report(df, dataset):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from io import BytesIO
    from datetime import datetime

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

    y -= 30
    c.drawString(50, y, f"Average churn probability: {df['churn_probability'].mean():.2%}")

    y -= 30
    c.drawString(50, y, "Risk Distribution:")
    y -= 20
    for k, v in df["risk_level"].value_counts().items():
        c.drawString(70, y, f"{k}: {v}")
        y -= 16

    # Add sample explanations
    y -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Sample Churn Explanations (Top 5 High-Risk):")
    y -= 20
    c.setFont("Helvetica", 10)

    high_risk = df[df["risk_level"] == "High"].head(5)
    for idx, row in high_risk.iterrows():
        if y < 80:
            c.showPage()
            y = 800
        text = f"Customer {idx}: {row.get('churn_reason', 'N/A')}"
        # Wrap long text
        while len(text) > 90:
            c.drawString(60, y, text[:90])
            text = text[90:]
            y -= 14
        c.drawString(60, y, text)
        y -= 20

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ================= UI ================= #

st.title("📊 Customer Churn Studio")
st.caption("Optimized ML models with explainable AI — understand *why* customers leave")

st.sidebar.header("⚙️ Configuration")
dataset = st.sidebar.selectbox("Dataset", ["Telco", "Telco Type 2 (Balanced)", "Bank"])
mode = st.sidebar.radio("Mode", ["Batch CSV", "Single Customer"])
threshold = st.sidebar.slider("Churn Threshold", 0.0, 1.0, 0.5)

# Map display name to internal key
DS_KEY_MAP = {
    "Telco": "telco",
    "Telco Type 2 (Balanced)": "telco2",
    "Bank": "bank",
}
ds_key = DS_KEY_MAP[dataset]

# Check model readiness
if not assets.get(f"{ds_key}_ready", False):
    st.error(f"❌ {dataset} models not loaded. Please run the training script first.")
    st.stop()

# Is this a Telco-type dataset?
is_telco = ds_key in ("telco", "telco2")

# ================================================================
# SINGLE CUSTOMER MODE
# ================================================================

if mode == "Single Customer":
    st.subheader(f"🧍 Single Customer — {dataset}")

    if is_telco:
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_str = st.selectbox("Senior Citizen", ["No", "Yes"])
            senior = 1 if senior_str == "Yes" else 0
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
        with col2:
            tenure = st.number_input("Tenure (months)", 0, 100, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        with col3:
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, 2000.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])

        # Additional services
        with st.expander("🔧 Additional Services", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                online_bkp = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            with c2:
                dev_prot = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
                tech_sup = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            with c3:
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                streaming_mov = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

        user_data = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple_lines, "InternetService": internet,
            "OnlineSecurity": online_sec, "OnlineBackup": online_bkp,
            "DeviceProtection": dev_prot, "TechSupport": tech_sup,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_mov,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

    else:  # Bank
        col1, col2, col3 = st.columns(3)
        with col1:
            credit_score = st.number_input("Credit Score", 300, 850, 650)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender_b = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", 18, 100, 35)
        with col2:
            tenure_b = st.number_input("Tenure (years)", 0, 10, 5)
            balance = st.number_input("Account Balance ($)", 0.0, 300000.0, 50000.0)
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
            has_card_str = st.selectbox("Has Credit Card", ["Yes", "No"])
            has_card = 1 if has_card_str == "Yes" else 0
        with col3:
            is_active_str = st.selectbox("Is Active Member", ["Yes", "No"])
            is_active = 1 if is_active_str == "Yes" else 0
            salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 80000.0)
            complain_str = st.selectbox("Has Complained", ["No", "Yes"])
            complain = 1 if complain_str == "Yes" else 0
            satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
            card_type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"])
            points = st.number_input("Points Earned", 0, 1000, 400)

        user_data = {
            "CreditScore": credit_score, "Geography": geography, "Gender": gender_b,
            "Age": age, "Tenure": tenure_b, "Balance": balance,
            "NumOfProducts": num_products, "HasCrCard": has_card,
            "IsActiveMember": is_active, "EstimatedSalary": salary,
            "Complain": complain, "Satisfaction Score": satisfaction,
            "Card Type": card_type, "Point Earned": points,
        }

    # --- Predict & Explain ---
    if st.button("🔮 Predict & Explain", type="primary", use_container_width=True):
        df_input = pd.DataFrame([user_data])
        results = explain_churn(df_input, dataset_type=ds_key, top_n=5)
        result = results[0]

        # Display results
        st.divider()
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Churn Probability", f"{result['probability']:.1%}")
        with m2:
            st.markdown(f"### Risk Level: {risk_badge(result['risk_level'])}")

        # Explanation — heading changes based on risk level
        st.divider()
        if result["risk_level"] == "Low":
            st.subheader("✅ Why This Customer Is Likely to Stay")
        elif result["risk_level"] == "Medium":
            st.subheader("⚠️ This Customer Needs Attention")
        else:
            st.subheader("🚨 Why This Customer May Leave")

        st.info(result["explanation"])

        # Visual chart
        if result["top_reasons"]:
            fig = plot_shap_bar(result["top_reasons"])
            st.pyplot(fig)

        # Detailed breakdown in simple language
        with st.expander("📋 See What Each Factor Does"):
            for r in result["top_reasons"]:
                if r["direction"] == "risk_factor":
                    st.markdown(
                        f"🔴 **{r['label']}** — this is pushing the customer toward leaving"
                    )
                else:
                    st.markdown(
                        f"🟢 **{r['label']}** — this is helping keep the customer"
                    )

# ================================================================
# BATCH CSV MODE
# ================================================================

else:
    st.subheader(f"📂 Batch CSV — {dataset}")
    file = st.file_uploader("Upload CSV", type="csv")

    if file:
        df = pd.read_csv(file)
        st.write(f"**Uploaded:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Run Predictions & Explain", type="primary", use_container_width=True):
            with st.spinner("Running predictions and generating explanations..."):
                # Drop target column if present
                target_col = "Churn" if is_telco else "Exited"
                df_input = df.drop(columns=[target_col, "customerID"], errors="ignore")

                results = explain_churn(df_input, dataset_type=ds_key, top_n=3)

                df["churn_probability"] = [r["probability"] for r in results]
                df["risk_level"] = [r["risk_level"] for r in results]
                df["churn_reason"] = [r["explanation"] for r in results]

            # Summary metrics
            st.divider()
            st.subheader("📈 Results Summary")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Customers", len(df))
            c2.metric("Avg Churn Prob", f"{df['churn_probability'].mean():.1%}")
            high_risk_count = (df["risk_level"] == "High").sum()
            c3.metric("High Risk", f"{high_risk_count} ({high_risk_count/len(df)*100:.1f}%)")
            c4.metric("Low Risk", f"{(df['risk_level'] == 'Low').sum()}")

            # Risk distribution chart
            risk_counts = df["risk_level"].value_counts()
            fig_risk, ax_risk = plt.subplots(figsize=(5, 3))
            colors_map = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
            bars = ax_risk.bar(
                risk_counts.index,
                risk_counts.values,
                color=[colors_map.get(k, "#95a5a6") for k in risk_counts.index],
                edgecolor="none",
            )
            ax_risk.set_ylabel("Count")
            ax_risk.set_title("Risk Distribution")
            for bar, val in zip(bars, risk_counts.values):
                ax_risk.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                           str(val), ha="center", fontsize=11, fontweight="bold")
            plt.tight_layout()

            col_chart, col_table = st.columns([1, 2])
            with col_chart:
                st.pyplot(fig_risk)
            with col_table:
                st.dataframe(
                    df[["churn_probability", "risk_level", "churn_reason"]].head(20),
                    use_container_width=True,
                )

            # Full results table
            st.divider()
            st.subheader("📊 Full Results")
            st.dataframe(df, use_container_width=True)

            # Show top high-risk explanations
            st.divider()
            st.subheader("🔴 Top High-Risk Customers — Why They Left")
            high_risk_df = df[df["risk_level"] == "High"].nlargest(5, "churn_probability")
            for idx, row in high_risk_df.iterrows():
                with st.container():
                    st.markdown(
                        f"**Customer #{idx}** — "
                        f"Probability: `{row['churn_probability']:.1%}` | "
                        f"Risk: {risk_badge(row['risk_level'])}"
                    )
                    st.caption(row["churn_reason"])
                    st.divider()

            # Download buttons
            col_csv, col_pdf = st.columns(2)
            with col_csv:
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Results CSV", csv_data,
                    file_name="churn_predictions.csv", mime="text/csv",
                    use_container_width=True,
                )
            with col_pdf:
                if st.button("📄 Generate PDF Report", use_container_width=True):
                    pdf = generate_pdf_report(df, dataset)
                    st.download_button(
                        "Download PDF", pdf,
                        file_name="churn_report.pdf",
                        use_container_width=True,
                    )
