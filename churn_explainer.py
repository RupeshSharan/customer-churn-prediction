"""
Churn Explainer Module
======================
Generates human-readable explanations for why a customer churned,
using SHAP TreeExplainer on the optimized XGBoost model.
"""

import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# ============================================================
# FEATURE NAME → HUMAN-READABLE MAPPING
# ============================================================

TELCO_FEATURE_LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior Citizen status",
    "Partner": "Partner status",
    "Dependents": "Dependents",
    "tenure": "how long they have been with the company",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Phone Lines",
    "InternetService": "Internet Service type",
    "OnlineSecurity": "Online Security service",
    "OnlineBackup": "Online Backup service",
    "DeviceProtection": "Device Protection plan",
    "TechSupport": "Tech Support access",
    "StreamingTV": "Streaming TV subscription",
    "StreamingMovies": "Streaming Movies subscription",
    "Contract": "Contract type",
    "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Method",
    "MonthlyCharges": "Monthly Charges amount",
    "TotalCharges": "Total amount paid so far",
    "avg_charges_per_month": "average monthly spending",
    "num_services": "number of services used",
    "is_new_customer": "being a new customer",
    "tenure_bin": "customer loyalty stage",
    "tenure_x_monthly": "time spent combined with monthly spending",
    "charge_ratio": "how much of the total bill is charged each month",
    "high_charge_short_tenure": "high charges despite being a newer customer",
}

BANK_FEATURE_LABELS = {
    "CreditScore": "Credit Score",
    "Geography": "Country or Region",
    "Gender": "Gender",
    "Age": "Age",
    "Tenure": "how long they have been with the bank",
    "Balance": "Account Balance",
    "NumOfProducts": "number of banking products used",
    "HasCrCard": "whether they have a Credit Card",
    "IsActiveMember": "whether they actively use the account",
    "EstimatedSalary": "Estimated Salary",
    "Complain": "whether they filed a complaint",
    "Satisfaction Score": "Satisfaction Score",
    "Card Type": "Card Type",
    "Point Earned": "Reward Points earned",
    "BalanceSalaryRatio": "balance compared to their salary",
    "LoyaltyScore": "overall loyalty score",
    "AgeGroup": "Age Group",
    "CreditCategory": "Credit Score range",
    "ProductsPerTenure": "how many products they picked up each year",
    "BalanceEngagement": "whether they have savings and actively use the account",
    "SatPerProduct": "satisfaction level per product used",
    "PointsPerTenure": "reward points earned per year",
    "HighValue": "whether they are a high-value customer",
}

# Reverse mappings for categorical decode (used in detailed explanations)
TELCO_CAT_DECODE = {
    "Contract": {0: "Month-to-month", 1: "One year", 2: "Two year"},
    "InternetService": {0: "DSL", 1: "Fiber optic", 2: "No"},
    "PaymentMethod": {
        0: "Bank transfer (automatic)",
        1: "Credit card (automatic)",
        2: "Electronic check",
        3: "Mailed check",
    },
}

BANK_CAT_DECODE = {
    "Geography": {0: "France", 1: "Germany", 2: "Spain"},
    "Gender": {0: "Female", 1: "Male"},
}


# ============================================================
# LOAD MODELS (lazy)
# ============================================================

_cache = {}


def _load(key, path):
    if key not in _cache:
        _cache[key] = joblib.load(path)
    return _cache[key]


def get_telco_xgb():
    return _load("telco_xgb", MODELS_DIR / "telco_xgb_model.pkl")

def get_telco_ensemble():
    return _load("telco_ens", MODELS_DIR / "telco_ensemble_model.pkl")

def get_telco_scaler():
    return _load("telco_scaler", MODELS_DIR / "telco_scaler.pkl")

def get_telco_encoders():
    return _load("telco_enc", MODELS_DIR / "telco_label_encoders.pkl")

def get_telco_features():
    return _load("telco_feat", MODELS_DIR / "telco_feature_order.pkl")

def get_telco2_xgb():
    return _load("telco2_xgb", MODELS_DIR / "telco2_xgb_model.pkl")

def get_telco2_ensemble():
    return _load("telco2_ens", MODELS_DIR / "telco2_ensemble_model.pkl")

def get_telco2_scaler():
    return _load("telco2_scaler", MODELS_DIR / "telco2_scaler.pkl")

def get_telco2_encoders():
    return _load("telco2_enc", MODELS_DIR / "telco2_label_encoders.pkl")

def get_telco2_features():
    return _load("telco2_feat", MODELS_DIR / "telco2_feature_order.pkl")

def get_bank_xgb():
    return _load("bank_xgb", MODELS_DIR / "bank_xgb_model.pkl")

def get_bank_ensemble():
    return _load("bank_ens", MODELS_DIR / "bank_ensemble_model.pkl")

def get_bank_scaler():
    return _load("bank_scaler", MODELS_DIR / "bank_scaler.pkl")

def get_bank_encoders():
    return _load("bank_enc", MODELS_DIR / "bank_label_encoders.pkl")

def get_bank_features():
    return _load("bank_feat", MODELS_DIR / "bank_feature_order.pkl")


# ============================================================
# PREPROCESSING (mirrors training scripts)
# ============================================================

def preprocess_telco(df, encoders_fn=None, scaler_fn=None, features_fn=None):
    """Preprocess a Telco DataFrame → scaled numpy array."""
    df = df.copy()
    encoders = (encoders_fn or get_telco_encoders)()
    scaler = (scaler_fn or get_telco_scaler)()
    features = (features_fn or get_telco_features)()

    # Numeric coercion
    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Engineered features
    df["avg_charges_per_month"] = df.get("TotalCharges", 0) / (df.get("tenure", 0) + 1)
    df["num_services"] = (
        (df.get("PhoneService", "No") == "Yes").astype(int)
        + (df.get("InternetService", "No") != "No").astype(int)
    )
    df["is_new_customer"] = (df.get("tenure", 0) <= 6).astype(int)

    # New interaction features
    df["tenure_x_monthly"] = df.get("tenure", 0) * df.get("MonthlyCharges", 0)
    df["charge_ratio"] = df.get("MonthlyCharges", 0) / (df.get("TotalCharges", 0) + 1)

    mc_median = 70.35  # approximate median from training data
    t_median = 29.0
    df["high_charge_short_tenure"] = (
        (df.get("MonthlyCharges", 0) > mc_median)
        & (df.get("tenure", 0) < t_median)
    ).astype(int)

    tenure_bins = [-1, 3, 12, 24, 48, 72, 200]
    tenure_labels = [0, 1, 2, 3, 4, 5]
    df["tenure_bin"] = pd.cut(
        df.get("tenure", pd.Series([0])),
        bins=tenure_bins, labels=tenure_labels
    ).fillna(0).astype(int)

    # Encode categoricals
    for col, le in encoders.items():
        if col in df.columns:
            mapping = {cls: i for i, cls in enumerate(le.classes_)}
            df[col] = df[col].astype(str).map(lambda x, m=mapping: m.get(x, 0)).astype(int)

    # Ensure all features exist
    for f in features:
        if f not in df.columns:
            df[f] = 0

    X = df[features].astype(float)
    X_scaled = scaler.transform(X)
    return X_scaled, features


def preprocess_bank(df):
    """Preprocess a Bank DataFrame → scaled numpy array."""
    df = df.copy()
    encoders = get_bank_encoders()
    scaler = get_bank_scaler()
    features = get_bank_features()

    # Drop ID columns
    df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True, errors="ignore")

    # Engineered features
    df["BalanceSalaryRatio"] = df.get("Balance", 0) / (df.get("EstimatedSalary", 0) + 1)
    df["LoyaltyScore"] = df.get("Tenure", 0) * df.get("NumOfProducts", 0)
    df["AgeGroup"] = pd.cut(
        df.get("Age", pd.Series([30])),
        bins=[0, 25, 35, 45, 55, 200], labels=[0, 1, 2, 3, 4]
    ).astype(int)
    df["CreditCategory"] = pd.cut(
        df.get("CreditScore", pd.Series([650])),
        bins=[0, 400, 550, 700, 850], labels=[0, 1, 2, 3]
    ).astype(int)
    df["ProductsPerTenure"] = df.get("NumOfProducts", 0) / (df.get("Tenure", 0) + 1)
    df["BalanceEngagement"] = (
        df.get("Balance", pd.Series([0])).apply(lambda x: 1 if x > 0 else 0)
        * df.get("IsActiveMember", 0)
    )
    df["SatPerProduct"] = df.get("Satisfaction Score", 0) / (df.get("NumOfProducts", 0) + 1)
    df["PointsPerTenure"] = df.get("Point Earned", 0) / (df.get("Tenure", 0) + 1)
    df["HighValue"] = (
        (df.get("Balance", 0) > 76000)  # approx median
        & (df.get("NumOfProducts", 0) >= 2)
    ).astype(int)

    # Encode categoricals
    for col, le in encoders.items():
        if col in df.columns:
            mapping = {cls: i for i, cls in enumerate(le.classes_)}
            df[col] = df[col].astype(str).map(lambda x, m=mapping: m.get(x, 0)).astype(int)

    # Ensure all features exist
    for f in features:
        if f not in df.columns:
            df[f] = 0

    X = df[features].astype(float)
    X_scaled = scaler.transform(X)
    return X_scaled, features


# ============================================================
# SHAP EXPLANATION
# ============================================================

def _get_shap_values(model, X_scaled):
    """Compute SHAP values using TreeExplainer (fast & exact for XGBoost)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = float(base_value[0])
    return shap_values, base_value


def _build_explanation(shap_vals_row, feature_names, feature_labels, risk_level, top_n=5):
    """
    Build a plain-English explanation from SHAP values for one customer.
    The language changes based on risk level so it always makes sense.
    """
    # Pair feature names with their SHAP values
    pairs = list(zip(feature_names, shap_vals_row))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)

    top_reasons = []
    for feat_name, shap_val in pairs[:top_n]:
        label = feature_labels.get(feat_name, feat_name)
        abs_val = abs(shap_val)

        # Use simple language: "risk factor" vs "keeping the customer"
        if shap_val > 0:
            direction = "risk_factor"
            simple_direction = "pushes toward leaving"
        else:
            direction = "retention_factor"
            simple_direction = "helps keep the customer"

        top_reasons.append({
            "feature": feat_name,
            "label": label,
            "shap_value": round(float(shap_val), 4),
            "direction": direction,
            "simple_direction": simple_direction,
            "impact": round(abs_val, 4),
        })

    # Separate into risk factors and retention factors
    risk_factors = [r for r in top_reasons if r["direction"] == "risk_factor"]
    retention_factors = [r for r in top_reasons if r["direction"] == "retention_factor"]

    # Build context-aware plain-English explanation
    sentences = []

    if risk_level == "Low":
        # --- LOW RISK: Focus on what's keeping them ---
        if retention_factors:
            keep_phrases = [r["label"] for r in retention_factors[:3]]
            if len(keep_phrases) == 1:
                sentences.append(
                    f"This customer is likely to stay mainly because of their {keep_phrases[0]}."
                )
            else:
                joined = ", ".join(keep_phrases[:-1]) + f" and {keep_phrases[-1]}"
                sentences.append(
                    f"This customer is likely to stay because of their {joined}."
                )
        if risk_factors:
            warn_phrases = [r["label"] for r in risk_factors[:2]]
            joined = " and ".join(warn_phrases)
            sentences.append(
                f"However, keep an eye on their {joined} — "
                f"if these are not managed well, the customer could become unhappy over time."
            )

    elif risk_level == "Medium":
        # --- MEDIUM RISK: Balanced view ---
        if risk_factors:
            risk_phrases = [r["label"] for r in risk_factors[:2]]
            joined = " and ".join(risk_phrases)
            sentences.append(
                f"This customer shows some signs of leaving, mainly due to their {joined}."
            )
        if retention_factors:
            keep_phrases = [r["label"] for r in retention_factors[:2]]
            joined = " and ".join(keep_phrases)
            sentences.append(
                f"On the positive side, their {joined} are helping them stay. "
                f"Taking action on the risk areas now could prevent this customer from leaving."
            )

    else:
        # --- HIGH RISK: Focus on why they might leave ---
        if risk_factors:
            risk_phrases = [r["label"] for r in risk_factors[:3]]
            if len(risk_phrases) == 1:
                sentences.append(
                    f"This customer is at high risk of leaving because of their {risk_phrases[0]}."
                )
            else:
                joined = ", ".join(risk_phrases[:-1]) + f" and {risk_phrases[-1]}"
                sentences.append(
                    f"This customer is at high risk of leaving because of their {joined}."
                )
            sentences.append(
                "The company should reach out to this customer immediately "
                "and address these concerns to prevent them from leaving."
            )
        if retention_factors:
            keep_phrases = [r["label"] for r in retention_factors[:2]]
            joined = " and ".join(keep_phrases)
            sentences.append(
                f"Their {joined} are the only things helping retain this customer right now."
            )

    if not sentences:
        sentences.append("This customer's profile does not show any strong signals in either direction.")

    explanation = " ".join(sentences)

    return {
        "top_reasons": top_reasons,
        "explanation": explanation,
    }


# ============================================================
# PUBLIC API
# ============================================================

def explain_churn(customer_df, dataset_type="telco", top_n=5):
    """
    Explain churn prediction for one or more customers.

    Args:
        customer_df: DataFrame with raw customer data (1 or more rows).
        dataset_type: "telco" or "bank".
        top_n: Number of top features to include in explanation.

    Returns:
        list of dicts, one per customer, each with:
            - probability: float
            - risk_level: "Low" / "Medium" / "High"
            - top_reasons: list of feature impact dicts
            - explanation: human-readable sentence
    """
    dataset_type = dataset_type.lower().strip()

    if dataset_type in ("telco", "telco2"):
        X_scaled, feature_names = preprocess_telco(customer_df,
            encoders_fn=get_telco2_encoders if dataset_type == "telco2" else get_telco_encoders,
            scaler_fn=get_telco2_scaler if dataset_type == "telco2" else get_telco_scaler,
            features_fn=get_telco2_features if dataset_type == "telco2" else get_telco_features,
        )
        xgb_model = get_telco2_xgb() if dataset_type == "telco2" else get_telco_xgb()
        ensemble = get_telco2_ensemble() if dataset_type == "telco2" else get_telco_ensemble()
        feature_labels = TELCO_FEATURE_LABELS
    elif dataset_type == "bank":
        X_scaled, feature_names = preprocess_bank(customer_df)
        xgb_model = get_bank_xgb()
        ensemble = get_bank_ensemble()
        feature_labels = BANK_FEATURE_LABELS
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'telco', 'telco2', or 'bank'.")

    # Get predictions from ensemble
    probas = ensemble.predict_proba(X_scaled)[:, 1]

    # Get SHAP values from XGBoost (TreeExplainer is exact & fast)
    shap_values, base_value = _get_shap_values(xgb_model, X_scaled)

    results = []
    for i in range(len(customer_df)):
        prob = float(probas[i])
        risk = "Low" if prob < 0.33 else "Medium" if prob < 0.66 else "High"

        explanation_data = _build_explanation(
            shap_values[i], feature_names, feature_labels,
            risk_level=risk, top_n=top_n
        )

        results.append({
            "probability": round(prob, 4),
            "risk_level": risk,
            "base_value": round(base_value, 4),
            "top_reasons": explanation_data["top_reasons"],
            "explanation": explanation_data["explanation"],
        })

    return results


def get_batch_explanations(customer_df, dataset_type="telco", top_n=3):
    """
    Convenience function for batch processing.
    Returns a DataFrame column with explanation sentences.
    """
    results = explain_churn(customer_df, dataset_type, top_n)
    return [r["explanation"] for r in results]


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("CHURN EXPLAINER — QUICK TEST")
    print("=" * 60)

    # Test with Telco
    try:
        telco_df = pd.read_csv(BASE_DIR / "data" / "telco_churn_processed_for_modeling.csv")
        sample = telco_df.head(3).drop(columns=["Churn"], errors="ignore")
        results = explain_churn(sample, "telco")
        print("\n--- TELCO (first 3 customers) ---")
        for i, r in enumerate(results):
            print(f"\nCustomer {i+1}:")
            print(f"  Probability: {r['probability']:.2%}")
            print(f"  Risk Level:  {r['risk_level']}")
            print(f"  Explanation: {r['explanation']}")
    except Exception as e:
        print(f"Telco test failed: {e}")

    # Test with Bank
    try:
        bank_df = pd.read_csv(BASE_DIR / "data" / "Customer-Churn-Records-bank.csv")
        sample = bank_df.head(3).drop(columns=["Exited"], errors="ignore")
        results = explain_churn(sample, "bank")
        print("\n\n--- BANK (first 3 customers) ---")
        for i, r in enumerate(results):
            print(f"\nCustomer {i+1}:")
            print(f"  Probability: {r['probability']:.2%}")
            print(f"  Risk Level:  {r['risk_level']}")
            print(f"  Explanation: {r['explanation']}")
    except Exception as e:
        print(f"Bank test failed: {e}")

    print("\n✅ Explainer test complete!")
