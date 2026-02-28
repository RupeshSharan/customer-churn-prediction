"""
Telco Type 2 — Balanced Dataset Training Pipeline
===================================================
Downloads the raw Telco churn dataset from Kaggle, physically undersamples
the majority class (No-churn) to match the minority class (Yes-churn = 1869),
then trains an optimized XGBoost + RF ensemble on this perfectly balanced data.
"""

import warnings
warnings.filterwarnings("ignore")

import kagglehub
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\rupes\Downloads\customer_churn")
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_OPTUNA_TRIALS = 40
TARGET_COL = "Churn"

# ============================================================
# 1. DOWNLOAD DATA FROM KAGGLE
# ============================================================

print("=" * 60)
print("TELCO TYPE 2 — BALANCED DATASET TRAINING")
print("=" * 60)

print("\n[1/7] Downloading dataset from Kaggle...")
dataset_path = kagglehub.dataset_download("blastchar/telco-customer-churn")
print(f"  Downloaded to: {dataset_path}")

# Find the CSV file
csv_path = None
for f in Path(dataset_path).rglob("*.csv"):
    csv_path = f
    break

if csv_path is None:
    raise FileNotFoundError(f"No CSV found in {dataset_path}")

df = pd.read_csv(csv_path)
print(f"  Raw dataset shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ============================================================
# 2. BALANCE THE DATASET (UNDERSAMPLE)
# ============================================================

print("\n[2/7] Balancing the dataset by undersampling...")

churn_yes = df[df[TARGET_COL] == "Yes"]
churn_no = df[df[TARGET_COL] == "No"]

print(f"  Before balancing:")
print(f"    Churn = Yes: {len(churn_yes)}")
print(f"    Churn = No:  {len(churn_no)}")

# Randomly sample from No-churn to match Yes-churn count
churn_no_sampled = churn_no.sample(n=len(churn_yes), random_state=RANDOM_STATE)

# Combine into balanced dataset
df_balanced = pd.concat([churn_yes, churn_no_sampled], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"  After balancing:")
print(f"    Churn = Yes: {(df_balanced[TARGET_COL] == 'Yes').sum()}")
print(f"    Churn = No:  {(df_balanced[TARGET_COL] == 'No').sum()}")
print(f"    Total:       {len(df_balanced)}")

# ============================================================
# 3. PREPROCESSING & FEATURE ENGINEERING
# ============================================================

print("\n[3/7] Preprocessing & feature engineering...")

# Drop customerID
df_balanced.drop(columns=["customerID"], inplace=True, errors="ignore")

# Fix TotalCharges (some have spaces)
df_balanced["TotalCharges"] = pd.to_numeric(df_balanced["TotalCharges"], errors="coerce").fillna(0)

# Engineered features
df_balanced["avg_charges_per_month"] = df_balanced["TotalCharges"] / (df_balanced["tenure"] + 1)
df_balanced["num_services"] = (
    (df_balanced["PhoneService"] == "Yes").astype(int)
    + (df_balanced["InternetService"] != "No").astype(int)
)
df_balanced["is_new_customer"] = (df_balanced["tenure"] <= 6).astype(int)

# Interaction features
df_balanced["tenure_x_monthly"] = df_balanced["tenure"] * df_balanced["MonthlyCharges"]
df_balanced["charge_ratio"] = df_balanced["MonthlyCharges"] / (df_balanced["TotalCharges"] + 1)

mc_median = df_balanced["MonthlyCharges"].median()
t_median = df_balanced["tenure"].median()
df_balanced["high_charge_short_tenure"] = (
    (df_balanced["MonthlyCharges"] > mc_median)
    & (df_balanced["tenure"] < t_median)
).astype(int)

# Tenure bins
tenure_bins = [-1, 3, 12, 24, 48, 72, 200]
tenure_labels = [0, 1, 2, 3, 4, 5]
df_balanced["tenure_bin"] = pd.cut(
    df_balanced["tenure"], bins=tenure_bins, labels=tenure_labels
).fillna(0).astype(int)

# ============================================================
# 4. ENCODE & SPLIT
# ============================================================

print("[4/7] Encoding & splitting...")

y = df_balanced[TARGET_COL].map({"Yes": 1, "No": 0})
X = df_balanced.drop(columns=[TARGET_COL])

# Identify and encode categoricals
cat_cols = X.select_dtypes(include="object").columns.tolist()

telco2_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    telco2_encoders[col] = le

TELCO2_FEATURE_ORDER = X.columns.tolist()
print(f"  Features ({len(TELCO2_FEATURE_ORDER)}): {TELCO2_FEATURE_ORDER}")

# Scale
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train/test split (no SMOTE needed — already balanced!)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
print(f"  Train balance: Yes={y_train.sum()}, No={(y_train==0).sum()}")

# ============================================================
# 5. OPTUNA HYPERPARAMETER OPTIMIZATION
# ============================================================

print("[5/7] Running Optuna hyperparameter search (XGBoost)...")


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    model = xgb.XGBClassifier(
        **params,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    return scores.mean()


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

best_params = study.best_params
print(f"  Best ROC-AUC (CV): {study.best_value:.4f}")
print(f"  Best params: {best_params}")

# ============================================================
# 6. TRAIN FINAL ENSEMBLE
# ============================================================

print("[6/7] Training final ensemble (XGBoost + RandomForest)...")

xgb_best = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    verbosity=0,
)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=best_params.get("max_depth", 7),
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

ensemble = VotingClassifier(
    estimators=[("xgb", xgb_best), ("rf", rf)],
    voting="soft",
    weights=[2, 1],
)

ensemble.fit(X_train, y_train)

# Standalone XGBoost for SHAP
xgb_standalone = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    verbosity=0,
)
xgb_standalone.fit(X_train, y_train)

# ============================================================
# 7. EVALUATE
# ============================================================

print("[7/7] Evaluating on test set...\n")

y_pred_proba_ens = ensemble.predict_proba(X_test)[:, 1]
y_pred_ens = (y_pred_proba_ens >= 0.5).astype(int)

y_pred_proba_xgb = xgb_standalone.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_pred_proba_xgb >= 0.5).astype(int)

print("=" * 60)
print("ENSEMBLE (XGBoost + RandomForest) — BALANCED DATA RESULTS:")
print("=" * 60)
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_ens):.4f}")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_ens):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_ens):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_ens):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred_ens):.4f}")
print(f"\n{classification_report(y_test, y_pred_ens)}")

print("\n" + "=" * 60)
print("STANDALONE XGBoost (for SHAP) RESULTS:")
print("=" * 60)
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_xgb):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred_xgb):.4f}")

# ============================================================
# 8. SAVE ARTIFACTS (with "telco2" prefix)
# ============================================================

print("\nSaving model artifacts (Telco Type 2)...")

joblib.dump(ensemble, MODELS_DIR / "telco2_ensemble_model.pkl")
joblib.dump(xgb_standalone, MODELS_DIR / "telco2_xgb_model.pkl")
joblib.dump(scaler, MODELS_DIR / "telco2_scaler.pkl")
joblib.dump(telco2_encoders, MODELS_DIR / "telco2_label_encoders.pkl")
joblib.dump(TELCO2_FEATURE_ORDER, MODELS_DIR / "telco2_feature_order.pkl")

print("  ✓ telco2_ensemble_model.pkl")
print("  ✓ telco2_xgb_model.pkl  (for SHAP explanations)")
print("  ✓ telco2_scaler.pkl")
print("  ✓ telco2_label_encoders.pkl")
print("  ✓ telco2_feature_order.pkl")
print("\n✅ Telco Type 2 training complete!")
print("   Dataset: physically balanced (undersampled No-churn to match Yes-churn)")
print(f"   Training samples: {len(X_train)} ({len(df_balanced)} total)")
