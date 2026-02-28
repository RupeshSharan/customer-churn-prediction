"""
Optimized Bank Customer Churn Training Pipeline
================================================
Uses XGBoost + Random Forest ensemble with Optuna hyperparameter tuning,
SMOTE for class imbalance, and SHAP-ready model export.
"""

import warnings
warnings.filterwarnings("ignore")

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

from imblearn.over_sampling import SMOTE

import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\rupes\Downloads\customer_churn")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_OPTUNA_TRIALS = 40
TARGET_COL = "Exited"

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("BANK CHURN — OPTIMIZED TRAINING PIPELINE")
print("=" * 60)

df = pd.read_csv(DATA_DIR / "Customer-Churn-Records-bank.csv")
print(f"\nDataset shape: {df.shape}")
print(f"Churn (Exited) distribution:\n{df[TARGET_COL].value_counts()}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

print("\n[1/6] Feature engineering...")

# Drop useless columns
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True, errors="ignore")

# Existing features
df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
df["LoyaltyScore"] = df["Tenure"] * df["NumOfProducts"]

# NEW features
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 25, 35, 45, 55, 200],
    labels=[0, 1, 2, 3, 4],
).astype(int)

# Credit score category
df["CreditCategory"] = pd.cut(
    df["CreditScore"],
    bins=[0, 400, 550, 700, 850],
    labels=[0, 1, 2, 3],
).astype(int)

# Products per tenure year
df["ProductsPerTenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)

# Balance engagement: has balance? is active?
df["BalanceEngagement"] = df["Balance"].apply(lambda x: 1 if x > 0 else 0) * df["IsActiveMember"]

# Satisfaction per product
df["SatPerProduct"] = df["Satisfaction Score"] / (df["NumOfProducts"] + 1)

# Points per tenure
df["PointsPerTenure"] = df["Point Earned"] / (df["Tenure"] + 1)

# High value customer flag
df["HighValue"] = (
    (df["Balance"] > df["Balance"].median())
    & (df["NumOfProducts"] >= 2)
).astype(int)

print(f"  Final feature count: {df.shape[1] - 1}")

# ============================================================
# 3. ENCODE & SPLIT
# ============================================================

print("[2/6] Encoding & splitting...")

y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

# Identify categorical columns
cat_cols = X.select_dtypes(include="object").columns.tolist()

bank_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    bank_encoders[col] = le

BANK_FEATURE_ORDER = X.columns.tolist()

# Scale
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
print(f"  Features: {X_train.shape[1]}")

# ============================================================
# 4. SMOTE FOR CLASS IMBALANCE
# ============================================================

print("[3/6] Applying SMOTE for class imbalance...")

smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"  After SMOTE — Train: {X_train_res.shape[0]} (balanced)")

# ============================================================
# 5. OPTUNA HYPERPARAMETER OPTIMIZATION
# ============================================================

print("[4/6] Running Optuna hyperparameter search (XGBoost)...")


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
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
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
        model, X_train_res, y_train_res, cv=cv, scoring="roc_auc", n_jobs=-1
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

print("[5/6] Training final ensemble (XGBoost + RandomForest)...")

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

ensemble.fit(X_train_res, y_train_res)

# Standalone XGBoost for SHAP
xgb_standalone = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    verbosity=0,
)
xgb_standalone.fit(X_train_res, y_train_res)

# ============================================================
# 7. EVALUATE
# ============================================================

print("[6/6] Evaluating on test set...\n")

y_pred_proba_ens = ensemble.predict_proba(X_test)[:, 1]
y_pred_ens = (y_pred_proba_ens >= 0.5).astype(int)

y_pred_proba_xgb = xgb_standalone.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_pred_proba_xgb >= 0.5).astype(int)

print("=" * 60)
print("ENSEMBLE (XGBoost + RandomForest) RESULTS:")
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
# 8. SAVE ARTIFACTS
# ============================================================

print("\nSaving model artifacts...")

joblib.dump(ensemble, MODELS_DIR / "bank_ensemble_model.pkl")
joblib.dump(xgb_standalone, MODELS_DIR / "bank_xgb_model.pkl")
joblib.dump(scaler, MODELS_DIR / "bank_scaler.pkl")
joblib.dump(bank_encoders, MODELS_DIR / "bank_label_encoders.pkl")
joblib.dump(BANK_FEATURE_ORDER, MODELS_DIR / "bank_feature_order.pkl")

print("  ✓ bank_ensemble_model.pkl")
print("  ✓ bank_xgb_model.pkl  (for SHAP explanations)")
print("  ✓ bank_scaler.pkl")
print("  ✓ bank_label_encoders.pkl")
print("  ✓ bank_feature_order.pkl")
print("\n✅ Bank training complete!")
