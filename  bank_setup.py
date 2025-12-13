import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

bank_csv_path = r"C:\Users\rupes\Downloads\customer_churn\bank_data.csv"
models_dir = Path(r"C:\Users\rupes\Downloads\customer_churn\models")
models_dir.mkdir(parents=True, exist_ok=True)

num_cols = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary','Satisfaction Score','Point Earned']
cat_cols = ['CreditCategory','Geography','Gender','Card Type']

if not Path(bank_csv_path).exists():
    print("ERROR: bank CSV not found at:", bank_csv_path)
    sys.exit(1)

df = pd.read_csv(bank_csv_path)
missing_num = [c for c in num_cols if c not in df.columns]
missing_cat = [c for c in cat_cols if c not in df.columns]
if missing_num or missing_cat:
    print("ERROR: CSV missing columns.")
    print("missing numeric:", missing_num)
    print("missing categorical:", missing_cat)
    sys.exit(1)

le_dict = {}
for c in cat_cols:
    le = LabelEncoder()
    vals = df[c].astype(str).values
    le.fit(vals)
    le_dict[c] = le
joblib.dump(le_dict, models_dir / "bank_label_encoders.pkl")
cardinalities = {c: int(len(le_dict[c].classes_)+1) for c in cat_cols}
joblib.dump(cardinalities, models_dir / "bank_cardinalities.pkl")

scaler = StandardScaler()
scaler.fit(df[num_cols].astype(float).values)
joblib.dump(scaler, models_dir / "bank_scaler.pkl")

candidates = list(models_dir.glob("bank_churn_model_emb*.keras")) + list(models_dir.glob("bank_churn_model*.keras")) + list(models_dir.glob("bank_churn_model_emb*.h5")) + list(models_dir.glob("bank_churn_model*.h5"))
if not candidates:
    print("ERROR: No bank model file found in models dir. Place bank_churn_model_emb.keras or bank_churn_model_best.h5 into:", models_dir)
    sys.exit(1)
model_path = str(candidates[0])
print("Using model:", model_path)
try:
    model = load_model(model_path)
except Exception as e:
    print("ERROR loading model:", e)
    sys.exit(1)

def df_to_bank_inputs(df_row):
    df2 = df_row.copy()
    df2['BalanceSalaryRatio'] = df2['Balance'] / (df2['EstimatedSalary'] + 1)
    df2['LoyaltyScore'] = df2['Tenure'] * df2['NumOfProducts']
    Xnum = scaler.transform(df2[num_cols].astype(float).values)
    Xcat_list = []
    for c in cat_cols:
        arr = le_dict[c].transform(df2[c].astype(str).values) + 1
        Xcat_list.append(arr.astype('int32'))
    inputs = [Xnum] + [Xcat_list[i] for i in range(len(Xcat_list))]
    return inputs

test_row = df.iloc[[0]].copy()
try:
    inputs = df_to_bank_inputs(test_row)
    proba = model.predict(inputs, verbose=0).ravel()[0]
    print("Sample prediction OK; churn proba:", float(proba))
except Exception as e:
    print("Prediction test failed:", e)
    sys.exit(1)

print("Saved files into:", models_dir)
print("- bank_label_encoders.pkl", (models_dir / "bank_label_encoders.pkl").exists())
print("- bank_scaler.pkl", (models_dir / "bank_scaler.pkl").exists())
print("- bank_cardinalities.pkl", (models_dir / "bank_cardinalities.pkl").exists())
