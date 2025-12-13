from pathlib import Path
import joblib
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys

MODELS_DIR = Path(r"C:\Users\rupes\Downloads\customer_churn\models")

def find_model(patterns):
    for p in MODELS_DIR.iterdir():
        name = p.name.lower()
        for pat in patterns:
            if pat in name:
                return p
    return None

bank_model_file = find_model(["bank_churn_model_emb.keras","bank_churn_model_emb",".bank_churn_model"])
telco_model_file = find_model(["telco_churn_model.keras","telco_churn_model",".telco_churn_model"])

print("MODELS_DIR:", MODELS_DIR)
print("bank_model_file:", bank_model_file)
print("telco_model_file:", telco_model_file)

def inspect_model(mpath, label):
    if mpath is None:
        print(f"{label}: NOT FOUND")
        return
    print(f"\n== {label} MODEL: {mpath} ==")
    try:
        model = load_model(str(mpath))
    except Exception as e:
        print("load_model ERROR:", e)
        return
    print("model.summary() ---")
    try:
        model.summary()
    except Exception:
        pass
    inp = model.inputs
    print("num_inputs:", len(inp))
    for i, t in enumerate(inp):
        print(f" input[{i}] name:", getattr(t, "name", None))
        print(f" input[{i}] shape:", tuple([None if s is None else int(s) for s in t.shape.as_list()]))
        print(f" input[{i}] dtype:", t.dtype)
    try:
        from tensorflow.keras import backend as K
        cfg = model.get_config()
        print("model_type:", cfg.get("name", "unknown"))
    except Exception:
        pass

inspect_model(bank_model_file, "BANK")
inspect_model(telco_model_file, "TELCO")

# scalers / encoders info
print("\n== AUX FILES IN MODELS DIR ==")
for fname in ["bank_scaler.pkl","bank_label_encoders.pkl","bank_cardinalities.pkl","telco_scaler.pkl","telco_label_encoders.pkl","telco_feature_order.pkl"]:
    f = MODELS_DIR / fname
    print(fname, "exists:", f.exists())
    if f.exists() and fname.endswith("scaler.pkl"):
        try:
            sc = joblib.load(f)
            print(" scaler: type", type(sc))
            if hasattr(sc, "feature_names_in_"):
                print(" scaler.feature_names_in_ length:", len(list(sc.feature_names_in_)))
                print(" scaler.feature_names_in_ sample:", list(sc.feature_names_in_)[:20])
        except Exception as e:
            print(" scaler load error:", e)
    if f.exists() and fname.endswith("label_encoders.pkl"):
        try:
            le = joblib.load(f)
            print(" encoders keys:", list(le.keys()))
            for k in list(le.keys()):
                try:
                    n = len(le[k].classes_)
                except Exception:
                    n = "?"
                print(" ", k, "classes:", n)
        except Exception as e:
            print(" encoders load error:", e)
    if f.exists() and fname.endswith("feature_order.pkl"):
        try:
            fo = joblib.load(f)
            print(" feature_order length:", len(fo))
            print(" feature_order sample:", fo[:40])
        except Exception as e:
            print(" feature_order load error:", e)
