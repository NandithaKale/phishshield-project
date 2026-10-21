
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml_model" / "src"))
import train_200k_experiment as exp

DATA_FILE = (
    ROOT / "ml_model" / "external_data" / "legitphish"
    / "url_features_extracted1.csv"
)
MODEL_FILE = (
    ROOT / "ml_model" / "results" / "cnn_candidate_domain_adapt_v1"
    / "character_cnn_candidate.keras"
)
OUT = ROOT / "ml_model" / "results" / "cnn_legitphish_eval_v1"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CNN CANDIDATE - LEGITPHISH EVALUATION")
print("=" * 60)

df = pd.read_csv(DATA_FILE, usecols=["URL", "ClassLabel"])
df["URL"] = df["URL"].map(exp.normalize_url)
df["ClassLabel"] = pd.to_numeric(df["ClassLabel"], errors="coerce")
df = df.dropna(subset=["URL", "ClassLabel"])
df = df[(df["URL"] != "") & df["ClassLabel"].isin([0, 1])]
df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)

# Verify this mapping from the dataset's documentation before
# interpreting class-specific metrics. These provisional values
# follow the apparent sample pattern: 0=phishing, 1=legitimate.
# Internal model target: 0=legitimate, 1=phishing.
df["target"] = df["ClassLabel"].map({0: 1, 1: 0}).astype("int32")

print("Rows evaluated:", len(df))
print("Source label counts:")
print(df["ClassLabel"].value_counts().sort_index().to_dict())
print("Using provisional mapping: source 0 -> phishing; source 1 -> legitimate")
print("This mapping must be verified against the dataset documentation.")

print("Loading separate candidate CNN...")
model = tf.keras.models.load_model(MODEL_FILE)

print("Encoding raw URLs...")
X = exp.encode_urls(df["URL"].tolist())

print("Predicting...")
prob = model.predict(X, batch_size=512, verbose=1).ravel()
pred = (prob >= 0.5).astype("int32")
y = df["target"].to_numpy()

tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

metrics = {
    "model": "character_cnn_candidate.keras",
    "dataset": "url_features_extracted1.csv",
    "rows": int(len(df)),
    "threshold": 0.5,
    "provisional_source_label_mapping": "0=phishing, 1=legitimate",
    "mapping_verified_from_documentation": False,
    "accuracy": float(accuracy_score(y, pred)),
    "precision_phishing": float(precision_score(y, pred, zero_division=0)),
    "phishing_recall": float(recall_score(y, pred, zero_division=0)),
    "f1_phishing": float(f1_score(y, pred, zero_division=0)),
    "false_positive_rate": float(fp / max(1, tn + fp)),
    "roc_auc": float(roc_auc_score(y, prob)),
    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    "warning": (
        "Provisional label mapping; verify from source documentation. "
        "Do not treat as final evidence until verified."
    ),
}

pd.DataFrame([metrics]).to_csv(OUT / "metrics.csv", index=False)
pd.DataFrame({
    "URL": df["URL"],
    "source_label": df["ClassLabel"],
    "actual_target_provisional": y,
    "phishing_probability": prob,
    "predicted_phishing_at_0_5": pred,
}).to_csv(OUT / "predictions.csv", index=False)

with open(OUT / "report.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nRESULTS")
print(json.dumps(metrics, indent=2))
print("\nSaved to:", OUT)
print("Existing models were not modified.")
