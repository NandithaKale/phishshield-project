
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
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

OUT = ROOT / "ml_model" / "results" / "cnn_candidate_domain_adapt_v1"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 65)
print("PHISHSHIELD - SEPARATE CNN CANDIDATE")
print("=" * 65)

# Load the original dataset using its existing label mapping.
source = exp.load_training_data()[["URL", "target", "host"]].copy()
source["dataset_source"] = "original"

# Load the cleaned external dataset.
external = exp.load_external_data()[["URL", "target"]].copy()
external["host"] = external["URL"].map(exp.get_host)
external["dataset_source"] = "external"

# Combine and remove exact URL duplicates.
data = pd.concat([source, external], ignore_index=True)
data = data.drop_duplicates(subset=["URL"]).reset_index(drop=True)
data = data[data["host"].fillna("") != ""].reset_index(drop=True)
data["target"] = data["target"].astype("int32")

print("Combined URLs:", len(data))
print("Legitimate (0):", int((data["target"] == 0).sum()))
print("Phishing (1):", int((data["target"] == 1).sum()))
print("Unique hostnames:", data["host"].nunique())

# Keep every hostname entirely in either train or validation.
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.15, random_state=SEED
)
train_idx, val_idx = next(
    splitter.split(data, data["target"], groups=data["host"])
)

train = data.iloc[train_idx].reset_index(drop=True)
val = data.iloc[val_idx].reset_index(drop=True)

overlap = set(train["host"]) & set(val["host"])
assert not overlap, "Hostname leakage detected!"

print("Training URLs:", len(train))
print("Validation URLs:", len(val))
print("Hostname overlap:", len(overlap))

print("Encoding URLs...")
X_train = exp.encode_urls(train["URL"].tolist())
X_val = exp.encode_urls(val["URL"].tolist())
y_train = train["target"].to_numpy()
y_val = val["target"].to_numpy()

# Build a new CNN. Do not load or overwrite the existing model.
model = exp.build_cnn()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    ),
]

print("Training candidate CNN...")
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=6,
    batch_size=256,
    class_weight={
        0: len(y_train) / (2 * max(1, int((y_train == 0).sum()))),
        1: len(y_train) / (2 * max(1, int((y_train == 1).sum()))),
    },
    callbacks=callbacks,
    verbose=1,
)

model_path = OUT / "character_cnn_candidate.keras"
model.save(model_path)

print("Predicting validation URLs...")
prob = model.predict(X_val, batch_size=512, verbose=1).ravel()
pred = (prob >= 0.5).astype("int32")

tn, fp, fn, tp = confusion_matrix(
    y_val, pred, labels=[0, 1]
).ravel()

metrics = {
    "model": "Character CNN candidate",
    "threshold": 0.5,
    "accuracy": float(accuracy_score(y_val, pred)),
    "precision_phishing": float(
        precision_score(y_val, pred, zero_division=0)
    ),
    "phishing_recall": float(
        recall_score(y_val, pred, zero_division=0)
    ),
    "f1_phishing": float(
        f1_score(y_val, pred, zero_division=0)
    ),
    "false_positive_rate": float(fp / max(1, tn + fp)),
    "roc_auc": float(roc_auc_score(y_val, prob)),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
    "training_urls": int(len(train)),
    "validation_urls": int(len(val)),
    "hostname_overlap": int(len(overlap)),
    "note": (
        "Combined grouped validation only; not an untouched "
        "external test. Threshold is the default 0.5."
    ),
}

pd.DataFrame([metrics]).to_csv(OUT / "metrics.csv", index=False)
pd.DataFrame({
    "URL": val["URL"],
    "actual_label": y_val,
    "phishing_probability": prob,
    "predicted_label": pred,
}).to_csv(OUT / "validation_predictions.csv", index=False)

import json
with open(OUT / "report.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nVALIDATION RESULTS")
print(json.dumps(metrics, indent=2))
print("\nCandidate saved to:", model_path)
print("Original models were not modified.")
