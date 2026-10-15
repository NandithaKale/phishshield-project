"""
Compare PhishShield baseline Character CNN vs multi-kernel CNN candidate.
Uses the same external URLs and labels. Does not train or overwrite models.
Run from repository root:
    python -u ml_model/src/compare_cnn_candidates_external.py
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINE_SCRIPT = ROOT / "ml_model" / "src" / "train_cnn_class_weighted.py"
BASELINE_MODEL = (
    ROOT / "ml_model" / "results" / "cnn_class_weighted_experiment"
    / "cnn_class_weighted_candidate.keras"
)
CANDIDATE_SCRIPT = ROOT / "ml_model" / "src" / "train_cnn_multikernel_candidate.py"
CANDIDATE_MODEL = (
    ROOT / "ml_model" / "results" / "cnn_multikernel_candidate"
    / "cnn_multikernel_candidate.keras"
)
EXTERNAL_FILE = (
    ROOT / "ml_model" / "external_data" / "alternative_test"
    / "dataset_web_fraud.csv"
)
OUT_DIR = ROOT / "ml_model" / "results" / "cnn_same_external_comparison"
BATCH_SIZE = 512


def metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_phishing": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "phishing_recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_phishing": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)),
    }


def main():
    print("=" * 68)
    print("PHISHSHIELD: SAME-EXTERNAL-DATASET CNN COMPARISON")
    print("=" * 68)

    required = [
        BASELINE_SCRIPT, BASELINE_MODEL, CANDIDATE_SCRIPT,
        CANDIDATE_MODEL, EXTERNAL_FILE
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required file(s) missing:\n" + "\n".join(missing)
        )

    print("\nLoading baseline preprocessing from training script...")
    baseline_helpers = runpy.run_path(str(BASELINE_SCRIPT))
    for name in ("normalize_for_model", "encode_url"):
        if not callable(baseline_helpers.get(name)):
            raise RuntimeError(
                f"Could not find {name} in {BASELINE_SCRIPT}"
            )
    baseline_normalize = baseline_helpers["normalize_for_model"]
    baseline_encode = baseline_helpers["encode_url"]
    baseline_max_len = int(baseline_helpers.get("MAX_URL_LENGTH", 300))

    print("Loading candidate preprocessing from candidate script...")
    candidate_helpers = runpy.run_path(str(CANDIDATE_SCRIPT))
    for name in ("normalize_url", "encode_urls"):
        if not callable(candidate_helpers.get(name)):
            raise RuntimeError(
                f"Could not find {name} in {CANDIDATE_SCRIPT}"
            )
    candidate_normalize = candidate_helpers["normalize_url"]
    candidate_encode_many = candidate_helpers["encode_urls"]

    print("\nReading external dataset...")
    data = pd.read_csv(EXTERNAL_FILE)
    if not {"url", "verdict"}.issubset(data.columns):
        raise ValueError(
            "Expected columns 'url' and 'verdict'. Found: "
            + str(list(data.columns))
        )

    data = data[["url", "verdict"]].copy()
    data["url"] = data["url"].astype(str).str.strip()
    data["verdict"] = pd.to_numeric(data["verdict"], errors="coerce")
    data = data[
        data["url"].ne("") & data["verdict"].isin([0, 1])
    ].copy()
    data["actual_phishing"] = data["verdict"].astype(int)

    # The candidate evaluation used verdict 0=legitimate, 1=phishing.
    # Keep this explicit so label polarity is not silently changed.
    data = data.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    y_true = data["actual_phishing"].to_numpy(dtype=np.int32)
    if len(data) == 0 or set(np.unique(y_true)) != {0, 1}:
        raise ValueError("External data must contain both verdict labels 0 and 1.")

    print(f"External URLs used by both models: {len(data):,}")
    print("Label mapping: verdict 0=legitimate, verdict 1=phishing")

    print("\nPreparing baseline input with its own original encoder...")
    baseline_urls = [
        baseline_normalize(url) for url in data["url"].tolist()
    ]
    baseline_sequences = [
        baseline_encode(url) for url in data["url"].tolist()
    ]
    X_baseline = pad_sequences(
        baseline_sequences,
        maxlen=baseline_max_len,
        padding="post",
        truncating="post",
        value=0,
        dtype="int32",
    )

    print("Preparing candidate input with its own candidate encoder...")
    candidate_urls = [
        candidate_normalize(url) for url in data["url"].tolist()
    ]
    X_candidate = candidate_encode_many(data["url"].tolist())

    print("\nLoading both saved models (no training)...")
    baseline_model = tf.keras.models.load_model(BASELINE_MODEL, compile=False)
    candidate_model = tf.keras.models.load_model(CANDIDATE_MODEL, compile=False)

    print("\nPredicting with baseline CNN...")
    baseline_prob = baseline_model.predict(
        X_baseline, batch_size=BATCH_SIZE, verbose=1
    ).reshape(-1)

    print("\nPredicting with multi-kernel CNN...")
    candidate_prob = candidate_model.predict(
        X_candidate, batch_size=BATCH_SIZE, verbose=1
    ).reshape(-1)

    baseline_metrics = metrics(y_true, baseline_prob)
    baseline_metrics["model"] = "Existing class-weighted Character CNN"
    candidate_metrics = metrics(y_true, candidate_prob)
    candidate_metrics["model"] = "New multi-kernel Character CNN"

    print("\nBASELINE RESULTS (threshold 0.50)")
    print(json.dumps(baseline_metrics, indent=2))
    print("\nMULTI-KERNEL CANDIDATE RESULTS (threshold 0.50)")
    print(json.dumps(candidate_metrics, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([baseline_metrics, candidate_metrics]).to_csv(
        OUT_DIR / "metrics.csv", index=False
    )

    pd.DataFrame({
        "url": data["url"],
        "actual_phishing": y_true,
        "baseline_normalized_url": baseline_urls,
        "baseline_phishing_probability": baseline_prob,
        "baseline_prediction_0_50": (baseline_prob >= 0.5).astype(int),
        "candidate_normalized_url": candidate_urls,
        "candidate_phishing_probability": candidate_prob,
        "candidate_prediction_0_50": (candidate_prob >= 0.5).astype(int),
    }).to_csv(OUT_DIR / "predictions.csv", index=False)

    print("\nSaved comparison files to:")
    print(OUT_DIR)
    print("\nNote: each model uses its own training-time preprocessing/encoding.")
    print("The URL rows and labels are the same for both models.")
    print("Neither model was changed.")


if __name__ == "__main__":
    main()
