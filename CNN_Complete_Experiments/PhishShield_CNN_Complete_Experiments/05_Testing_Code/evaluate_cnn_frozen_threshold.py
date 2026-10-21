
from pathlib import Path
import json
import sys

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
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml_model" / "src"))

import train_200k_experiment as exp

MODEL_FILE = (
    ROOT / "ml_model" / "results" / "experiment_200k_v1"
    / "character_cnn_200k.keras"
)

OUT_DIR = ROOT / "ml_model" / "results" / "cnn_frozen_threshold_v1"

SEED = 42
VALIDATION_FPR_LIMIT = 0.10
BATCH_SIZE = 512


def evaluate(y_true, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predicted, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision_phishing": float(
            precision_score(y_true, predicted, zero_division=0)
        ),
        "phishing_recall": float(
            recall_score(y_true, predicted, zero_division=0)
        ),
        "f1_phishing": float(
            f1_score(y_true, predicted, zero_division=0)
        ),
        "false_positive_rate": float(fp / (fp + tn)) if tn + fp else 0.0,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
    }


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"CNN model not found: {MODEL_FILE}")

    print("=" * 65)
    print("PHISHSHIELD MAIN CNN - FROZEN THRESHOLD EVALUATION")
    print("=" * 65)

    # Recreate the exact training/validation split from the experiment.
    data = exp.load_training_data()
    external = exp.load_external_data()

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.15, random_state=SEED
    )
    train_idx, val_idx = next(
        splitter.split(
            data["URL"],
            data["target"],
            groups=data["host"],
        )
    )

    train = data.iloc[train_idx].reset_index(drop=True)
    validation = data.iloc[val_idx].reset_index(drop=True)

    overlap = set(train["host"]) & set(validation["host"])
    if overlap:
        raise RuntimeError("Training/validation hostname overlap detected.")

    url_overlap = set(data["URL"]) & set(external["URL"])
    if url_overlap:
        raise RuntimeError(
            f"Found {len(url_overlap)} exact URL overlaps with external data."
        )

    y_val = validation["target"].to_numpy(dtype=np.int32)
    y_ext = external["target"].to_numpy(dtype=np.int32)

    print(f"Validation URLs: {len(validation):,}")
    print(f"External URLs: {len(external):,}")
    print("Loading saved CNN...")
    model = tf.keras.models.load_model(MODEL_FILE)

    print("Encoding validation URLs...")
    X_val = exp.encode_urls(validation["URL"].tolist())

    print("Predicting validation probabilities...")
    p_val = model.predict(
        X_val, batch_size=BATCH_SIZE, verbose=0
    ).reshape(-1)

    # Choose threshold on validation only:
    # maximize phishing recall while keeping validation FPR <= 10%.
    fpr, tpr, thresholds = roc_curve(y_val, p_val)
    valid_indices = np.where(
        (fpr <= VALIDATION_FPR_LIMIT) & np.isfinite(thresholds)
    )[0]

    if len(valid_indices) == 0:
        raise RuntimeError("No finite threshold meets validation FPR limit.")

    best_idx = valid_indices[np.argmax(tpr[valid_indices])]
    chosen_threshold = float(thresholds[best_idx])

    validation_metrics = evaluate(y_val, p_val, chosen_threshold)

    print("\nTHRESHOLD SELECTED ON VALIDATION ONLY")
    print(json.dumps(validation_metrics, indent=2))
    print(f"\nFrozen threshold: {chosen_threshold:.8f}")

    # Freeze threshold before external evaluation.
    print("\nEncoding external URLs...")
    X_ext = exp.encode_urls(external["URL"].tolist())

    print("Predicting external probabilities...")
    p_ext = model.predict(
        X_ext, batch_size=BATCH_SIZE, verbose=0
    ).reshape(-1)

    external_metrics = evaluate(y_ext, p_ext, chosen_threshold)

    print("\nEXTERNAL TEST RESULTS — FROZEN THRESHOLD")
    print(json.dumps(external_metrics, indent=2))

    # Also show the default threshold as a reference.
    default_metrics = evaluate(y_ext, p_ext, 0.5)
    print("\nEXTERNAL TEST RESULTS — DEFAULT THRESHOLD 0.50")
    print(json.dumps(default_metrics, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([
        {"dataset": "grouped_validation", **validation_metrics},
        {"dataset": "clean_external_test_frozen_threshold", **external_metrics},
        {"dataset": "clean_external_test_default_0_50", **default_metrics},
    ]).to_csv(OUT_DIR / "metrics.csv", index=False)

    pd.DataFrame({
        "URL": external["URL"],
        "actual_label": y_ext,
        "phishing_probability": p_ext,
        "predicted_label_frozen_threshold": (
            p_ext >= chosen_threshold
        ).astype(int),
    }).to_csv(OUT_DIR / "external_predictions.csv", index=False)

    report = {
        "model": str(MODEL_FILE),
        "threshold_selection": "grouped validation only",
        "validation_fpr_limit": VALIDATION_FPR_LIMIT,
        "chosen_threshold": chosen_threshold,
        "validation_metrics": validation_metrics,
        "external_metrics_frozen_threshold": external_metrics,
        "external_metrics_default_0_50": default_metrics,
        "exact_url_overlap": len(url_overlap),
        "warning": (
            "The 200k data was used in previous experiments. "
            "The external dataset has been kept out of threshold selection. "
            "Exact URL deduplication does not guarantee domain/campaign independence."
        ),
    }

    with open(OUT_DIR / "report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nRESULTS SAVED TO:")
    print(OUT_DIR)
    print("\nThe saved CNN was not modified.")


if __name__ == "__main__":
    main()
