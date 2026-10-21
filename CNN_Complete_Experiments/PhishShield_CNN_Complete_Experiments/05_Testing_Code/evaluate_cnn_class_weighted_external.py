
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_SCRIPT = ROOT / "ml_model" / "src" / "train_cnn_class_weighted.py"

DATASET_PATH = (
    ROOT / "ml_model" / "external_data"
    / "alternative_test" / "dataset_web_fraud.csv"
)

MODEL_PATH = (
    ROOT / "ml_model" / "results"
    / "cnn_class_weighted_experiment"
    / "cnn_class_weighted_candidate.keras"
)

OUTPUT_DIR = (
    ROOT / "ml_model" / "results"
    / "cnn_class_weighted_experiment"
    / "external_evaluation_consistent_preprocessing"
)

THRESHOLD = 0.50
BATCH_SIZE = 512

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REUSE THE EXACT TRAINING PREPROCESSING FUNCTIONS
# runpy.run_path executes the training script's top-level code,
# but does not call main() because it is not __main__.
# ============================================================

print("Loading preprocessing functions from training script...")

training_namespace = runpy.run_path(str(TRAIN_SCRIPT))

normalize_for_model = training_namespace["normalize_for_model"]
encode_url = training_namespace["encode_url"]
MAX_URL_LENGTH = training_namespace["MAX_URL_LENGTH"]

print("Training preprocessing functions loaded.")
print("Maximum URL length:", MAX_URL_LENGTH)


# ============================================================
# LABEL MAPPING
# Dataset convention confirmed in the conversation:
# 0 = legitimate
# 1 = phishing
# ============================================================

def convert_label(value):
    if pd.isna(value):
        return np.nan

    label = str(value).strip().lower()

    if label in {"0", "legitimate", "legit", "good", "benign", "safe"}:
        return 0

    if label in {"1", "phishing", "phish", "bad", "malicious", "malware", "fraud"}:
        return 1

    return np.nan


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 65)
    print("CNN EXTERNAL EVALUATION - CONSISTENT PREPROCESSING")
    print("=" * 65)

    print("Threshold:", THRESHOLD)
    print("Dataset:", DATASET_PATH)
    print("Model:", MODEL_PATH)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # 1. Load data.
    df = pd.read_csv(DATASET_PATH)

    if not {"url", "verdict"}.issubset(df.columns):
        raise ValueError(
            f"Expected columns url and verdict; found {list(df.columns)}"
        )

    print("\nRaw verdict counts:")
    print(df["verdict"].value_counts(dropna=False).to_string())

    df = df[["url", "verdict"]].copy()
    df["NormalizedURL"] = df["url"].map(normalize_for_model)
    df["Label"] = df["verdict"].map(convert_label)

    df = df[
        df["NormalizedURL"].ne("")
        & df["Label"].notna()
    ].copy()

    df["Label"] = df["Label"].astype(int)

    # Exclude normalized URLs with conflicting labels.
    label_counts = df.groupby("NormalizedURL")["Label"].nunique()
    conflicting_urls = set(label_counts[label_counts > 1].index)

    if conflicting_urls:
        print("Conflicting URLs excluded:", len(conflicting_urls))
        df = df[~df["NormalizedURL"].isin(conflicting_urls)].copy()

    df = df.drop_duplicates(
        subset=["NormalizedURL"],
        keep="first"
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows remain after cleaning.")

    y_true = df["Label"].to_numpy(dtype=np.int32)

    print("\nRows evaluated:", len(df))
    print("Legitimate (0):", int((y_true == 0).sum()))
    print("Phishing (1):", int((y_true == 1).sum()))

    if len(np.unique(y_true)) != 2:
        raise ValueError("Both legitimate and phishing labels are required.")

    # 2. Encode using the exact training function.
    print("\nEncoding URLs using the training function...")

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    encoded_sequences = [
        encode_url(url) for url in df["NormalizedURL"]
    ]

    X = pad_sequences(
        encoded_sequences,
        maxlen=MAX_URL_LENGTH,
        padding="post",
        truncating="post",
        value=0,
        dtype="int32",
    )

    print("Input shape:", X.shape)

    # 3. Load model.
    print("\nLoading CNN candidate...")
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded.")

    # 4. Predict.
    print("\nGenerating predictions...")

    probabilities = model.predict(
        X,
        batch_size=BATCH_SIZE,
        verbose=1,
    ).reshape(-1)

    if len(probabilities) != len(y_true):
        raise ValueError("Prediction count does not match labels.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Invalid probabilities produced by the model.")

    predictions = (probabilities >= THRESHOLD).astype(np.int32)

    # 5. Metrics.
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    auc = roc_auc_score(y_true, probabilities)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print("\n" + "=" * 65)
    print(f"EXTERNAL RESULTS AT THRESHOLD {THRESHOLD:.2f}")
    print("=" * 65)
    print(f"Accuracy:            {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision:           {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:              {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-score:            {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"ROC-AUC:             {auc:.5f}")
    print(f"False-positive rate: {fpr:.4f} ({fpr * 100:.2f}%)")

    print("\nConfusion Matrix:")
    print("TN - Legitimate correctly identified:", tn)
    print("FP - Legitimate flagged as phishing: ", fp)
    print("FN - Phishing missed:                ", fn)
    print("TP - Phishing correctly identified:  ", tp)

    # 6. Save predictions.
    output = df[["url", "NormalizedURL", "verdict", "Label"]].copy()
    output["PhishingProbability"] = probabilities
    output["PredictedLabel"] = predictions
    output["Threshold"] = THRESHOLD

    output["ActualClass"] = np.where(
        output["Label"] == 1, "Phishing", "Legitimate"
    )

    output["PredictedClass"] = np.where(
        output["PredictedLabel"] == 1, "Phishing", "Legitimate"
    )

    predictions_path = OUTPUT_DIR / "external_predictions.csv"
    output.to_csv(predictions_path, index=False)

    # 7. Save metrics.
    metrics = {
        "threshold": THRESHOLD,
        "samples": int(len(y_true)),
        "legitimate_samples": int((y_true == 0).sum()),
        "phishing_samples": int((y_true == 1).sum()),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "false_positive_rate": float(fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "model_path": str(MODEL_PATH),
        "dataset_path": str(DATASET_PATH),
        "preprocessing_source": str(TRAIN_SCRIPT),
    }

    metrics_path = OUTPUT_DIR / "external_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nSaved:")
    print(predictions_path)
    print(metrics_path)
    print("\nProduction models were not modified.")


if __name__ == "__main__":
    main()
