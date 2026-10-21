
import os
import json
import ipaddress
import re
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.utils import resample
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "ml_model",
    "external_data",
    "alternative_test",
    "dataset_web_fraud.csv",
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ml_model",
    "results",
    "cnn_generalization_experiment",
    "cnn_generalization_candidate.keras",
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "ml_model",
    "results",
    "cnn_generalization_experiment",
    "external_evaluation",
)

os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_URL_LENGTH = 300
PAD_ID = 0
UNK_ID = 1
ASCII_OFFSET = 2
VOCAB_SIZE = 130


# ==========================================================
# NORMALIZATION AND ENCODING
# Must match the training script.
# ==========================================================

def normalize_for_model(url):
    url = str(url).strip()

    if not url:
        return ""

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "https://" + url

    try:
        parts = urlsplit(url)
        hostname = (parts.hostname or "").lower()

        if not hostname:
            return url.lower()

        try:
            ipaddress.ip_address(hostname)
            normalized_host = hostname
        except ValueError:
            try:
                normalized_host = hostname.encode("idna").decode("ascii")
            except UnicodeError:
                normalized_host = hostname

        host_for_url = normalized_host
        if ":" in normalized_host and not normalized_host.startswith("["):
            host_for_url = f"[{normalized_host}]"

        netloc = host_for_url
        if parts.port is not None:
            netloc += f":{parts.port}"

        return (
            parts.scheme.lower()
            + "://"
            + netloc
            + parts.path
            + (("?" + parts.query) if parts.query else "")
        ).lower()

    except (ValueError, TypeError, AttributeError):
        return url.lower()


def encode_url(url):
    url = normalize_for_model(url)
    encoded = []

    for char in url[:MAX_URL_LENGTH]:
        code = ord(char)

        if 0 <= code < 128:
            encoded.append(code + ASCII_OFFSET)
        else:
            encoded.append(UNK_ID)

    return encoded


def convert_external_labels(series):
    """
    The known DatasetWebFraudDetection file uses verdict labels.
    Inspect the actual values and fail rather than silently guessing.
    """
    values = series.astype(str).str.strip().str.lower()

    mapping = {
        "phishing": 1,
        "phish": 1,
        "malicious": 1,
        "fraud": 1,
        "bad": 1,
        "1": 1,
        "legitimate": 0,
        "legit": 0,
        "benign": 0,
        "good": 0,
        "0": 0,
    }

    converted = values.map(mapping)

    if converted.isna().any():
        unknown = sorted(values[converted.isna()].unique().tolist())
        raise ValueError(
            "Unrecognised verdict labels: "
            + str(unknown[:20])
            + ". Check the dataset's label meanings."
        )

    return converted.astype(int)


# ==========================================================
# METRICS
# ==========================================================

def calculate_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "Threshold": float(threshold),
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "F1 Score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "ROC-AUC": float(roc_auc_score(y_true, probabilities)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "FPR": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("\n" + "=" * 72)
    print("PHISHSHIELD - EXTERNAL CNN EVALUATION")
    print("=" * 72)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(DATASET_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    data = pd.read_csv(DATASET_PATH)

    print("\nDataset columns:", data.columns.tolist())
    print("Original dataset shape:", data.shape)

    # This dataset is expected to have URL and verdict columns.
    url_col = next(
        (c for c in data.columns if c.lower() == "url"),
        None,
    )
    label_col = next(
        (c for c in data.columns if c.lower() == "verdict"),
        None,
    )

    if url_col is None or label_col is None:
        raise ValueError(
            "Expected URL and verdict columns. "
            "Actual columns: " + str(data.columns.tolist())
        )

    data = data[[url_col, label_col]].dropna().copy()
    data.columns = ["URL", "Verdict"]
    data["URL"] = data["URL"].astype(str).str.strip()
    data = data[data["URL"] != ""].copy()

    data["True_Phishing"] = convert_external_labels(data["Verdict"])
    data["NormalizedURL"] = data["URL"].map(normalize_for_model)

    # Keep one record per normalized URL.
    # Conflicting labels must not be silently resolved.
    conflicts = data.groupby("NormalizedURL")["True_Phishing"].nunique()
    conflicting_urls = set(conflicts[conflicts > 1].index)

    print("Conflicting URLs:", len(conflicting_urls))

    data = data[
        ~data["NormalizedURL"].isin(conflicting_urls)
    ].copy()

    data = data.drop_duplicates(
        subset=["NormalizedURL"]
    ).reset_index(drop=True)

    print("Rows evaluated:", len(data))
    print("\nExternal label distribution (0=legitimate, 1=phishing):")
    print(data["True_Phishing"].value_counts().sort_index())

    # Encode using exactly the same rules used during training.
    X_external = pad_sequences(
        [encode_url(url) for url in data["NormalizedURL"]],
        maxlen=MAX_URL_LENGTH,
        padding="post",
        truncating="post",
        value=PAD_ID,
        dtype="int32",
    )

    y_true = data["True_Phishing"].to_numpy(dtype=np.int32)

    print("\nLoading candidate model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Input shape:", X_external.shape)
    print("Predicting external URLs...")

    probabilities = model.predict(
        X_external,
        batch_size=256,
        verbose=1,
    ).reshape(-1)

    metrics = calculate_metrics(y_true, probabilities, threshold=0.5)

    print("\nExternal metrics at the unchanged threshold 0.50:")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    metrics_path = os.path.join(RESULTS_DIR, "external_metrics.json")

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    predictions = data[
        ["URL", "Verdict", "NormalizedURL", "True_Phishing"]
    ].copy()

    predictions["CNN_Probability"] = probabilities
    predictions["CNN_Prediction"] = (probabilities >= 0.5).astype(int)

    predictions_path = os.path.join(
        RESULTS_DIR,
        "external_predictions.csv",
    )

    predictions.to_csv(predictions_path, index=False)

    print("\nSaved metrics:", metrics_path)
    print("Saved predictions:", predictions_path)
    print("\nOriginal models were not modified.")
    print("External labels were not used for training or threshold tuning.")


if __name__ == "__main__":
    main()
