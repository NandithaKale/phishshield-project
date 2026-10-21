"""
Evaluate the saved Character CNN on the reconstructed original random holdout.
Read-only: does not train or save/replace any model.
Run from project root:
    python ml_model/src/evaluate_cnn_original_holdout.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.url_normalizer import normalize_url

DATA_PATH = PROJECT_ROOT / "ml_model" / "dataset" / "phishing.csv"
CNN_PATH = PROJECT_ROOT / "ml_model" / "results" / "character_cnn_comparison.keras"
OUT_DIR = PROJECT_ROOT / "ml_model" / "results" / "cnn_original_holdout_diagnostic"
MAX_URL_LENGTH = 200
VOCAB_SIZE = 128
RANDOM_STATE = 42

def is_valid_url(url):
    try:
        normalize_url(str(url))
        return True
    except (ValueError, TypeError, AttributeError):
        return False

def encode_url(url):
    # Match compare_models.py: encode the raw URL string, lowercase only;
    # do not add a scheme or otherwise normalize the URL before encoding.
    encoded = []
    for character in str(url).lower()[:MAX_URL_LENGTH]:
        value = ord(character)
        if value >= VOCAB_SIZE:
            value = VOCAB_SIZE - 1
        encoded.append(value)
    encoded.extend([0] * (MAX_URL_LENGTH - len(encoded)))
    return encoded

def main():
    for path in (DATA_PATH, CNN_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Loading original dataset...")
    data = pd.read_csv(DATA_PATH, encoding="latin1")
    data = data[["URL", "Label"]].dropna().copy()
    data["URL"] = data["URL"].astype(str).str.strip()
    data["Label"] = (
        data["Label"].astype(str).str.lower()
        .map({"good": 0, "bad": 1, "0": 0, "1": 1})
    )
    data = data.dropna(subset=["Label"]).copy()
    data["Label"] = data["Label"].astype(int)
    data = data.drop_duplicates(subset=["URL"])
    data = data[data["URL"].map(is_valid_url)].reset_index(drop=True)

    print("Cleaned rows:", len(data))
    print("Labels:", data["Label"].value_counts().sort_index().to_dict())

    _, test_data = train_test_split(
        data, test_size=0.20, stratify=data["Label"],
        random_state=RANDOM_STATE
    )
    test_data = test_data.reset_index(drop=True)
    print("Reconstructed holdout rows:", len(test_data))
    print("Holdout labels:", test_data["Label"].value_counts().sort_index().to_dict())

    print("Loading saved CNN (no training)...")
    model = tf.keras.models.load_model(CNN_PATH)

    urls = test_data["URL"].tolist()
    X = np.asarray([encode_url(url) for url in urls], dtype=np.int32)
    y_true = test_data["Label"].to_numpy(dtype=int)

    print("Predicting", len(urls), "URLs...")
    probabilities = model.predict(X, batch_size=512, verbose=0).reshape(-1)
    y_pred = (probabilities >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion matrix (rows=true [legitimate, phishing], columns=predicted):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=["Legitimate", "Phishing"], zero_division=0
    ))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Phishing precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Phishing recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"False-positive rate: {fp / (fp + tn):.4f}" if fp + tn else "FPR: undefined")
    print("\nMean CNN phishing probability by true class:")
    print(pd.DataFrame({
        "True_Label": np.where(y_true == 1, "Phishing", "Legitimate"),
        "CNN_Phishing_Probability": probabilities
    }).groupby("True_Label")["CNN_Phishing_Probability"].describe().round(4))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = OUT_DIR / "cnn_original_holdout_predictions.csv"
    metrics_path = OUT_DIR / "cnn_original_holdout_metrics.csv"
    predictions = test_data[["URL", "Label"]].copy()
    predictions["True_Label"] = np.where(y_true == 1, "Phishing", "Legitimate")
    predictions["CNN_Phishing_Probability"] = probabilities
    predictions["CNN_Prediction"] = y_pred
    predictions.to_csv(predictions_path, index=False)

    metrics = pd.DataFrame([{
        "Evaluation_URLs": len(y_true),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0),
        "False_Positive_Rate": fp / (fp + tn) if fp + tn else np.nan,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Input_preprocessing": "raw URL, lowercase, truncate/pad to 200; same encoding as compare_models.py"
    }])
    metrics.to_csv(metrics_path, index=False)
    print("\nNew files saved:")
    print(predictions_path)
    print(metrics_path)
    print("Existing models and result files were not modified.")

if __name__ == "__main__":
    main()
