
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml_model" / "src"))

import train_200k_experiment as exp

DATA_FILE = ROOT / "ml_model" / "external_data" / "new_test" / "balanced_urls_340k.csv"
MODEL_FILE = ROOT / "ml_model" / "results" / "cnn_candidate_domain_adapt_v1" / "character_cnn_candidate.keras"
OUT_DIR = ROOT / "ml_model" / "results" / "cnn_340k_eval_v1"
THRESHOLD = 0.5
CHUNK_SIZE = 10000


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE, usecols=["url", "label"])
    df["url"] = df["url"].map(exp.normalize_url)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["url"].ne("") & df["label"].isin(
        ["phishing", "legitimate"]
    )].copy()
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    # Internal target: 0 = legitimate, 1 = phishing.
    y = df["label"].map(
        {"legitimate": 0, "phishing": 1}
    ).to_numpy(dtype=np.int32)

    print("Unique valid URLs:", len(df))
    print("Legitimate:", int((y == 0).sum()))
    print("Phishing:", int((y == 1).sum()))

    print("\nChecking exact overlap with candidate training data...")
    training = exp.load_training_data()
    external = exp.load_external_data()
    training_urls = set(training["URL"]) | set(external["URL"])
    overlaps = df["url"].isin(training_urls).to_numpy()
    print("Exact URL overlaps:", int(overlaps.sum()))
    print("Overlap percentage:", round(100 * overlaps.mean(), 4), "%")

    print("\nLoading existing candidate CNN...")
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)

    probabilities = np.empty(len(df), dtype=np.float32)

    for start in range(0, len(df), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(df))
        print(f"Predicting {start:,}–{end:,} of {len(df):,}...")
        X = exp.encode_urls(df["url"].iloc[start:end].tolist())
        probabilities[start:end] = model.predict(
            X, batch_size=512, verbose=0
        ).reshape(-1)

    pred = (probabilities >= THRESHOLD).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "dataset": DATA_FILE.name,
        "rows_evaluated": int(len(df)),
        "threshold": THRESHOLD,
        "accuracy": float(accuracy_score(y, pred)),
        "precision_phishing": float(
            precision_score(y, pred, zero_division=0)
        ),
        "phishing_recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "f1_phishing": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if tn + fp else 0.0,
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
        "exact_url_overlap": int(overlaps.sum()),
        "overlap_fraction": float(overlaps.mean()),
        "note": (
            "Existing candidate; no retraining. Exact URL overlap "
            "does not rule out shared domains or campaigns."
        ),
    }

    df["actual_target"] = y
    df["phishing_probability"] = probabilities
    df["predicted_target"] = pred
    df["predicted_label"] = np.where(
        pred == 1, "phishing", "legitimate"
    )
    df["exact_training_url_overlap"] = overlaps

    df.to_csv(OUT_DIR / "predictions.csv", index=False)
    with open(OUT_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(
        OUT_DIR / "metrics.csv", index=False
    )

    print("\nFINAL RESULTS")
    print(json.dumps(metrics, indent=2))
    print("\nResults saved to:", OUT_DIR)
    print("Saved models were not modified.")


if __name__ == "__main__":
    main()
