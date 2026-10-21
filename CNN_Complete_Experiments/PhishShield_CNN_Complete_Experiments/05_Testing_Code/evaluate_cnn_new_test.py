
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

DATA_DIR = ROOT / "ml_model" / "external_data" / "new_test"
MODEL_PATH = (
    ROOT / "ml_model" / "results"
    / "cnn_candidate_domain_adapt_v1"
    / "character_cnn_candidate.keras"
)
OUT_DIR = ROOT / "ml_model" / "results" / "cnn_new_unseen_eval_v1"

FILES = ["dns_feature_94k.xlsx", "merged_features_final.xlsx"]
THRESHOLD = 0.5


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Candidate model not found: {MODEL_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading candidate training URLs for overlap checks...")
    source = exp.load_training_data()
    external = exp.load_external_data()

    training_urls = set(source["URL"])
    training_urls.update(external["URL"])

    print("Unique candidate training-pool URLs:", len(training_urls))

    print("\nLoading saved CNN candidate...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    reports = []

    test_url_sets = {}

    for filename in FILES:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        print("\n" + "=" * 70)
        print("DATASET:", filename)

        df = pd.read_excel(path)

        if not {"url", "label"}.issubset(df.columns):
            raise ValueError(
                f"{filename} must contain 'url' and 'label'. "
                f"Found: {df.columns.tolist()}"
            )

        df = df[["url", "label"]].copy()
        df["url"] = df["url"].map(exp.normalize_url)
        df["label"] = df["label"].astype(str).str.strip().str.lower()

        df = df[
            df["url"].ne("")
            & df["label"].isin(["phishing", "legitimate"])
        ].copy()

        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

        # Internal convention: 0 = legitimate, 1 = phishing.
        y_true = (
            df["label"].map({"legitimate": 0, "phishing": 1})
            .to_numpy(dtype=np.int32)
        )

        url_set = set(df["url"])
        test_url_sets[filename] = url_set
        overlap_urls = url_set & training_urls

        print("Valid unique URLs:", len(df))
        print("Legitimate:", int((y_true == 0).sum()))
        print("Phishing:", int((y_true == 1).sum()))
        print("Exact URL overlap with candidate training pool:", len(overlap_urls))

        print("Encoding URLs...")
        X = exp.encode_urls(df["url"].tolist())

        print("Predicting (no retraining)...")
        probabilities = model.predict(
            X, batch_size=512, verbose=1
        ).reshape(-1)

        predictions = (probabilities >= THRESHOLD).astype(np.int32)

        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        metrics = {
            "dataset": filename,
            "rows_evaluated": int(len(df)),
            "threshold": THRESHOLD,
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
            "false_positive_rate": float(fp / (fp + tn))
            if (fp + tn) else 0.0,
            "roc_auc": float(roc_auc_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2 else None,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "exact_url_overlap_with_candidate_training_pool": int(
                len(overlap_urls)
            ),
            "overlap_fraction": float(len(overlap_urls) / len(df))
            if len(df) else 0.0,
            "note": (
                "Existing saved candidate; no retraining; threshold 0.5. "
                "Exact URL overlap does not rule out shared domains "
                "or campaigns. Verify original dataset provenance."
            ),
        }

        reports.append(metrics)

        predictions_df = df.copy()
        predictions_df["actual_target"] = y_true
        predictions_df["phishing_probability"] = probabilities
        predictions_df["predicted_target"] = predictions
        predictions_df["predicted_label"] = np.where(
            predictions == 1, "phishing", "legitimate"
        )
        predictions_df["exact_training_url_overlap"] = (
            predictions_df["url"].isin(training_urls)
        )

        output_name = Path(filename).stem
        predictions_df.to_csv(
            OUT_DIR / f"{output_name}_predictions.csv",
            index=False,
        )

        print("\nMETRICS")
        print(json.dumps(metrics, indent=2))

    if len(test_url_sets) == 2:
        common_test_urls = (
            test_url_sets[FILES[0]] & test_url_sets[FILES[1]]
        )
        print(
            "\nExact URLs shared between the two test datasets:",
            len(common_test_urls),
        )

    with open(OUT_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    pd.DataFrame(reports).to_csv(
        OUT_DIR / "metrics.csv", index=False
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("Results folder:", OUT_DIR)
    print("Metrics file:", OUT_DIR / "metrics.csv")
    print("The saved candidate and original models were not modified.")


if __name__ == "__main__":
    main()
