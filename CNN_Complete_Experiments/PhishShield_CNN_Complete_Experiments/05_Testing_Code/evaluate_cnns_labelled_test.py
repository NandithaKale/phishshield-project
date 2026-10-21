"""
PhishShield: evaluate the saved baseline and multi-kernel CNNs on the separate
200,000-row labelled dataset. Does not train or overwrite either model.

Label mapping for this dataset (confirmed from the project context):
    Label 0 = phishing
    Label 1 = legitimate

Run from repository root:
    python -u ml_model\src\evaluate_cnns_labelled_test.py
"""
from pathlib import Path
import json
import runpy
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[2]
BASELINE_SCRIPT = ROOT / "ml_model" / "src" / "train_cnn_class_weighted.py"
CANDIDATE_SCRIPT = ROOT / "ml_model" / "src" / "train_cnn_multikernel_candidate.py"
BASELINE_MODEL = ROOT / "ml_model" / "results" / "cnn_class_weighted_experiment" / "cnn_class_weighted_candidate.keras"
CANDIDATE_MODEL = ROOT / "ml_model" / "results" / "cnn_multikernel_candidate" / "cnn_multikernel_candidate.keras"
DATASET = ROOT / "ml_model" / "external_data" / "labelled_test" / "phishing_url_dataset.csv"
OUT_DIR = ROOT / "ml_model" / "results" / "cnn_labelled_test_evaluation"
MAX_BATCH_SIZE = 512

# Thresholds selected during the earlier diagnostic dataset sweep.
THRESHOLDS = {"Baseline CNN": 0.40, "Multi-kernel CNN": 0.55}

def evaluate(y_true, probabilities, threshold, model_name):
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_phishing": float(precision_score(y_true, y_pred, zero_division=0)),
        "phishing_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_phishing": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)),
    }

def prepare_baseline(urls, helpers):
    encode_url = helpers["encode_url"]
    max_len = int(helpers.get("MAX_URL_LENGTH", 300))
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seqs = [encode_url(str(url)) for url in urls]
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post", value=0)

def prepare_candidate(urls, helpers):
    encode_urls = helpers["encode_urls"]
    return np.asarray(encode_urls([str(url) for url in urls]), dtype=np.int32)

def predict_in_batches(model, x, batch_size=MAX_BATCH_SIZE):
    outputs = []
    for start in range(0, len(x), batch_size):
        batch = model.predict(x[start:start + batch_size], verbose=0)
        outputs.append(np.asarray(batch).reshape(-1))
    return np.concatenate(outputs)

def main():
    for path in (BASELINE_SCRIPT, CANDIDATE_SCRIPT, BASELINE_MODEL, CANDIDATE_MODEL, DATASET):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("=" * 76)
    print("PHISHSHIELD — SEPARATE 200,000-ROW LABELLED DATASET EVALUATION")
    print("=" * 76)
    print("No training. No model files will be changed.")
    print("\nLoading the saved models' matching preprocessing functions...")
    base_helpers = runpy.run_path(str(BASELINE_SCRIPT))
    cand_helpers = runpy.run_path(str(CANDIDATE_SCRIPT))

    print(f"\nReading dataset: {DATASET}")
    df = pd.read_csv(DATASET)
    if not {"URL", "Label"}.issubset(df.columns):
        raise ValueError(f"Expected columns URL and Label; found {df.columns.tolist()}")
    df = df[["URL", "Label"]].dropna().copy()
    df["URL"] = df["URL"].astype(str).str.strip()
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df.dropna(subset=["Label"])
    df = df[df["Label"].isin([0, 1])]
    df = df[df["URL"].str.len() > 0].reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable URL rows found.")
    # Grounded project mapping: 0=phishing, 1=legitimate.
    y_phishing = (df["Label"].astype(int).to_numpy() == 0).astype(int)
    urls = df["URL"].tolist()
    print(f"Usable rows: {len(df):,}")
    print("Source labels: Label 0 = phishing; Label 1 = legitimate")
    print(f"Phishing: {int(y_phishing.sum()):,} | Legitimate: {int(len(y_phishing)-y_phishing.sum()):,}")

    print("\nEncoding URLs using baseline model's own training-time encoder...")
    x_base = prepare_baseline(urls, base_helpers)
    print(f"Baseline input shape: {x_base.shape}")
    print("\nEncoding URLs using multi-kernel model's own training-time encoder...")
    x_cand = prepare_candidate(urls, cand_helpers)
    print(f"Multi-kernel input shape: {x_cand.shape}")

    print("\nLoading saved CNNs...")
    baseline = load_model(str(BASELINE_MODEL), compile=False)
    candidate = load_model(str(CANDIDATE_MODEL), compile=False)

    print("\nPredicting with baseline CNN...")
    p_base = predict_in_batches(baseline, x_base)
    print("Predicting with multi-kernel CNN...")
    p_cand = predict_in_batches(candidate, x_cand)

    # Sanity-check probabilities before interpreting output as phishing probability.
    for name, probs in (("Baseline CNN", p_base), ("Multi-kernel CNN", p_cand)):
        if not np.isfinite(probs).all():
            raise ValueError(f"{name} produced non-finite predictions.")
        if probs.min() < -1e-6 or probs.max() > 1.000001:
            raise ValueError(f"{name} outputs are not probabilities in [0,1].")
    # Both existing evaluation scripts use the model output as phishing probability.
    rows = []
    rows.append(evaluate(y_phishing, p_base, THRESHOLDS["Baseline CNN"], "Baseline CNN"))
    rows.append(evaluate(y_phishing, p_cand, THRESHOLDS["Multi-kernel CNN"], "Multi-kernel CNN"))
    # Include the common 0.50 operating point for direct reference as well.
    rows.append(evaluate(y_phishing, p_base, 0.50, "Baseline CNN"))
    rows.append(evaluate(y_phishing, p_cand, 0.50, "Multi-kernel CNN"))
    metrics = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT_DIR / "metrics.csv"
    predictions_path = OUT_DIR / "predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame({
        "URL": urls,
        "source_label": df["Label"].astype(int).to_numpy(),
        "actual_phishing": y_phishing,
        "baseline_phishing_probability": p_base,
        "candidate_phishing_probability": p_cand,
    }).to_csv(predictions_path, index=False)

    print("\nRESULTS")
    print("-" * 76)
    for row in rows:
        print(f"\n{row['model']} | threshold={row['threshold']:.2f}")
        print(f"Accuracy:           {row['accuracy']:.2%}")
        print(f"Phishing precision: {row['precision_phishing']:.2%}")
        print(f"Phishing recall:    {row['phishing_recall']:.2%}")
        print(f"F1-score:           {row['f1_phishing']:.2%}")
        print(f"False positive rate:{row['false_positive_rate']:.2%}")
        print(f"ROC-AUC:            {row['roc_auc']:.4f}")
        print(f"TN={row['tn']:,}  FP={row['fp']:,}  FN={row['fn']:,}  TP={row['tp']:,}")
    print("\nSaved metrics:", metrics_path)
    print("Saved per-URL predictions:", predictions_path)
    print("\nCaution: thresholds 0.40 and 0.55 were selected on a previous dataset.")
    print("Treat this as a separate-dataset evaluation; do not tune thresholds on these results.")

if __name__ == "__main__":
    main()
