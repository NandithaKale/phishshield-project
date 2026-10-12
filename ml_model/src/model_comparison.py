"""
PhishShield: Random Forest vs XGBoost comparison.

Run from the repository root:
    python ml_model/src/model_comparison.py

Uses the existing shared feature extractor and URL normalizer.
Both models use the same stratified 80/20 train/test split and the
same training-only class-balancing procedure as compare_models.py.

Outputs (under ml_model/results/; separate from the earlier experiment):
    model_comparison_rf_vs_xgboost_leakage_resistant.csv
    model_comparison_rf_vs_xgboost_leakage_resistant.png
    model_comparison_label_distribution_leakage_resistant.csv
"""

import os
import random
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.feature_extractor import extract_features
from shared.url_normalizer import normalize_url

RANDOM_STATE = 42
DATASET_PATH = os.path.join(
    PROJECT_ROOT, "ml_model", "dataset", "phishing.csv"
)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "ml_model", "results")


def convert_labels(data):
    """Convert project labels to 0=good and 1=bad."""
    data = data.copy()
    data["Label"] = (
        data["Label"].astype(str).str.strip().str.lower()
        .map({"good": 0, "bad": 1, "0": 0, "1": 1})
    )
    return data.dropna(subset=["Label"]).assign(
        Label=lambda frame: frame["Label"].astype(int)
    )


def is_valid_url(url):
    """Use the project's existing URL normalizer for validation."""
    try:
        normalize_url(str(url))
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1 Score": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading dataset: {DATASET_PATH}")
    data = pd.read_csv(DATASET_PATH, encoding="latin1")

    if not {"URL", "Label"}.issubset(data.columns):
        raise ValueError("phishing.csv must contain 'URL' and 'Label' columns.")

    print("\nRaw label distribution (including missing labels):")
    print(data["Label"].value_counts(dropna=False).to_string())

    # Record raw label distribution separately; preserve earlier experiment outputs.
    raw_counts = data["Label"].value_counts(dropna=False)
    distribution = pd.DataFrame({
        "Label": [str(label) if pd.notna(label) else "MISSING" for label in raw_counts.index],
        "Count": raw_counts.values,
        "Percentage": raw_counts.values / len(data) * 100,
    })
    distribution.to_csv(
        os.path.join(RESULTS_DIR, "model_comparison_label_distribution_leakage_resistant.csv"),
        index=False,
    )

    # Clean labels and URLs, then normalize BEFORE splitting so equivalent URL
    # variants cannot land in both train and test.
    data = data[["URL", "Label"]].dropna().copy()
    data["URL"] = data["URL"].astype(str).str.strip()
    data = convert_labels(data)
    data = data[data["URL"] != ""].reset_index(drop=True)

    print("\nNormalizing URLs and removing invalid entries...")
    normalized_urls = []
    for url in data["URL"]:
        try:
            normalized_urls.append(normalize_url(url))
        except (ValueError, TypeError, AttributeError):
            normalized_urls.append(None)
    data["normalized_url"] = normalized_urls
    invalid_count = int(data["normalized_url"].isna().sum())
    data = data.dropna(subset=["normalized_url"]).reset_index(drop=True)
    print("Invalid URLs removed:", invalid_count)

    # Identify conflicting normalized URLs before deduplication. Exclude the
    # entire ambiguous group instead of arbitrarily keeping whichever label
    # happens to occur first in the CSV.
    label_counts = data.groupby("normalized_url")["Label"].nunique()
    conflicting_urls = set(label_counts[label_counts > 1].index)
    print("Normalized URLs with conflicting labels excluded:", len(conflicting_urls))
    if conflicting_urls:
        data = data[~data["normalized_url"].isin(conflicting_urls)].copy()

    # Each normalized URL now appears once. Use the normalized URL itself for
    # feature extraction to make the grouping key and model input consistent.
    data = data.drop_duplicates(subset=["normalized_url"]).copy()
    data["URL"] = data["normalized_url"]
    data = data.drop(columns=["normalized_url"]).reset_index(drop=True)

    print("\nLabel distribution after normalization and conflict removal:")
    print(data["Label"].value_counts().sort_index().rename(index={0: "good (0)", 1: "bad (1)"}))
    print("Unique normalized URLs retained:", len(data))

    if data.empty or data["Label"].nunique() != 2:
        raise ValueError("Need valid URLs from both classes to compare models.")

    # Validate feature extraction BEFORE splitting. Some URLs can pass the
    # normalizer but still fail inside the full feature extractor (for example,
    # during its security checks). Exclude those rows now so the split and
    # feature matrices stay aligned and the run cannot fail halfway through.
    print("\nChecking feature extraction for each retained normalized URL...")
    feature_rows = []
    feature_errors = []
    for url in data["URL"]:
        try:
            values = extract_features(url)
            feature_rows.append(np.asarray(values, dtype=np.float32))
            feature_errors.append(None)
        except Exception as exc:
            feature_rows.append(None)
            feature_errors.append(f"{type(exc).__name__}: {exc}")

    feature_ok = np.array([row is not None for row in feature_rows], dtype=bool)
    print("URLs rejected by feature extraction:", int((~feature_ok).sum()))
    if (~feature_ok).any():
        print("Examples rejected (up to 5):")
        bad_indices = np.flatnonzero(~feature_ok)[:5]
        for idx in bad_indices:
            print(" ", repr(data.iloc[idx]["URL"]), "->", feature_errors[idx])

    data = data.loc[feature_ok].reset_index(drop=True)
    feature_rows = [row for row in feature_rows if row is not None]
    if data.empty or data["Label"].nunique() != 2:
        raise ValueError("Need feature-extractable URLs from both classes to compare models.")

    # Split only after normalized URL deduplication and successful feature
    # extraction, so each URL stays in one split and feature rows remain aligned.
    train_indices, test_indices = train_test_split(
        np.arange(len(data)),
        test_size=0.20,
        stratify=data["Label"],
        random_state=RANDOM_STATE,
    )
    train_data = data.iloc[train_indices].reset_index(drop=True)
    test_data = data.iloc[test_indices].reset_index(drop=True)
    all_features = np.asarray(feature_rows, dtype=np.float32)
    X_train_unbalanced = all_features[train_indices]
    X_test = all_features[test_indices]
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    overlap = set(train_data["URL"]) & set(test_data["URL"])
    print("Normalized URLs shared between train and test:", len(overlap))
    if overlap:
        raise RuntimeError("Leakage check failed: normalized URLs overlap across splits.")

    # Match compare_models.py: upsample only the minority class in training data.
    train_good = train_data[train_data["Label"] == 0]
    train_bad = train_data[train_data["Label"] == 1]
    if len(train_good) == 0 or len(train_bad) == 0:
        raise ValueError("Training split must contain both classes.")

    if len(train_good) > len(train_bad):
        train_bad = train_bad.sample(
            n=len(train_good), replace=True, random_state=RANDOM_STATE
        )
    elif len(train_bad) > len(train_good):
        train_good = train_good.sample(
            n=len(train_bad), replace=True, random_state=RANDOM_STATE
        )

    balanced_train = pd.concat([train_good, train_bad], ignore_index=True)
    balanced_train = balanced_train.sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print("\nTraining label distribution after balancing:")
    print(balanced_train["Label"].value_counts().sort_index())
    print("Training rows:", len(balanced_train), "| Test rows:", len(test_data))

    # Reuse precomputed feature vectors. Map each balanced training URL back
    # to its precomputed vector; upsampled duplicates intentionally repeat rows.
    print("\nPreparing feature matrices from validated feature vectors...")
    feature_by_url = {url: features for url, features in zip(data["URL"], all_features)}
    X_train = np.asarray(
        [feature_by_url[url] for url in balanced_train["URL"]],
        dtype=np.float32,
    )
    y_train = balanced_train["Label"].to_numpy(dtype=int)
    y_test = test_data["Label"].to_numpy(dtype=int)

    if X_train.ndim != 2 or X_test.ndim != 2 or X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Feature extraction produced incompatible feature matrices.")

    print("Training feature shape:", X_train.shape)
    print("Test feature shape:", X_test.shape)

    print("\nTraining Random Forest baseline...")
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    print("Training XGBoost...")
    negative = int(np.sum(y_train == 0))
    positive = int(np.sum(y_train == 1))
    scale_pos_weight = negative / positive if positive else 1.0
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

    results = pd.DataFrame([
        {"Model": "Random Forest", **calculate_metrics(y_test, rf_prob)},
        {"Model": "XGBoost", **calculate_metrics(y_test, xgb_prob)},
    ])

    print("\n=== RANDOM FOREST vs XGBOOST ===")
    print(results[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]].to_string(index=False))

    results_csv = os.path.join(RESULTS_DIR, "model_comparison_rf_vs_xgboost_leakage_resistant.csv")
    results.to_csv(results_csv, index=False)

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11, 5.5))
    rf_values = results.loc[results["Model"] == "Random Forest", metrics].iloc[0].to_numpy()
    xgb_values = results.loc[results["Model"] == "XGBoost", metrics].iloc[0].to_numpy()
    bars1 = ax.bar(x - width / 2, rf_values, width, label="Random Forest")
    bars2 = ax.bar(x + width / 2, xgb_values, width, label="XGBoost")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Random Forest vs XGBoost")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "model_comparison_rf_vs_xgboost_leakage_resistant.png")
    fig.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved results:", results_csv)
    print("Saved graph:", chart_path)
    print("Saved label distribution:", os.path.join(
        RESULTS_DIR, "model_comparison_label_distribution_leakage_resistant.csv"
    ))


if __name__ == "__main__":
    main()
