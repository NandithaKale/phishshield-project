
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# PHISHSHIELD: ISOLATED 200K DATASET EXPERIMENT
# ============================================================

SEED = 42
MAX_URL_LENGTH = 300
VOCAB_SIZE = 130
BATCH_SIZE = 256
MAX_EPOCHS = 6

ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    ROOT / "ml_model" / "external_data"
    / "labelled_test" / "phishing_url_dataset.csv"
)

EXTERNAL_FILE = (
    ROOT / "ml_model" / "external_data"
    / "alternative_test" / "clean_external_test.csv"
)

OUT_DIR = ROOT / "ml_model" / "results" / "experiment_200k_v1"


def normalize_url(value):
    if pd.isna(value):
        return ""

    url = str(value).strip()
    if not url:
        return ""

    if "://" not in url:
        url = "https://" + url

    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        scheme = (parts.scheme or "https").lower()

        if not host:
            return url.lower().split("#", 1)[0]

        # Retain a non-default port when valid.
        try:
            port = parts.port
        except ValueError:
            port = None

        if ":" in host and not host.startswith("["):
            host = f"[{host}]"

        netloc = host
        if port is not None:
            netloc += f":{port}"

        return (
            f"{scheme}://{netloc}{parts.path}"
            f"{'?' + parts.query if parts.query else ''}"
        )
    except Exception:
        return url.lower().split("#", 1)[0]


def get_host(url):
    try:
        return (urlsplit(url).hostname or "").lower()
    except Exception:
        return ""


def load_training_data():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    df = pd.read_csv(TRAIN_FILE, usecols=["URL", "Label"])
    df["URL"] = df["URL"].map(normalize_url)

    # SOURCE DATASET MAPPING:
    # Label 0 = phishing; Label 1 = legitimate.
    # INTERNAL MODEL MAPPING:
    # target 0 = legitimate; target 1 = phishing.
    source_label = pd.to_numeric(df["Label"], errors="coerce")
    df["target"] = source_label.map({0: 1, 1: 0})

    df = df[
        (df["URL"] != "") & df["target"].notna()
    ].copy()

    df["target"] = df["target"].astype("int32")
    df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)
    df["host"] = df["URL"].map(get_host)
    df = df[df["host"] != ""].reset_index(drop=True)

    if df["target"].nunique() != 2:
        raise ValueError(
            "Training dataset must contain source labels 0 and 1. "
            f"Found: {sorted(df['target'].unique().tolist())}"
        )

    return df[["URL", "target", "host"]]


def load_external_data():
    if not EXTERNAL_FILE.exists():
        raise FileNotFoundError(f"External test file not found: {EXTERNAL_FILE}")

    df = pd.read_csv(EXTERNAL_FILE)

    if not {"URL", "Label"}.issubset(df.columns):
        raise ValueError(
            "Expected URL and Label columns in cleaned external test file. "
            f"Found: {list(df.columns)}"
        )

    df = df[["URL", "Label"]].copy()
    df["URL"] = df["URL"].map(normalize_url)

    # CLEAN EXTERNAL DATASET MAPPING:
    # Label 0 = legitimate; Label 1 = phishing.
    df["target"] = pd.to_numeric(df["Label"], errors="coerce")

    df = df[
        (df["URL"] != "")
        & df["target"].isin([0, 1])
    ].copy()

    df["target"] = df["target"].astype("int32")
    df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)

    if df["target"].nunique() != 2:
        raise ValueError("External dataset must contain both classes.")

    return df[["URL", "target"]]


def lexical_features(url):
    """Fast, reproducible URL features for this separate RF experiment."""
    url = normalize_url(url)
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    path = parsed.path or ""

    suspicious_words = (
        "login", "secure", "verify", "account", "bank",
        "update", "free", "bonus", "paypal", "signin",
        "confirm", "security", "alert",
    )
    suspicious_tlds = (
        ".xyz", ".tk", ".ml", ".ga", ".cf",
        ".gq", ".top", ".biz", ".info",
    )

    try:
        ip_host = int(
            bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host))
        )
    except Exception:
        ip_host = 0

    return [
        len(url),
        url.count("."),
        url.count("-"),
        url.count("/"),
        url.count("="),
        int(url.startswith("https://")),
        int(url.startswith("http://")),
        int("@" in url),
        sum(c.isdigit() for c in url),
        int(any(w in url.lower() for w in suspicious_words)),
        sum(url.lower().count(w) for w in suspicious_words),
        int(any(tld in host.lower() for tld in suspicious_tlds)),
        host.count("."),
        int("-" in host),
        sum(c.isdigit() for c in host),
        len(host),
        int(len(url) > 60),
        ip_host,
        len(path.split("/")),
    ]


def encode_urls(urls):
    encoded = np.zeros((len(urls), MAX_URL_LENGTH), dtype=np.int32)

    for i, url in enumerate(urls):
        for j, char in enumerate(normalize_url(url)[:MAX_URL_LENGTH]):
            code = ord(char)
            encoded[i, j] = code + 2 if 0 < code < 128 else 1

    return encoded


def build_cnn():
    inputs = tf.keras.Input(shape=(MAX_URL_LENGTH,), dtype="int32")
    embedding = tf.keras.layers.Embedding(
        VOCAB_SIZE, 32, mask_zero=False
    )(inputs)

    branches = []
    for kernel in (3, 5, 7):
        branch = tf.keras.layers.Conv1D(
            64, kernel, padding="same", activation="relu"
        )(embedding)
        branch = tf.keras.layers.GlobalMaxPooling1D()(branch)
        branches.append(branch)

    x = tf.keras.layers.Concatenate()(branches)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(96, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, output, name="PhishShield_200k_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype("int32")
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    try:
        auc = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        auc = None

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
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate data, labels and split without training models.",
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    print("=" * 65)
    print("PHISHSHIELD: 200K EXPERIMENT")
    print("=" * 65)
    print(f"Training file: {TRAIN_FILE}")
    print(f"External file: {EXTERNAL_FILE}")
    print(f"Output folder: {OUT_DIR}")

    train_df = load_training_data()
    external_df = load_external_data()

    print(f"\nUnique training URLs: {len(train_df):,}")
    print("Training labels (0=legitimate, 1=phishing):")
    print(train_df["target"].value_counts().sort_index().to_string())

    print(f"\nUnique external URLs: {len(external_df):,}")
    print("External labels (0=legitimate, 1=phishing):")
    print(external_df["target"].value_counts().sort_index().to_string())

    # Check exact normalized URL overlap before any training.
    overlap = set(train_df["URL"]) & set(external_df["URL"])
    print(f"\nExact normalized URL overlap: {len(overlap):,}")

    if overlap:
        raise RuntimeError(
            "Training and external test still share exact URLs. "
            "Stop and clean the external test before training."
        )

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.15, random_state=SEED
    )
    train_idx, val_idx = next(
        splitter.split(
            train_df["URL"],
            train_df["target"],
            groups=train_df["host"],
        )
    )

    training = train_df.iloc[train_idx].reset_index(drop=True)
    validation = train_df.iloc[val_idx].reset_index(drop=True)

    host_overlap = set(training["host"]) & set(validation["host"])
    if host_overlap:
        raise RuntimeError("Hostname overlap found in training/validation.")

    if (
        training["target"].nunique() != 2
        or validation["target"].nunique() != 2
    ):
        raise RuntimeError(
            "A split does not contain both classes. No models were trained."
        )

    print(f"\nTraining split: {len(training):,}")
    print(f"Validation split: {len(validation):,}")
    print(f"Training/validation hostname overlap: {len(host_overlap)}")

    if args.check_only:
        print("\nCHECK PASSED. No models were trained or saved.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # --------------------------------------------------------
    # Random Forest using 19 lightweight lexical features.
    # --------------------------------------------------------
    print("\nExtracting Random Forest features...")
    X_train_rf = np.asarray(
        [lexical_features(u) for u in training["URL"]],
        dtype=np.float32,
    )
    X_val_rf = np.asarray(
        [lexical_features(u) for u in validation["URL"]],
        dtype=np.float32,
    )
    X_ext_rf = np.asarray(
        [lexical_features(u) for u in external_df["URL"]],
        dtype=np.float32,
    )

    y_train = training["target"].to_numpy()
    y_val = validation["target"].to_numpy()
    y_ext = external_df["target"].to_numpy()

    print("\nTraining separate Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(X_train_rf, y_train)

    rf_val_prob = rf.predict_proba(X_val_rf)[:, 1]
    rf_ext_prob = rf.predict_proba(X_ext_rf)[:, 1]
    rf.save_path = None  # No effect on model; artifact saved below.

    import joblib
    joblib.dump(rf, OUT_DIR / "random_forest_lexical.joblib")

    # --------------------------------------------------------
    # Character CNN using the same training/validation split.
    # --------------------------------------------------------
    print("\nEncoding URLs for the CNN. This may take time...")
    X_train_cnn = encode_urls(training["URL"].tolist())
    X_val_cnn = encode_urls(validation["URL"].tolist())
    X_ext_cnn = encode_urls(external_df["URL"].tolist())

    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    class_weights = {
        int(c): float(w) for c, w in zip(classes, weights)
    }

    print("\nBuilding and training the separate Character CNN...")
    cnn = build_cnn()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=0.00001,
            verbose=1,
        ),
    ]

    history = cnn.fit(
        X_train_cnn,
        y_train,
        validation_data=(X_val_cnn, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )

    cnn.save(OUT_DIR / "character_cnn_200k.keras")

    results = []
    for name, model, X_val, X_ext in [
        ("RandomForest_lexical", rf, X_val_rf, X_ext_rf),
        ("CharacterCNN", cnn, X_val_cnn, X_ext_cnn),
    ]:
        if name == "RandomForest_lexical":
            val_prob = model.predict_proba(X_val)[:, 1]
            ext_prob = model.predict_proba(X_ext)[:, 1]
        else:
            val_prob = model.predict(
                X_val, batch_size=512, verbose=0
            ).reshape(-1)
            ext_prob = model.predict(
                X_ext, batch_size=512, verbose=0
            ).reshape(-1)

        for split_name, y_true, probs in [
            ("grouped_validation", y_val, val_prob),
            ("clean_external_test", y_ext, ext_prob),
        ]:
            row = metrics(y_true, probs, threshold=0.5)
            row["model"] = name
            row["dataset"] = split_name
            results.append(row)

            print(f"\n{name} — {split_name} (threshold 0.50)")
            print(json.dumps(row, indent=2))

        pd.DataFrame({
            "URL": external_df["URL"],
            "actual_label": y_ext,
            "phishing_probability": ext_prob,
            "predicted_label_at_0_50": (ext_prob >= 0.5).astype(int),
        }).to_csv(
            OUT_DIR / f"{name}_external_predictions.csv",
            index=False,
        )

    pd.DataFrame(results).to_csv(OUT_DIR / "metrics.csv", index=False)

    run_info = {
        "seed": SEED,
        "training_file": str(TRAIN_FILE),
        "external_file": str(EXTERNAL_FILE),
        "source_training_label_mapping": "0=phishing, 1=legitimate",
        "internal_label_mapping": "0=legitimate, 1=phishing",
        "training_urls": int(len(training)),
        "validation_urls": int(len(validation)),
        "external_urls": int(len(external_df)),
        "exact_url_overlap": len(overlap),
        "hostname_overlap_train_validation": len(host_overlap),
        "cnn_epochs_completed": len(history.history.get("loss", [])),
        "threshold_used_for_report": 0.5,
        "rf_features": "19 lightweight lexical features; not the production security-aware extractor",
        "runtime_seconds": round(time.time() - start, 2),
        "warning": (
            "The 200k dataset was used in prior threshold experiments. "
            "The external dataset is a stronger check, but exact URL "
            "deduplication alone does not guarantee domain/campaign independence."
        ),
    }

    with open(OUT_DIR / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETE")
    print(f"Results: {OUT_DIR}")
    print(f"Metrics: {OUT_DIR / 'metrics.csv'}")
    print(f"Runtime: {(time.time() - start) / 60:.1f} minutes")
    print("Original models were not overwritten.")
    print("=" * 65)


if __name__ == "__main__":
    main()
