
from __future__ import annotations

import os
import sys
import time
import warnings
import traceback
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ml_model" / "final_all_models"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = ROOT / "ml_model" / "dataset" / "phishing.csv"

N_FEATURES = 19
SEED = 42
MAX_NEURAL_TRAIN = 160_000
MAX_URL_LEN = 200
VOCAB_SIZE = 128


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------
# 2. Feature extractor and dataset loading
# --------------------------------------------------

def load_extractor():
    sys.path.insert(0, str(ROOT))
    from shared.feature_extractor import extract_features
    return extract_features


def url_text(value):
    return "" if pd.isna(value) else str(value).strip()


def internal_label(value):
    if pd.isna(value):
        return None

    s = str(value).strip().lower()

    if s == "good":
        return 0
    if s == "bad":
        return 1
    if s in ("0", "0.0"):
        return 0
    if s in ("1", "1.0"):
        return 1

    return None


def load_internal():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {TRAIN_FILE}")

    df = pd.read_csv(
        TRAIN_FILE,
        usecols=["URL", "Label"],
        on_bad_lines="warn",
    )

    df["URL"] = df["URL"].map(url_text)
    df["target"] = df["Label"].map(internal_label)

    df = df[
        (df["URL"] != "") & df["target"].notna()
    ].copy()

    df["target"] = df["target"].astype(int)

    return (
        df.drop_duplicates("URL")
        .reset_index(drop=True)[["URL", "target"]]
    )


def feature_matrix(urls, extractor, cache, name):
    rows = []
    invalid = 0

    for i, raw_url in enumerate(urls):
        url = url_text(raw_url)

        if url not in cache:
            try:
                features = np.asarray(
                    extractor(url), dtype=np.float32
                ).reshape(-1)

                if (
                    not url
                    or features.size != N_FEATURES
                    or not np.isfinite(features).all()
                ):
                    raise ValueError("Invalid URL features")

            except Exception as exc:
                invalid += 1
                features = np.zeros(
                    N_FEATURES, dtype=np.float32
                )

                if invalid <= 5:
                    log(
                        f"Warning {name}, row {i}: "
                        f"{url[:70]!r} -> zero features ({exc})"
                    )

            cache[url] = features

        rows.append(cache[url])

        if (i + 1) % 50000 == 0:
            log(f"{name}: processed {i + 1:,}/{len(urls):,}")

    X = (
        np.vstack(rows).astype(np.float32)
        if rows
        else np.empty((0, N_FEATURES), dtype=np.float32)
    )

    log(f"{name}: X={X.shape}; invalid rows={invalid:,}")
    return X


# --------------------------------------------------
# 3. Convert URLs into character sequences
# --------------------------------------------------

def url_chars(urls):
    X = np.zeros(
        (len(urls), MAX_URL_LEN), dtype=np.int32
    )

    for i, url in enumerate(urls):
        for j, char in enumerate(str(url)[:MAX_URL_LEN]):
            code = ord(char)
            X[i, j] = (
                code if 0 < code < VOCAB_SIZE else 1
            )

    return X


# --------------------------------------------------
# 4. Neural network architectures
# --------------------------------------------------

def make_tf_model(kind):
    import tensorflow as tf

    inputs = tf.keras.Input(
        shape=(MAX_URL_LEN,), dtype="int32"
    )

    x = tf.keras.layers.Embedding(
        VOCAB_SIZE, 32, mask_zero=False
    )(inputs)

    if kind == "Character CNN":
        x = tf.keras.layers.Conv1D(
            64, 5, padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)

    elif kind == "CNN-LSTM":
        x = tf.keras.layers.Conv1D(
            64, 5, padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32)
        )(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)

    elif kind == "FNN":
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.25)(x)

    elif kind == "DNN":
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)

    elif kind == "Wide & Deep":
        wide = tf.keras.layers.Flatten()(x)
        deep = tf.keras.layers.GlobalAveragePooling1D()(x)
        deep = tf.keras.layers.Dense(128, activation="relu")(deep)
        x = tf.keras.layers.Concatenate()([wide, deep])

    elif kind == "URL Transformer":
        positions = tf.range(
            start=0, limit=MAX_URL_LEN, delta=1
        )
        pos_embedding = tf.keras.layers.Embedding(
            MAX_URL_LEN, 32
        )(positions)

        x = x + pos_embedding

        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=16
        )(x, x)

        x = tf.keras.layers.LayerNormalization()(x + attention)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)

    else:
        raise ValueError(f"Unknown neural model: {kind}")

    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid"
    )(x)

    model = tf.keras.Model(
        inputs, outputs, name=kind.replace(" ", "_")
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# --------------------------------------------------
# 5. Traditional machine learning models
# --------------------------------------------------

def models_classic():
    models = {
        "Logistic Regression": make_pipeline(
            SimpleImputer(),
            StandardScaler(),
            LogisticRegression(
                max_iter=700,
                class_weight="balanced",
                solver="liblinear",
                random_state=SEED,
            ),
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=160,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=SEED,
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=160,
            n_jobs=-1,
            class_weight="balanced",
            random_state=SEED,
        ),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=SEED,
        ),

        "Gaussian Naive Bayes": make_pipeline(
            SimpleImputer(), GaussianNB()
        ),

        "Linear SVM (calibrated)": make_pipeline(
            SimpleImputer(),
            StandardScaler(),
            CalibratedClassifierCV(
                LinearSVC(
                    class_weight="balanced",
                    random_state=SEED,
                ),
                cv=3,
            ),
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=140,
            learning_rate=0.08,
            random_state=SEED,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=2,
            random_state=SEED,
        ),

        "MLP (feature-based)": make_pipeline(
            SimpleImputer(),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=35,
                early_stopping=True,
                random_state=SEED,
            ),
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=180,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.9,
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist",
            random_state=SEED,
        )
    except Exception as exc:
        log(f"XGBoost unavailable: {exc}")

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=180,
            learning_rate=0.08,
            num_leaves=31,
            verbosity=-1,
            n_jobs=-1,
            random_state=SEED,
            class_weight="balanced",
        )
    except Exception as exc:
        log(f"LightGBM unavailable: {exc}")

    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = CatBoostClassifier(
            iterations=180,
            depth=7,
            learning_rate=0.08,
            verbose=False,
            thread_count=-1,
            random_seed=SEED,
            auto_class_weights="Balanced",
        )
    except Exception as exc:
        log(f"CatBoost unavailable: {exc}")

    try:
        from pytorch_tabnet.tab_model import TabNetClassifier

        models["TabNet"] = TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=3,
            gamma=1.3,
            seed=SEED,
            verbose=0,
        )
    except Exception as exc:
        log(f"TabNet unavailable: {exc}")

    return models


# --------------------------------------------------
# 6. Evaluation metrics
# --------------------------------------------------

def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (
            np.ptp(scores) + 1e-12
        )

    return model.predict(X).astype(float)


def score_metrics(model_name, y_true, predictions, scores,
                  train_seconds=None, inference_seconds=None):
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = np.nan

    return {
        "model": model_name,
        "dataset": "SeenTest",
        "n_samples": len(y_true),
        "accuracy": accuracy_score(y_true, predictions),
        "precision_phishing": precision_score(
            y_true, predictions, zero_division=0
        ),
        "phishing_recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1_phishing": f1_score(
            y_true, predictions, zero_division=0
        ),
        "false_positive_rate": (
            fp / (fp + tn) if (fp + tn) else np.nan
        ),
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "status": "OK",
    }


# --------------------------------------------------
# 7. Main seen-test-only comparison
# --------------------------------------------------

def main():
    log("Loading internal dataset only")

    extractor = load_extractor()
    internal = load_internal()

    log(
        f"Cleaned URLs: {len(internal):,}; "
        f"class counts: "
        f"{internal.target.value_counts().sort_index().to_dict()}"
    )

    train_df, test_df = train_test_split(
        internal,
        test_size=0.20,
        random_state=SEED,
        stratify=internal.target,
    )

    cache = {}

    X_train = feature_matrix(
        train_df.URL.tolist(), extractor, cache, "SeenTrain"
    )
    X_test = feature_matrix(
        test_df.URL.tolist(), extractor, cache, "SeenTest"
    )

    y_train = train_df.target.to_numpy(dtype=np.int32)
    y_test = test_df.target.to_numpy(dtype=np.int32)

    classic_models = models_classic()

    seen_rows = []
    status_rows = []
    seen_scores = {}

    # Traditional/tabular models
    for name, model in classic_models.items():
        try:
            log(f"[SEEN] Training {name}")

            start = time.perf_counter()

            if name == "TabNet":
                model.fit(
                    X_train.astype(np.float32),
                    y_train,
                    max_epochs=35,
                    patience=7,
                    batch_size=2048,
                    virtual_batch_size=256,
                )
            else:
                weighted_models = (
                    "HistGradientBoosting",
                    "Gradient Boosting",
                )

                if name in weighted_models:
                    weights = compute_sample_weight(
                        "balanced", y_train
                    )
                    model.fit(
                        X_train, y_train, sample_weight=weights
                    )
                else:
                    model.fit(X_train, y_train)

            train_seconds = time.perf_counter() - start

            start = time.perf_counter()
            scores = get_scores(model, X_test)
            inference_seconds = time.perf_counter() - start

            predictions = (scores >= 0.5).astype(int)

            seen_rows.append(
                score_metrics(
                    name,
                    y_test,
                    predictions,
                    scores,
                    train_seconds,
                    inference_seconds,
                )
            )

            seen_scores[name] = scores

            status_rows.append({
                "model": name,
                "family": "traditional/tabular",
                "seen_status": "OK",
                "notes": "",
            })

        except Exception as exc:
            log(f"ERROR training {name}: {exc}")
            traceback.print_exc()

            status_rows.append({
                "model": name,
                "family": "traditional/tabular",
                "seen_status": "ERROR",
                "notes": str(exc),
            })

    # Neural URL models
    neural_names = [
        "FNN",
        "DNN",
        "Wide & Deep",
        "Character CNN",
        "CNN-LSTM",
        "URL Transformer",
    ]

    try:
        import tensorflow as tf

        neural_train = train_df

        if len(neural_train) > MAX_NEURAL_TRAIN:
            neural_train, _ = train_test_split(
                train_df,
                train_size=MAX_NEURAL_TRAIN,
                random_state=SEED,
                stratify=train_df.target,
            )

        X_neural = url_chars(neural_train.URL.tolist())
        y_neural = neural_train.target.to_numpy(dtype=np.int32)
        X_test_chars = url_chars(test_df.URL.tolist())

        for name in neural_names:
            try:
                log(f"[SEEN] Training {name}")

                tf.keras.backend.clear_session()
                model = make_tf_model(name)

                start = time.perf_counter()

                model.fit(
                    X_neural,
                    y_neural,
                    epochs=4,
                    batch_size=512,
                    validation_split=0.05,
                    verbose=0,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss",
                            patience=1,
                            restore_best_weights=True,
                        )
                    ],
                )

                train_seconds = time.perf_counter() - start

                start = time.perf_counter()
                scores = model.predict(
                    X_test_chars,
                    batch_size=1024,
                    verbose=0,
                ).reshape(-1)
                inference_seconds = time.perf_counter() - start

                predictions = (scores >= 0.5).astype(int)

                seen_rows.append(
                    score_metrics(
                        name,
                        y_test,
                        predictions,
                        scores,
                        train_seconds,
                        inference_seconds,
                    )
                )

                seen_scores[name] = scores

                status_rows.append({
                    "model": name,
                    "family": "character/neural",
                    "seen_status": "OK",
                    "notes": (
                        f"Trained on up to {MAX_NEURAL_TRAIN:,} "
                        "training URLs for CPU runtime."
                    ),
                })

                del model

            except Exception as exc:
                log(f"Neural model {name} failed: {exc}")
                traceback.print_exc()

                status_rows.append({
                    "model": name,
                    "family": "character/neural",
                    "seen_status": "ERROR",
                    "notes": str(exc),
                })

    except Exception as exc:
        log(f"TensorFlow neural models unavailable: {exc}")

        for name in neural_names:
            status_rows.append({
                "model": name,
                "family": "character/neural",
                "seen_status": "NOT RUN",
                "notes": str(exc),
            })

    # Probability-average ensembles on the same held-out test split
    ensemble_specs = [
        ("RF + XGBoost", ["Random Forest", "XGBoost"]),
        ("RF + CNN", ["Random Forest", "Character CNN"]),
        ("CNN + XGBoost", ["Character CNN", "XGBoost"]),
        (
            "RF + CNN + XGBoost",
            ["Random Forest", "Character CNN", "XGBoost"],
        ),
    ]

    for ensemble_name, members in ensemble_specs:
        if all(member in seen_scores for member in members):
            scores = np.mean(
                [seen_scores[member] for member in members],
                axis=0,
            )

            seen_rows.append(
                score_metrics(
                    ensemble_name,
                    y_test,
                    (scores >= 0.5).astype(int),
                    scores,
                )
            )

            status = "OK"
            notes = "Mean of member scores on the same test split."
        else:
            status = "NOT RUN"
            notes = "One or more required models were unavailable."

        status_rows.append({
            "model": ensemble_name,
            "family": "ensemble",
            "seen_status": status,
            "notes": notes,
        })

    # Advanced models requiring unavailable input types
    for name, note in [
        (
            "GNN",
            "NOT RUN: graph nodes and edges are not supplied.",
        ),
        (
            "Vision Transformer",
            "NOT RUN: webpage screenshots are not supplied.",
        ),
        (
            "Multimodal Fusion",
            "NOT RUN: aligned URL, HTML/DOM and screenshot data "
            "are not supplied.",
        ),
    ]:
        status_rows.append({
            "model": name,
            "family": "research/advanced",
            "seen_status": "NOT RUN",
            "notes": note,
        })

    # Save only seen-test results
    seen_df = pd.DataFrame(seen_rows)
    status_df = pd.DataFrame(status_rows)

    seen_path = OUT / "seen_test_metrics.csv"
    status_path = OUT / "model_status.csv"
    workbook_path = OUT / "phishshield_all_models.xlsx"

    seen_df.to_csv(seen_path, index=False)
    status_df.to_csv(status_path, index=False)

    try:
        with pd.ExcelWriter(
            workbook_path, engine="openpyxl"
        ) as writer:
            seen_df.to_excel(
                writer, sheet_name="Seen Test", index=False
            )
            status_df.to_excel(
                writer, sheet_name="Model Coverage", index=False
            )

            pd.DataFrame([{
                "internal_training_rows": len(train_df),
                "seen_test_rows": len(test_df),
                "split": "Stratified 80/20",
                "seed": SEED,
                "note": (
                    "Seen-test-only comparison. "
                    "No external/unseen evaluation performed."
                ),
            }]).to_excel(
                writer, sheet_name="Run Info", index=False
            )

    except Exception as exc:
        log(f"Workbook creation failed: {exc}")

    log("SEEN-TEST COMPARISON COMPLETE")
    log(f"Seen metrics: {seen_path}")
    log(f"Model coverage: {status_path}")
    log(f"Workbook: {workbook_path}")


if __name__ == "__main__":
    main()
