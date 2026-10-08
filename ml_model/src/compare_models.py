import os
import sys
import random
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import EarlyStopping


warnings.filterwarnings("ignore")


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../"
    )
)

sys.path.append(PROJECT_ROOT)


DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "ml_model",
    "dataset",
    "phishing.csv"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "ml_model",
    "results"
)


# ==========================================================
# IMPORT YOUR EXISTING PHISHSHIELD COMPONENTS
# ==========================================================

from shared.feature_extractor import extract_features

from shared.url_normalizer import normalize_url


# ==========================================================
# SETTINGS
# ==========================================================

RANDOM_STATE = 42

MAX_URL_LENGTH = 200

VOCAB_SIZE = 128


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

random.seed(RANDOM_STATE)

np.random.seed(RANDOM_STATE)

tf.random.set_seed(RANDOM_STATE)


# ==========================================================
# FEATURE NAMES
# ==========================================================

FEATURE_NAMES = [

    "URL Length",
    "Number of Dots",
    "Number of Hyphens",
    "Number of Slashes",
    "Number of Equal Signs",
    "HTTPS Present",
    "HTTP Without HTTPS",
    "@ Symbol",
    "Digit Count",
    "Suspicious Word Present",
    "Suspicious Word Count",
    "Suspicious TLD",
    "Multiple Subdomains",
    "Hyphen in Domain",
    "Digit in Domain",
    "Domain Length",
    "URL Length > 60",
    "Direct IP Address",
    "Look-alike Domain",
]


# ==========================================================
# LABEL CONVERSION
# ==========================================================

def convert_labels(data):

    data["Label"] = (

        data["Label"]
        .astype(str)
        .str.lower()
        .map({
            "good": 0,
            "bad": 1,
            "0": 0,
            "1": 1,
        })

    )

    data = data.dropna(
        subset=["Label"]
    )

    data["Label"] = (
        data["Label"].astype(int)
    )

    return data


# ==========================================================
# URL VALIDATION
# ==========================================================

def is_valid_url(url):

    try:

        normalize_url(
            str(url)
        )

        return True

    except (
        ValueError,
        TypeError,
        AttributeError,
    ):

        return False


# ==========================================================
# CNN URL ENCODING
# ==========================================================

def encode_url(url):

    url = str(url).lower()

    encoded = []

    for character in url[:MAX_URL_LENGTH]:

        value = ord(character)

        if value >= VOCAB_SIZE:

            value = VOCAB_SIZE - 1

        encoded.append(value)


    while len(encoded) < MAX_URL_LENGTH:

        encoded.append(0)


    return encoded


# ==========================================================
# BUILD CHARACTER CNN
# ==========================================================

def build_cnn():

    model = Sequential([

        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=64,
        ),

        Conv1D(
            filters=128,
            kernel_size=5,
            activation="relu",
        ),

        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
        ),

        GlobalMaxPooling1D(),

        Dense(
            64,
            activation="relu",
        ),

        Dropout(0.30),

        Dense(
            1,
            activation="sigmoid",
        ),
    ])


    model.compile(

        optimizer="adam",

        loss="binary_crossentropy",

        metrics=["accuracy"],
    )


    return model


# ==========================================================
# CALCULATE METRICS
# ==========================================================

def calculate_metrics(
    y_true,
    probabilities,
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)


    accuracy = accuracy_score(
        y_true,
        predictions,
    )


    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )


    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )


    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )


    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )


    cm = confusion_matrix(
        y_true,
        predictions,
    )


    return {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc,

        "TN": cm[0][0],

        "FP": cm[0][1],

        "FN": cm[1][0],

        "TP": cm[1][1],
    }


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\n")

    print("=" * 80)

    print(
        "PHISHSHIELD - 7 MODEL COMPARISON"
    )

    print("=" * 80)


    # ======================================================
    # 1. LOAD DATASET
    # ======================================================

    print("\n[1] Loading phishing.csv...")


    data = pd.read_csv(

        DATASET_PATH,

        encoding="latin1",

    )


    print(
        "Original dataset shape:",
        data.shape,
    )


    if "URL" not in data.columns:

        raise ValueError(
            "URL column not found."
        )


    if "Label" not in data.columns:

        raise ValueError(
            "Label column not found."
        )


    data = data[
        ["URL", "Label"]
    ].dropna()


    data["URL"] = (
        data["URL"]
        .astype(str)
        .str.strip()
    )


    # ======================================================
    # 2. CONVERT LABELS
    # ======================================================

    data = convert_labels(
        data
    )


    print("\nOriginal label distribution:")

    print(
        data["Label"].value_counts()
    )


    # ======================================================
    # 3. REMOVE DUPLICATES
    # ======================================================

    old_count = len(data)


    data = data.drop_duplicates(
        subset=["URL"]
    )


    print(
        "\nDuplicate URLs removed:",
        old_count - len(data)
    )


    # ======================================================
    # 4. VALIDATE URLS
    # ======================================================
    #
    # IMPORTANT:
    # Your existing PhishShield normalizer only accepts
    # HTTP and HTTPS URLs.
    #
    # Invalid URLs are removed BEFORE feature extraction.
    #
    # ======================================================

    print(
        "\n[2] Validating URLs..."
    )


    valid_mask = data["URL"].map(
        is_valid_url
    )


    invalid_count = (
        (~valid_mask).sum()
    )


    data = data[
        valid_mask
    ].reset_index(
        drop=True
    )


    print(
        "Invalid URLs removed:",
        invalid_count
    )


    print(
        "Valid URLs remaining:",
        len(data)
    )


    # ======================================================
    # 5. COMMON 80/20 SPLIT
    # ======================================================

    print(
        "\n[3] Creating common 80/20 train-test split..."
    )


    train_data, test_data = train_test_split(

        data,

        test_size=0.20,

        stratify=data["Label"],

        random_state=RANDOM_STATE,
    )


    train_data = train_data.reset_index(
        drop=True
    )

    test_data = test_data.reset_index(
        drop=True
    )


    print(
        "Training samples:",
        len(train_data)
    )

    print(
        "Testing samples:",
        len(test_data)
    )


    # ======================================================
    # 6. BALANCE TRAINING DATA ONLY
    # ======================================================

    print(
        "\n[4] Balancing training data..."
    )


    train_good = train_data[
        train_data["Label"] == 0
    ]


    train_bad = train_data[
        train_data["Label"] == 1
    ]


    if len(train_good) == 0:

        raise ValueError(
            "No good URLs in training data."
        )


    if len(train_bad) == 0:

        raise ValueError(
            "No phishing URLs in training data."
        )


    if len(train_good) > len(train_bad):

        train_bad = train_bad.sample(

            n=len(train_good),

            replace=True,

            random_state=RANDOM_STATE,
        )

    elif len(train_bad) > len(train_good):

        train_good = train_good.sample(

            n=len(train_bad),

            replace=True,

            random_state=RANDOM_STATE,
        )


    balanced_train = pd.concat(

        [
            train_good,
            train_bad,
        ],

        ignore_index=True,
    )


    balanced_train = balanced_train.sample(

        frac=1,

        random_state=RANDOM_STATE,

    ).reset_index(
        drop=True
    )


    print(
        "\nBalanced training distribution:"
    )

    print(
        balanced_train["Label"].value_counts()
    )


    # ======================================================
    # 7. EXTRACT 19 FEATURES
    # ======================================================

    print(
        "\n[5] Extracting existing 19 features..."
    )


    X_train_features = []

    X_test_features = []


    # ------------------------------------------------------
    # TRAIN FEATURES
    # ------------------------------------------------------

    for i, url in enumerate(
        balanced_train["URL"]
    ):

        features = extract_features(
            url
        )

        X_train_features.append(
            features
        )


        if (
            (i + 1) % 10000 == 0
        ):

            print(
                "Training URLs:",
                i + 1,
                "/",
                len(balanced_train)
            )


    # ------------------------------------------------------
    # TEST FEATURES
    # ------------------------------------------------------

    for i, url in enumerate(
        test_data["URL"]
    ):

        features = extract_features(
            url
        )

        X_test_features.append(
            features
        )


        if (
            (i + 1) % 10000 == 0
        ):

            print(
                "Testing URLs:",
                i + 1,
                "/",
                len(test_data)
            )


    X_train_features = np.asarray(

        X_train_features,

        dtype=np.float32,
    )


    X_test_features = np.asarray(

        X_test_features,

        dtype=np.float32,
    )


    y_train = balanced_train[
        "Label"
    ].to_numpy(
        dtype=int
    )


    y_test = test_data[
        "Label"
    ].to_numpy(
        dtype=int
    )


    # ======================================================
    # 8. VERIFY 19 FEATURES
    # ======================================================

    print(
        "\nTraining feature shape:",
        X_train_features.shape
    )

    print(
        "Testing feature shape:",
        X_test_features.shape
    )


    if X_train_features.shape[1] != 19:

        raise ValueError(

            "ERROR: Expected 19 features but received "
            + str(X_train_features.shape[1])

        )


    print(
        "\n19 FEATURES CONFIRMED."
    )


    # ======================================================
    # 9. CNN INPUT
    # ======================================================

    print(
        "\n[6] Preparing Character CNN input..."
    )


    cnn_train = np.asarray(

        [
            encode_url(url)

            for url in balanced_train["URL"]

        ],

        dtype=np.int32,
    )


    cnn_test = np.asarray(

        [
            encode_url(url)

            for url in test_data["URL"]

        ],

        dtype=np.int32,
    )


    print(
        "CNN training shape:",
        cnn_train.shape
    )

    print(
        "CNN testing shape:",
        cnn_test.shape
    )


    # ======================================================
    # MODEL 1 - RANDOM FOREST
    # ======================================================

    print("\n")

    print("=" * 80)

    print(
        "MODEL 1/7 - RANDOM FOREST"
    )

    print("=" * 80)


    rf_model = RandomForestClassifier(

        n_estimators=500,

        max_depth=25,

        min_samples_split=3,

        min_samples_leaf=1,

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1,
    )


    rf_model.fit(

        X_train_features,

        y_train,
    )


    rf_prob = rf_model.predict_proba(

        X_test_features

    )[:, 1]


    rf_metrics = calculate_metrics(

        y_test,

        rf_prob,
    )


    print(
        rf_metrics
    )


    # ======================================================
    # MODEL 2 - XGBOOST
    # ======================================================

    print("\n")

    print("=" * 80)

    print(
        "MODEL 2/7 - XGBOOST"
    )

    print("=" * 80)


    negative = np.sum(
        y_train == 0
    )

    positive = np.sum(
        y_train == 1
    )


    scale_pos_weight = (

        negative / positive

        if positive > 0

        else 1
    )


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


    xgb_model.fit(

        X_train_features,

        y_train,
    )


    xgb_prob = xgb_model.predict_proba(

        X_test_features

    )[:, 1]


    xgb_metrics = calculate_metrics(

        y_test,

        xgb_prob,
    )


    print(
        xgb_metrics
    )


    # ======================================================
    # MODEL 3 - CHARACTER CNN
    # ======================================================

    print("\n")

    print("=" * 80)

    print(
        "MODEL 3/7 - CHARACTER CNN"
    )

    print("=" * 80)


    cnn_model = build_cnn()


    early_stopping = EarlyStopping(

        monitor="val_loss",

        patience=2,

        restore_best_weights=True,
    )


    cnn_model.fit(

        cnn_train,

        y_train,

        validation_split=0.10,

        epochs=8,

        batch_size=512,

        callbacks=[
            early_stopping
        ],

        verbose=1,
    )


    cnn_prob = cnn_model.predict(

        cnn_test,

        batch_size=512,

        verbose=0,

    ).reshape(-1)


    cnn_metrics = calculate_metrics(

        y_test,

        cnn_prob,
    )


    print(
        cnn_metrics
    )


    # ======================================================
    # MODEL 4 - RF + XGBOOST
    # ======================================================

    print("\n")

    print("=" * 80)

    print(
        "MODEL 4/7 - RF + XGBOOST"
    )

    print("=" * 80)


    rf_xgb_prob = (

        rf_prob + xgb_prob

    ) / 2


    rf_xgb_metrics = calculate_metrics(

        y_test,

        rf_xgb_prob,
    )


    print(
        rf_xgb_metrics
    )


    # ======================================================
    # MODEL 5 - RF + CNN
    # ======================================================

    print("\n")

    print("=" * 80)

    print(
        "MODEL 5/7 - RF + CNN"
    )

    print("=" * 80)


    rf_cnn_prob = (

        rf_prob + cnn_prob

    ) / 2


    rf_cnn_metrics = calculate_metrics(

        y_test,

        rf_cnn_prob,
    )


    print(
        rf_cnn_metrics
    )


    # ======================================================
    # MODEL 6 - CNN + XGBOOST
    # ======================================================

    print("\n")

    print("=" * 80)

    print(
        "MODEL 6/7 - CNN + XGBOOST"
    )

    print("=" * 80)


    cnn_xgb_prob = (

        cnn_prob + xgb_prob

    ) / 2


    cnn_xgb_metrics = calculate_metrics(

        y_test,

        cnn_xgb_prob,
    )


    print(
        cnn_xgb_metrics
    )


    # ======================================================
    # MODEL 7 - RF + CNN + XGBOOST
    # ======================================================

    print("\n")

    print("=" * 80)

    print(
        "MODEL 7/7 - RF + CNN + XGBOOST"
    )

    print("=" * 80)


    all_prob = (

        rf_prob
        +
        cnn_prob
        +
        xgb_prob

    ) / 3


    all_metrics = calculate_metrics(

        y_test,

        all_prob,
    )


    print(
        all_metrics
    )


    # ======================================================
    # FINAL RESULTS TABLE
    # ======================================================

    results = pd.DataFrame(

        [

            {
                "Model":
                    "Random Forest",

                **rf_metrics,
            },

            {
                "Model":
                    "XGBoost",

                **xgb_metrics,
            },

            {
                "Model":
                    "Character CNN",

                **cnn_metrics,
            },

            {
                "Model":
                    "RF + XGBoost",

                **rf_xgb_metrics,
            },

            {
                "Model":
                    "RF + CNN",

                **rf_cnn_metrics,
            },

            {
                "Model":
                    "CNN + XGBoost",

                **cnn_xgb_metrics,
            },

            {
                "Model":
                    "RF + CNN + XGBoost",

                **all_metrics,
            },

        ]

    )


    # ======================================================
    # DISPLAY RESULTS
    # ======================================================

    print("\n")

    print("=" * 110)

    print(
        "FINAL PHISHSHIELD MODEL COMPARISON"
    )

    print("=" * 110)


    print(

        results[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
            ]
        ].to_string(
            index=False
        )

    )


    # ======================================================
    # SAVE RESULTS
    # ======================================================

    os.makedirs(

        RESULTS_DIR,

        exist_ok=True,
    )


    results_path = os.path.join(

        RESULTS_DIR,

        "model_comparison.csv"
    )


    results.to_csv(

        results_path,

        index=False,
    )


    # ======================================================
    # SAVE MODELS
    # ======================================================

    joblib.dump(

        rf_model,

        os.path.join(

            RESULTS_DIR,

            "random_forest_comparison.pkl"
        )
    )


    joblib.dump(

        xgb_model,

        os.path.join(

            RESULTS_DIR,

            "xgboost_comparison.pkl"
        )
    )


    cnn_model.save(

        os.path.join(

            RESULTS_DIR,

            "character_cnn_comparison.keras"
        )
    )


    # ======================================================
    # RANDOM FOREST FEATURE IMPORTANCE
    # ======================================================

    importance = pd.DataFrame(

        {

            "Feature":
                FEATURE_NAMES,

            "Importance":
                rf_model.feature_importances_,

        }

    ).sort_values(

        "Importance",

        ascending=False,
    )


    importance.to_csv(

        os.path.join(

            RESULTS_DIR,

            "random_forest_feature_importance.csv"
        ),

        index=False,
    )


    # ======================================================
    # BEST MODEL
    # ======================================================

    best_model = results.loc[

        results["F1 Score"].idxmax()

    ]


    print("\n")

    print("=" * 80)

    print(
        "BEST MODEL BASED ON F1 SCORE"
    )

    print("=" * 80)


    print(
        "Model:",
        best_model["Model"]
    )

    print(
        "Accuracy:",
        round(
            best_model["Accuracy"],
            4
        )
    )

    print(
        "Precision:",
        round(
            best_model["Precision"],
            4
        )
    )

    print(
        "Recall:",
        round(
            best_model["Recall"],
            4
        )
    )

    print(
        "F1 Score:",
        round(
            best_model["F1 Score"],
            4
        )
    )

    print(
        "ROC-AUC:",
        round(
            best_model["ROC-AUC"],
            4
        )
    )


    print("\n")

    print(
        "Results saved to:"
    )

    print(
        results_path
    )


    print("\n")

    print(
        "7-MODEL COMPARISON COMPLETED SUCCESSFULLY."
    )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    main()