import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../"
        )
    )
)

from shared.feature_extractor import extract_features


# ==========================================================
# LOAD DATASET
# ==========================================================

data = pd.read_csv(
    "ml_model/dataset/phishing.csv",
    encoding="latin1",
).dropna(subset=["URL", "Label"])


# ==========================================================
# CONVERT LABELS
# ==========================================================

data["Label"] = (
    data["Label"]
    .astype(str)
    .str.lower()
    .map({
        "good": 0,
        "bad": 1,
    })
)

data = data.dropna(
    subset=["Label"]
)

data["Label"] = data["Label"].astype(int)


# ==========================================================
# ORIGINAL LABEL COUNTS
# ==========================================================

print("\nOriginal label counts:")
print(data["Label"].value_counts())


# ==========================================================
# BALANCE DATASET
# ==========================================================

df_good = data[
    data["Label"] == 0
]

df_bad = data[
    data["Label"] == 1
]

df_bad_upsampled = resample(
    df_bad,
    replace=True,
    n_samples=len(df_good),
    random_state=42,
)

data = pd.concat(
    [
        df_good,
        df_bad_upsampled,
    ],
    ignore_index=True,
)


print("\nBalanced label counts:")
print(data["Label"].value_counts())


# ==========================================================
# VALIDATE URLS
# ==========================================================
#
# The enhanced extractor now performs URL normalization.
# Some rows in the dataset may contain malformed URLs.
#
# We remove only URLs that cannot be processed while keeping
# their corresponding labels together.
# ==========================================================

print("\nValidating URLs...")

valid_urls = []
valid_labels = []

invalid_count = 0

for index, row in data.iterrows():

    url = str(row["URL"])
    label = int(row["Label"])

    try:

        # Test whether the enhanced extractor can process
        # this URL successfully.
        extract_features(url)

        valid_urls.append(url)
        valid_labels.append(label)

    except (
        ValueError,
        TypeError,
        AttributeError,
    ) as e:

        invalid_count += 1

        # Print only the first 20 invalid URLs.
        if invalid_count <= 20:

            print(
                f"\nInvalid URL at index {index}:"
            )

            print(
                f"URL: {repr(url)}"
            )

            print(
                f"Reason: {e}"
            )


print("\nURL validation complete.")

print(
    "Valid URLs:",
    len(valid_urls)
)

print(
    "Invalid URLs removed:",
    invalid_count
)


# ==========================================================
# REBUILD CLEAN DATA
# ==========================================================

if len(valid_urls) == 0:
    raise ValueError(
        "No valid URLs remain after URL validation."
    )


data = pd.DataFrame({
    "URL": valid_urls,
    "Label": valid_labels,
})


# Make sure both classes still exist.

if data["Label"].nunique() < 2:
    raise ValueError(
        "Only one class remains after URL validation."
    )


# ==========================================================
# EXTRACT ENHANCED FEATURES
# ==========================================================
#
# IMPORTANT:
# The enhanced extractor produces 19 features:
#
# Original 17
# +
# Direct IP
# +
# Look-alike domain
# ==========================================================

print("\nExtracting enhanced features...")

X = []

for index, url in enumerate(data["URL"]):

    features = extract_features(
        str(url)
    )

    X.append(features)

    if (
        (index + 1) % 10000 == 0
        or index + 1 == len(data)
    ):

        print(
            f"Processed "
            f"{index + 1}/"
            f"{len(data)} URLs"
        )


# Convert to NumPy array.

X = np.asarray(
    X,
    dtype=float,
)


y = data[
    "Label"
].to_numpy(
    dtype=int
)


# ==========================================================
# VERIFY FEATURE COUNT
# ==========================================================

print(
    "\nFeature matrix shape:",
    X.shape
)

if X.shape[1] != 19:

    raise ValueError(
        f"Expected 19 features, "
        f"but received {X.shape[1]} features."
    )

print(
    "Feature count confirmed:",
    X.shape[1]
)


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ==========================================================
# RANDOM FOREST MODEL
# ==========================================================

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


# ==========================================================
# TRAIN MODEL
# ==========================================================

print(
    "\nTraining enhanced Random Forest model..."
)

model.fit(
    X_train,
    y_train,
)


# ==========================================================
# PREDICTIONS
# ==========================================================

predictions = model.predict(
    X_test
)


# ==========================================================
# EVALUATION
# ==========================================================

print(
    "\n=== Enhanced Model Evaluation ==="
)

print(
    "Accuracy :",
    round(
        accuracy_score(
            y_test,
            predictions,
        ),
        4,
    ),
)

print(
    "Precision:",
    round(
        precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        4,
    ),
)

print(
    "Recall   :",
    round(
        recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        4,
    ),
)

print(
    "F1 Score :",
    round(
        f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        4,
    ),
)


# ==========================================================
# SAVE MODEL
# ==========================================================

save_path = (
    "ml_model/saved_model/url_model.pkl"
)

os.makedirs(
    os.path.dirname(save_path),
    exist_ok=True,
)

joblib.dump(
    model,
    save_path,
)


print(
    f"\nEnhanced model saved to: {save_path}"
)

print(
    "Feature count:",
    model.n_features_in_,
)