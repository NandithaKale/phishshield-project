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


# ==========================================================
# ADD PROJECT ROOT TO PATH
# ==========================================================

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
).dropna(
    subset=["URL", "Label"]
)


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

print("\n" + "=" * 60)
print("ORIGINAL LABEL DISTRIBUTION")
print("=" * 60)

print(
    data["Label"].value_counts()
    .sort_index()
)

print(
    "\nLabel 0 (Good / Legitimate):",
    (data["Label"] == 0).sum()
)

print(
    "Label 1 (Bad / Phishing):",
    (data["Label"] == 1).sum()
)


# ==========================================================
# VALIDATE URLS
# ==========================================================
#
# The enhanced extractor performs URL normalization.
# Some rows may contain malformed URLs.
#
# Invalid URLs are removed while keeping their labels
# together.
# ==========================================================

print("\n" + "=" * 60)
print("VALIDATING URLs")
print("=" * 60)

valid_urls = []
valid_labels = []

invalid_count = 0

for index, row in data.iterrows():

    url = str(row["URL"])
    label = int(row["Label"])

    try:

        # Test whether the enhanced extractor
        # can process this URL.
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


# ==========================================================
# CLEANED LABEL DISTRIBUTION
# ==========================================================

print("\n" + "=" * 60)
print("CLEANED LABEL DISTRIBUTION")
print("=" * 60)

print(
    data["Label"].value_counts()
    .sort_index()
)


# Make sure both classes exist.

if data["Label"].nunique() < 2:

    raise ValueError(
        "Only one class remains after URL validation."
    )


# ==========================================================
# EXTRACT 19 FEATURES
# ==========================================================
#
# The enhanced extractor produces 19 features:
#
# 1.  URL length
# 2.  Dot count
# 3.  Hyphen count
# 4.  Slash count
# 5.  Equal sign count
# 6.  HTTPS indicator
# 7.  HTTP indicator
# 8.  @ symbol
# 9.  Digit count
# 10. Suspicious word presence
# 11. Suspicious word count
# 12. Suspicious TLD
# 13. Multiple subdomain indicator
# 14. Hyphen in domain
# 15. Digit in domain
# 16. Domain length
# 17. Long URL indicator
# 18. Direct IP address
# 19. Look-alike domain
# ==========================================================

print("\n" + "=" * 60)
print("EXTRACTING 19 FEATURES")
print("=" * 60)

X = []

for index, url in enumerate(
    data["URL"]
):

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
#
# IMPORTANT:
#
# The dataset is split BEFORE oversampling.
#
# This prevents duplicated minority-class samples
# from appearing in both training and testing data.
#
# The test set remains completely untouched.
# ==========================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)


print(
    "Training samples before balancing:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


print(
    "\nTraining label distribution before balancing:"
)

print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)


print(
    "\nTesting label distribution:"
)

print(
    pd.Series(y_test)
    .value_counts()
    .sort_index()
)


# ==========================================================
# BALANCE TRAINING DATA ONLY
# ==========================================================
#
# Label 0 = Good / Legitimate
# Label 1 = Bad / Phishing
#
# The minority class is oversampled with replacement
# until it reaches the size of the majority class.
#
# The TEST SET IS NOT BALANCED.
# ==========================================================

print("\n" + "=" * 60)
print("BALANCING TRAINING DATA")
print("=" * 60)


# Separate training classes.

X_train_good = X_train[
    y_train == 0
]

y_train_good = y_train[
    y_train == 0
]


X_train_bad = X_train[
    y_train == 1
]

y_train_bad = y_train[
    y_train == 1
]


print(
    "Training Label 0 before balancing:",
    len(X_train_good)
)

print(
    "Training Label 1 before balancing:",
    len(X_train_bad)
)


# ----------------------------------------------------------
# Oversample minority class
# ----------------------------------------------------------
#
# In your dataset, Label 1 (phishing) is the minority class.
#
# If this changes in the future, the code below automatically
# identifies the minority class instead of assuming it.
# ----------------------------------------------------------

if len(X_train_good) > len(X_train_bad):

    # Label 1 is the minority class.

    X_train_bad_upsampled = resample(
        X_train_bad,
        replace=True,
        n_samples=len(X_train_good),
        random_state=42,
    )

    y_train_bad_upsampled = resample(
        y_train_bad,
        replace=True,
        n_samples=len(y_train_good),
        random_state=42,
    )

    X_train_balanced = np.vstack(
        [
            X_train_good,
            X_train_bad_upsampled,
        ]
    )

    y_train_balanced = np.concatenate(
        [
            y_train_good,
            y_train_bad_upsampled,
        ]
    )


elif len(X_train_bad) > len(X_train_good):

    # Label 0 is the minority class.

    X_train_good_upsampled = resample(
        X_train_good,
        replace=True,
        n_samples=len(X_train_bad),
        random_state=42,
    )

    y_train_good_upsampled = resample(
        y_train_good,
        replace=True,
        n_samples=len(y_train_bad),
        random_state=42,
    )

    X_train_balanced = np.vstack(
        [
            X_train_good_upsampled,
            X_train_bad,
        ]
    )

    y_train_balanced = np.concatenate(
        [
            y_train_good_upsampled,
            y_train_bad,
        ]
    )


else:

    # Already balanced.

    X_train_balanced = X_train.copy()

    y_train_balanced = y_train.copy()


# ==========================================================
# SHUFFLE BALANCED TRAINING DATA
# ==========================================================

shuffle_indices = np.random.RandomState(
    42
).permutation(
    len(X_train_balanced)
)

X_train_balanced = X_train_balanced[
    shuffle_indices
]

y_train_balanced = y_train_balanced[
    shuffle_indices
]


# ==========================================================
# PRINT BALANCED DISTRIBUTION
# ==========================================================

print(
    "\nTraining label distribution after balancing:"
)

print(
    pd.Series(y_train_balanced)
    .value_counts()
    .sort_index()
)

print(
    "\nBalanced training samples:",
    len(X_train_balanced)
)

print(
    "Testing samples remain untouched:",
    len(X_test)
)


# ==========================================================
# RANDOM FOREST MODEL
# ==========================================================

print("\n" + "=" * 60)
print("RANDOM FOREST MODEL")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=3,
    min_samples_leaf=1,
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
    X_train_balanced,
    y_train_balanced,
)


print(
    "Training completed."
)


# ==========================================================
# PREDICTIONS
# ==========================================================

print(
    "\nGenerating predictions..."
)

predictions = model.predict(
    X_test
)


# ==========================================================
# EVALUATION
# ==========================================================

print("\n" + "=" * 60)
print("ENHANCED RANDOM FOREST MODEL EVALUATION")
print("=" * 60)


accuracy = accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)


print(
    "Accuracy :",
    round(
        accuracy,
        4,
    ),
)

print(
    "Precision:",
    round(
        precision,
        4,
    ),
)

print(
    "Recall   :",
    round(
        recall,
        4,
    ),
)

print(
    "F1 Score :",
    round(
        f1,
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

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
