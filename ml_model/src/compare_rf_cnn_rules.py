
import shutil
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

ROOT = Path("ml_model/results")

OUTPUT = (
    ROOT
    / "cnn_same_external_comparison"
    / "metrics.csv"
)

BACKUP = (
    ROOT
    / "cnn_same_external_comparison"
    / "metrics_backup_before_rf_cnn_rules.csv"
)

TEMP_OUTPUT = (
    ROOT
    / "cnn_same_external_comparison"
    / "metrics_new_temp.csv"
)

BASELINE_PREDICTIONS = (
    ROOT
    / "cnn_same_external_comparison"
    / "predictions.csv"
)

MULTIKERNEL_PREDICTIONS = (
    ROOT
    / "cnn_multikernel_candidate"
    / "external_predictions.csv"
)

RF_PREDICTIONS = (
    ROOT
    / "alternative_dataset_test"
    / "rf_predictions.csv"
)


# --------------------------------------------------
# 2. Check that input files exist
# --------------------------------------------------

for path in [
    OUTPUT,
    BASELINE_PREDICTIONS,
    MULTIKERNEL_PREDICTIONS,
    RF_PREDICTIONS,
]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

print("Reading prediction files...")


# --------------------------------------------------
# 3. Load existing predictions
# --------------------------------------------------

a = pd.read_csv(BASELINE_PREDICTIONS)
b = pd.read_csv(MULTIKERNEL_PREDICTIONS)
r = pd.read_csv(RF_PREDICTIONS)


def normalize_url(series):
    return series.astype(str).str.strip().str.lower()


a["k"] = normalize_url(a["url"])
b["k"] = normalize_url(b["URL"])
r["k"] = normalize_url(r["url"])


# --------------------------------------------------
# 4. Remove verified duplicate RF rows
# --------------------------------------------------

# Check duplicate rows before removing them.
duplicate_rows = r[r["k"].duplicated(keep=False)]

if not duplicate_rows.empty:
    for _, group in duplicate_rows.groupby("k"):
        if group["True_Phishing"].nunique() > 1:
            raise ValueError(
                "Duplicate RF URLs have conflicting true labels."
            )

        if group["RF_Prediction"].nunique() > 1:
            raise ValueError(
                "Duplicate RF URLs have conflicting predictions."
            )

r = r.drop_duplicates(subset="k", keep="first")


# --------------------------------------------------
# 5. Align predictions by exact URL
# --------------------------------------------------

d = (
    a[
        [
            "k",
            "actual_phishing",
            "baseline_prediction_0_50",
            "candidate_prediction_0_50",
        ]
    ]
    .merge(
        b[
            [
                "k",
                "actual_label",
                "predicted_label_at_0_50",
            ]
        ],
        on="k",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        r[
            [
                "k",
                "True_Phishing",
                "RF_Prediction",
            ]
        ],
        on="k",
        how="inner",
        validate="one_to_one",
    )
)


# --------------------------------------------------
# 6. Verify the evaluation data
# --------------------------------------------------

if len(d) != 8984:
    raise ValueError(
        f"Expected 8984 matching URLs, found {len(d)}. "
        "No output replacement was performed."
    )

if not (d["actual_phishing"] == d["actual_label"]).all():
    raise ValueError("CNN label mismatch. No replacement performed.")

if not (d["actual_phishing"] == d["True_Phishing"]).all():
    raise ValueError("RF label mismatch. No replacement performed.")

if not (
    d["candidate_prediction_0_50"]
    == d["predicted_label_at_0_50"]
).all():
    raise ValueError(
        "Multi-kernel CNN prediction mismatch. "
        "No replacement performed."
    )

print("Verified matching URLs:", len(d))
print("Labels and CNN predictions agree.")


# --------------------------------------------------
# 7. Prepare model predictions
# --------------------------------------------------

y = d["actual_phishing"].astype(int)

rf = d["RF_Prediction"].astype(int)
cw = d["baseline_prediction_0_50"].astype(int)
mk = d["candidate_prediction_0_50"].astype(int)


# OR: flag phishing if either model predicts phishing.
# AND: flag phishing only if both models predict phishing.

models = {
    "Random Forest": rf,
    "Class-weighted CNN": cw,
    "Multi-kernel CNN": mk,

    "RF + class-weighted CNN (OR)": (
        (rf == 1) | (cw == 1)
    ).astype(int),

    "RF + multi-kernel CNN (OR)": (
        (rf == 1) | (mk == 1)
    ).astype(int),

    "RF + class-weighted CNN (AND)": (
        (rf == 1) & (cw == 1)
    ).astype(int),

    "RF + multi-kernel CNN (AND)": (
        (rf == 1) & (mk == 1)
    ).astype(int),
}


# --------------------------------------------------
# 8. Calculate metrics
# --------------------------------------------------

rows = []

for name, pred in models.items():

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    rows.append(
        {
            "threshold": 0.5,
            "accuracy": accuracy_score(y, pred),
            "precision_phishing": precision_score(
                y, pred, zero_division=0
            ),
            "phishing_recall": recall_score(
                y, pred, zero_division=0
            ),
            "f1_phishing": f1_score(
                y, pred, zero_division=0
            ),
            "false_positive_rate": false_positive_rate,

            # ROC-AUC requires continuous scores.
            # These input files do not provide RF probabilities.
            "roc_auc": float("nan"),

            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "n": int(len(d)),
            "model": name,
        }
    )

result = pd.DataFrame(rows)

if len(result) != 7:
    raise ValueError("Expected exactly 7 model rows.")


# --------------------------------------------------
# 9. Preserve the original metrics file
# --------------------------------------------------

# Create the backup only if one does not already exist.
# Never overwrite the existing backup.

if not BACKUP.exists():
    shutil.copy2(OUTPUT, BACKUP)
    print("Original metrics backed up:", BACKUP)
else:
    print("Existing backup preserved:", BACKUP)


# --------------------------------------------------
# 10. Write temporary file and replace metrics.csv
# --------------------------------------------------

# Write to a separate file first so the original is not
# truncated if the write operation fails.

try:
    result.to_csv(TEMP_OUTPUT, index=False)
    print("Temporary metrics file created:", TEMP_OUTPUT)

    # Replace the target only after the temporary CSV
    # has been written successfully.
    TEMP_OUTPUT.replace(OUTPUT)

except PermissionError:
    print("\nERROR: Windows denied access to the output file.")
    print("Close metrics.csv in Excel or another application.")
    print("The temporary results, if created, are at:", TEMP_OUTPUT)
    print("The original metrics file has not been intentionally deleted.")
    raise


# --------------------------------------------------
# 11. Verify saved output
# --------------------------------------------------

saved = pd.read_csv(OUTPUT)

if len(saved) != 7:
    raise ValueError(
        "Output verification failed: expected 7 rows."
    )

if saved["n"].nunique() != 1 or int(saved["n"].iloc[0]) != 8984:
    raise ValueError(
        "Output verification failed: incorrect evaluation size."
    )

print("\nSUCCESS: metrics.csv replaced and verified.")
print("Models saved:", len(saved))
print("URLs evaluated per model:", len(d))
print("Updated file:", OUTPUT)
print("Backup file:", BACKUP)

print("\nUpdated comparison:")
print(saved.to_string(index=False))
