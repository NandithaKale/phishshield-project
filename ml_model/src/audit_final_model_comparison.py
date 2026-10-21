
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

BASE = Path("ml_model/results/final_model_comparison")
METRICS_FILE = BASE / "model_comparison_metrics.csv"
STATUS_FILE = BASE / "model_status.csv"
PRED = BASE / "predictions"

EXPECTED_MODELS = {
    "Logistic Regression",
    "Random Forest",
    "Extra Trees",
    "Linear SVM",
    "XGBoost",
    "LightGBM",
    "HistGradientBoosting",
    "CatBoost",
    "FNN",
    "DNN",
    "Wide & Deep",
    "TabNet",
    "Character CNN",
    "URL Transformer",
    "Random Forest + XGBoost",
    "Random Forest + Character CNN",
    "Character CNN + XGBoost",
    "Random Forest + Character CNN + XGBoost",
    "CNN-LSTM",
    "GNN",
    "Vision Transformer",
    "Multimodal Fusion",
}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def main():
    print("=" * 70)
    print("PHISHSHIELD FINAL MODEL COMPARISON AUDIT")
    print("=" * 70)

    if not METRICS_FILE.exists():
        print(f"ERROR: Metrics file not found: {METRICS_FILE}")
        return

    metrics = pd.read_csv(METRICS_FILE)

    if "model" not in metrics.columns:
        print("ERROR: Missing 'model' column in metrics CSV.")
        print("Available columns:", metrics.columns.tolist())
        return

    print("\n1. MODEL COUNT")
    print("Metrics rows:", len(metrics))
    print("Unique models:", metrics["model"].nunique())
    print("Duplicate model rows:", metrics["model"].duplicated().sum())

    actual_models = set(metrics["model"].dropna().astype(str))
    print("Missing models:", sorted(EXPECTED_MODELS - actual_models) or "None")
    print("Unexpected models:", sorted(actual_models - EXPECTED_MODELS) or "None")

    print("\n2. METRIC VALUE CHECK")

    metric_columns = [
        "accuracy",
        "precision_phishing",
        "recall_phishing",
        "f1_phishing",
        "false_positive_rate",
        "roc_auc",
    ]

    for column in metric_columns:
        if column not in metrics.columns:
            print(f"{column}: COLUMN MISSING")
            continue

        values = pd.to_numeric(metrics[column], errors="coerce")
        missing = int(values.isna().sum())
        out_of_range = int(((values < 0) | (values > 1)).sum())

        print(
            f"{column}: values={values.notna().sum()}, "
            f"missing={missing}, out_of_range={out_of_range}"
        )

    print("\n3. MODEL STATUS CHECK")

    if STATUS_FILE.exists():
        status = pd.read_csv(STATUS_FILE)
        print(status.to_string(index=False))
    else:
        print("Status file not found:", STATUS_FILE)

    print("\n4. HOLDOUT PREDICTION AUDIT")

    if not PRED.exists():
        print("ERROR: Prediction directory not found:", PRED)
        return

    audit_results = []

    for _, row in metrics.iterrows():
        model_name = str(row["model"])
        prediction_file = PRED / f"{slug(model_name)}_holdout.csv"

        if not prediction_file.exists():
            result = {
                "model": model_name,
                "prediction_file": prediction_file.name,
                "rows": 0,
                "result": "FAIL: prediction file missing",
            }
            audit_results.append(result)
            print(f"FAIL: {model_name} - prediction file missing")
            continue

        try:
            predictions = pd.read_csv(prediction_file)

            required = {"actual_label", "predicted_label"}
            if not required.issubset(predictions.columns):
                result = {
                    "model": model_name,
                    "prediction_file": prediction_file.name,
                    "rows": len(predictions),
                    "result": "FAIL: required columns missing",
                }
                audit_results.append(result)
                print(
                    f"FAIL: {model_name} - columns found: "
                    f"{predictions.columns.tolist()}"
                )
                continue

            if predictions.empty:
                raise ValueError("Prediction file is empty")

            y_true = pd.to_numeric(
                predictions["actual_label"], errors="raise"
            )
            y_pred = pd.to_numeric(
                predictions["predicted_label"], errors="raise"
            )

            if y_true.isna().any() or y_pred.isna().any():
                raise ValueError("Missing labels or predictions")

            if not set(y_true.unique()).issubset({0, 1}):
                raise ValueError("Unexpected actual-label values")

            if not set(y_pred.unique()).issubset({0, 1}):
                raise ValueError("Unexpected predicted-label values")

            calculated = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_phishing": precision_score(
                    y_true, y_pred, zero_division=0
                ),
                "recall_phishing": recall_score(
                    y_true, y_pred, zero_division=0
                ),
                "f1_phishing": f1_score(
                    y_true, y_pred, zero_division=0
                ),
            }

            tn, fp, fn, tp = confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            ).ravel()

            calculated["false_positive_rate"] = (
                fp / (fp + tn) if (fp + tn) else 0.0
            )

            score_column = (
                "phishing_score"
                if "phishing_score" in predictions.columns
                else None
            )

            if score_column and y_true.nunique() == 2:
                scores = pd.to_numeric(
                    predictions[score_column], errors="raise"
                )

                if scores.isna().any():
                    raise ValueError("Missing phishing scores")

                calculated["roc_auc"] = roc_auc_score(y_true, scores)

            mismatches = []

            for metric_name, calculated_value in calculated.items():
                if metric_name not in metrics.columns:
                    mismatches.append(
                        f"{metric_name}: absent from summary CSV"
                    )
                    continue

                expected_value = pd.to_numeric(
                    pd.Series([row[metric_name]]), errors="coerce"
                ).iloc[0]

                if pd.isna(expected_value):
                    mismatches.append(
                        f"{metric_name}: missing in summary"
                    )
                elif abs(calculated_value - expected_value) > 0.001:
                    mismatches.append(
                        f"{metric_name}: summary={expected_value:.6f}, "
                        f"recalculated={calculated_value:.6f}"
                    )

            if mismatches:
                result_text = "FAIL: metric mismatch"
                print(f"FAIL: {model_name}")
                for mismatch in mismatches:
                    print("  -", mismatch)
            else:
                result_text = "PASS: metrics verified"
                print(
                    f"PASS: {model_name} | rows={len(predictions)} | "
                    f"accuracy={calculated['accuracy']:.4f} | "
                    f"F1={calculated['f1_phishing']:.4f} | "
                    f"FPR={calculated['false_positive_rate']:.4f}"
                )

            audit_results.append({
                "model": model_name,
                "prediction_file": prediction_file.name,
                "rows": len(predictions),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "result": result_text,
            })

        except Exception as exc:
            audit_results.append({
                "model": model_name,
                "prediction_file": prediction_file.name,
                "rows": 0,
                "result": f"FAIL: {type(exc).__name__}: {exc}",
            })
            print(f"FAIL: {model_name} - {type(exc).__name__}: {exc}")

    audit_file = BASE / "audit_results.csv"
    audit_df = pd.DataFrame(audit_results)
    audit_df.to_csv(audit_file, index=False)

    print("\n5. FINAL AUDIT SUMMARY")
    if not audit_df.empty:
        print(audit_df["result"].value_counts().to_string())

    print("\nAudit report saved to:", audit_file)
    print("\nNOTE:")
    print("- This verifies saved predictions against reported metrics.")
    print("- It does not prove the dataset is free from leakage or bias.")
    print("- External testing is still required to evaluate generalization.")


if __name__ == "__main__":
    main()
