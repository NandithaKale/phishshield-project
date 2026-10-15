
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# --------------------------------------------------
# PHISHSHIELD: SEEN-TEST GRAPHS ONLY
# Uses existing metrics; does not retrain models.
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "ml_model"
    / "final_all_models"
    / "seen_test_metrics.csv"
)

OUTPUT_DIR = (
    ROOT
    / "ml_model"
    / "results"
    / "original_dataset_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Seen-test metrics not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    # Keep only successfully evaluated seen-test models.
    df = df[
        (df["dataset"] == "SeenTest")
        & (df["status"] == "OK")
    ].copy()

    if df.empty:
        raise ValueError("No valid seen-test results found.")

    metrics = [
        "accuracy",
        "precision_phishing",
        "phishing_recall",
        "f1_phishing",
        "false_positive_rate",
        "roc_auc",
    ]

    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["accuracy"])

    # Save the exact data used for the graphs.
    df.to_csv(
        OUTPUT_DIR / "seen_test_metrics.csv",
        index=False
    )

    # 1. Accuracy comparison
    make_bar_chart(
        df,
        "accuracy",
        "Seen-Test Accuracy",
        "accuracy_comparison.png"
    )

    # 2. Precision, recall and F1-score
    make_grouped_chart(
        df,
        [
            "precision_phishing",
            "phishing_recall",
            "f1_phishing",
        ],
        "Seen-Test Phishing Metrics",
        "precision_recall_f1_comparison.png"
    )

    # 3. ROC-AUC
    make_bar_chart(
        df,
        "roc_auc",
        "Seen-Test ROC-AUC",
        "roc_auc_comparison.png"
    )

    # 4. False positive rate
    make_bar_chart(
        df,
        "false_positive_rate",
        "Seen-Test False Positive Rate",
        "false_positive_rate_comparison.png"
    )

    # 5. Confusion matrices
    make_confusion_matrices(df)

    # Summary report
    best_accuracy = df.loc[df["accuracy"].idxmax()]
    best_recall = df.loc[df["phishing_recall"].idxmax()]
    best_f1 = df.loc[df["f1_phishing"].idxmax()]
    best_auc = df.loc[df["roc_auc"].idxmax()]

    report = f"""# PhishShield Seen-Test Comparison

This report uses the existing seen-test results from the
original PhishShield dataset. No external dataset is used,
and models are not retrained by this graph-generation script.

- Test samples per model: {int(df["n_samples"].iloc[0]):,}
- Successfully evaluated models/ensembles: {len(df)}
- Positive class: phishing (label 1)
- Decision threshold: 0.50

## Best observed results

- Accuracy: {best_accuracy["model"]} ({best_accuracy["accuracy"]:.2%})
- Phishing recall: {best_recall["model"]} ({best_recall["phishing_recall"]:.2%})
- F1-score: {best_f1["model"]} ({best_f1["f1_phishing"]:.2%})
- ROC-AUC: {best_auc["model"]} ({best_auc["roc_auc"]:.2%})

## Graphs

- accuracy_comparison.png
- precision_recall_f1_comparison.png
- roc_auc_comparison.png
- false_positive_rate_comparison.png
- confusion_matrices.png

These are seen-test results and should not be interpreted
as external or unseen-data performance.
"""

    (OUTPUT_DIR / "README.md").write_text(
        report,
        encoding="utf-8"
    )

    print("\nSeen-test graphs generated successfully.")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Models included: {len(df)}")


def make_bar_chart(df, metric, title, filename):
    plot_df = df.sort_values(metric)

    fig, ax = plt.subplots(
        figsize=(12, max(7, len(plot_df) * 0.35))
    )

    ax.barh(plot_df["model"], plot_df[metric])
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)


def make_grouped_chart(df, columns, title, filename):
    plot_df = df.sort_values("f1_phishing")
    y = np.arange(len(plot_df))
    height = 0.24

    fig, ax = plt.subplots(
        figsize=(13, max(7, len(plot_df) * 0.4))
    )

    offsets = [-height, 0, height]

    for offset, column in zip(offsets, columns):
        ax.barh(
            y + offset,
            plot_df[column],
            height=height,
            label=column.replace("_", " ").title()
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["model"])
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)


def make_confusion_matrices(df):
    required = ["tn", "fp", "fn", "tp"]

    if not all(col in df.columns for col in required):
        print("Confusion-matrix columns unavailable; skipped.")
        return

    cols = 3
    rows = int(np.ceil(len(df) / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(13, rows * 3.1),
        squeeze=False
    )

    axes = axes.flatten()

    for i, (_, model) in enumerate(df.iterrows()):
        ax = axes[i]

        matrix = np.array([
            [model["tn"], model["fp"]],
            [model["fn"], model["tp"]]
        ])

        ax.imshow(matrix, aspect="auto")
        ax.set_title(model["model"], fontsize=9)

        ax.set_xticks(
            [0, 1],
            ["Pred. Legit", "Pred. Phishing"],
            fontsize=7
        )
        ax.set_yticks(
            [0, 1],
            ["Actual Legit", "Actual Phishing"],
            fontsize=7
        )

        for row in range(2):
            for col in range(2):
                ax.text(
                    col,
                    row,
                    f"{int(matrix[row, col]):,}",
                    ha="center",
                    va="center",
                    fontsize=8
                )

    for i in range(len(df), len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        "Seen-Test Confusion Matrices",
        y=1.002
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "confusion_matrices.png",
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
