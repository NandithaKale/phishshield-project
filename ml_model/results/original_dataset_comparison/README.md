# PhishShield Seen-Test Comparison

This report uses the existing seen-test results from the
original PhishShield dataset. No external dataset is used,
and models are not retrained by this graph-generation script.

- Test samples per model: 101,178
- Successfully evaluated models/ensembles: 21
- Positive class: phishing (label 1)
- Decision threshold: 0.50

## Best observed results

- Accuracy: RF + CNN (95.97%)
- Phishing recall: RF + CNN (87.73%)
- F1-score: RF + CNN (90.68%)
- ROC-AUC: CNN-LSTM (98.57%)

## Graphs

- accuracy_comparison.png
- precision_recall_f1_comparison.png
- roc_auc_comparison.png
- false_positive_rate_comparison.png
- confusion_matrices.png

These are seen-test results and should not be interpreted
as external or unseen-data performance.
