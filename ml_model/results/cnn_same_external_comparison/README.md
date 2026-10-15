\# PhishShield: CNN Candidate Comparison on Unseen Data



\## Objective

Compare the class-weighted Character CNN and the multi-kernel CNN on the same external dataset of 9,046 URLs.



\## Dataset

\- Dataset: `dataset\_web\_fraud.csv`

\- Location: `ml\_model/external\_data/alternative\_test/`

\- Columns: `nr`, `url`, `verdict`

\- Labels: 0 = legitimate, 1 = phishing

\- Both models were evaluated on the same test URLs using their respective preprocessing methods.



\## Results at threshold 0.50



| Metric | Class-weighted CNN | Multi-kernel CNN |

|---|---:|---:|

| Accuracy | 76.75% | 77.95% |

| Phishing precision | 76.35% | 76.17% |

| Phishing recall | 83.03% | 86.58% |

| F1-score | 79.55% | 81.05% |

| False positive rate | 30.75% | 32.38% |

| ROC-AUC | 86.54% | 85.28% |

| False negatives | 836 | 661 |

| False positives | 1,267 | 1,334 |



\## Analysis

The multi-kernel CNN achieved higher accuracy, phishing recall, and F1-score. It missed 175 fewer phishing URLs in this evaluation. However, it had slightly lower precision and ROC-AUC, and a higher false positive rate.



These results indicate a trade-off rather than universal superiority. The findings apply to this external dataset and should not be treated as proof of performance on every real-world URL.



\## Files

\- `metrics.csv`: numerical comparison results.

\- `compare\_cnn\_candidates\_external.py`: script used for the comparison.



\## Reproducibility note

The external dataset and trained model files are not included in this results folder. Reproducing the experiment requires the corresponding dataset, saved models, dependencies, and preprocessing setup.



