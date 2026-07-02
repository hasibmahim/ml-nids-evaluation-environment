# Results Interpretation Guide

High within-dataset scores do not prove deployment readiness. Models can perform well when training and test data come from the same dataset but fail when traffic distributions change.

Cross-dataset performance drops are expected. A drop may indicate dataset shift, feature mismatch, or overfitting to dataset-specific artifacts.

False positive rate matters because network traffic volume is high. Even a low FPR can create many alerts when the attack base rate is very small.

PR-AUC is important for imbalanced data because it focuses on performance for the positive attack class.

Isolation Forest should be interpreted differently from supervised models. It is trained as an anomaly detector, often on benign traffic, and does not learn attack labels in the same way as Logistic Regression, Random Forest, or XGBoost.

Do not report sample-mode results as thesis results. Sample mode exists only to verify that the code runs.
