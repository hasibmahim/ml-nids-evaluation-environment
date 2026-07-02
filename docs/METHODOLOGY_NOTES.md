# Methodology Notes

## Within-Dataset Evaluation

The dataset is split into training and test partitions. Preprocessing is fitted only on the training partition and then applied to the test partition.

For CIC-IDS2017 row-limited runs, `max_rows` is applied as per-file sampling across the available day CSV files before concatenation and shuffling. This helps preserve attack-category coverage when attacks are distributed across different collection days.

## Cross-Dataset Evaluation

One dataset is used for training and another for testing. Feature alignment uses two strategies: strict normalized-name intersection and a manual mapping of common flow-level features such as duration, packet counts, byte counts, packet size means, and packet rate.

If fewer than three aligned features are found, the experiment fails with a clear error because the comparison would not be defensible.

## Leave-One-Attack-Class-Out

One attack category is held out for testing. Training uses benign flows and other attack categories. This tests whether models detect attack categories that were not present in training.

## Threshold and False-Alarm Analysis

Threshold analysis evaluates how score thresholds change recall, F1-score, and false positive rate. It also estimates false alerts per 100,000 flows under attack base rates of 1%, 0.1%, and 0.01%.

## Data Leakage Prevention

Scaling, imputation, and constant-feature removal are based only on the training data in each experiment. The test data is transformed using the fitted training pipeline.

## Limitations

Cross-dataset comparisons depend strongly on feature compatibility, dataset collection conditions, and label definitions. Results should be interpreted as evidence about generalization difficulty, not as direct deployment performance.
