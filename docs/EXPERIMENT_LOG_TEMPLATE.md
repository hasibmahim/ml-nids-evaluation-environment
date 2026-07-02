# Experiment Log Template

## Date

YYYY-MM-DD

## Machine Used

Example: Google Colab, Kaggle, university server, local workstation.

## Dataset Version/Source

- CIC-IDS2017:
- UNSW-NB15:

## Row Limit

Example: `50000` or full dataset.

## Models

- Logistic Regression
- Random Forest
- XGBoost
- Isolation Forest

## Experiment Type

Within-dataset, cross-dataset, leave-one-attack-class-out, or threshold analysis.

## Config File

Example: `configs/within_cicids2017.yaml`

## Command Used

```bash
python src/run_all.py --configs configs --max-rows 50000 --pattern "within_*.yaml"
```

## Output Files

- Metrics CSV:
- Per-attack recall CSV:
- Threshold CSV:
- Figures:
- Metadata JSON:

## Notes/Problems

Record warnings, failed models, memory issues, dataset cleaning decisions, and interpretation notes.
