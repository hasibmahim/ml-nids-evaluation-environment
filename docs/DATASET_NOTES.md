# Dataset Notes

## CIC-IDS2017

Raw CIC-IDS2017 files are not included. Place CSV files in:

```text
data/raw/cicids2017/
```

The loader looks for a label column such as `Label`, normalizes column names, creates `target`, and derives `attack_category`.

## UNSW-NB15

Raw UNSW-NB15 files are not included. Place CSV files in:

```text
data/raw/unsw_nb15/
```

The loader looks for a binary label column such as `label` and an attack category column such as `attack_cat`.

## Why Raw Data Is Excluded

Raw datasets are large and may have license, redistribution, and storage restrictions. Keeping them out of GitHub makes the repository easier to share and keeps the experiments reproducible from documented data placement instructions.
