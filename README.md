# Machine learning based network intrusion detection

Development of an evaluation environment and comparative analysis of detection methods.

This repository contains a clean Python evaluation environment for a two-author Bachelor's thesis project. It supports experiments with CIC-IDS2017 and UNSW-NB15, comparing Logistic Regression, Random Forest, XGBoost, and Isolation Forest under normal and harder evaluation settings.

No real thesis results are included. If the real datasets are missing, the code can run in sample mode using synthetic demo data for smoke testing only.

## Authors

- Rubayat Kabir
- Md Mohiuddin Islam

## Dataset Warning

Raw CIC-IDS2017 and UNSW-NB15 files are not included in this repository. Place downloaded CSV files under:

- `data/raw/cicids2017/`
- `data/raw/unsw_nb15/`

Do not commit raw datasets, processed datasets, trained models, or generated results to GitHub.

## Folder Structure

```text
configs/       YAML experiment configurations
data/          Local dataset placement only; raw data is ignored by Git
docs/          Methodology, workflow, and experiment notes
notebooks/     Colab runner notebook
results/       Generated tables, figures, and metadata
scripts/       Quickstart helper scripts
src/           Python source code
tests/         Pytest tests
```

## Installation

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Sample Test

Sample mode uses generated synthetic data and is only for smoke testing:

```bash
python src/run_all.py --configs configs --sample --pattern "within_*.yaml"
```

Run tests:

```bash
python -m pytest
```

## Google Colab

In Colab, clone or upload the repository, install dependencies, then run:

```bash
pip install -r requirements.txt
python -m pytest
python src/run_all.py --configs configs --sample --pattern "within_*.yaml"
python src/run_all.py --configs configs --sample --pattern "cross_*.yaml"
```

See `scripts/colab_quickstart.md` and `notebooks/colab_runner.ipynb` for a fuller workflow with Google Drive.

## University Server

On a server, create a virtual environment, install requirements, place raw CSV files in `data/raw/`, and run limited real-data experiments first:

```bash
python -m pytest
python src/run_all.py --configs configs --max-rows 50000 --pattern "within_*.yaml"
python src/run_all.py --configs configs --max-rows 50000 --pattern "cross_*.yaml"
```

Remove `--max-rows` only after confirming the pipeline works and the server has enough memory and time.

## Real Dataset Commands

```bash
python src/run_all.py --configs configs --max-rows 50000
python src/run_all.py --configs configs --pattern "leave_one_attack_out_*.yaml"
python src/run_all.py --configs configs --pattern "threshold_analysis.yaml"
```

## Experiments

- Within-dataset evaluation: train and test on one dataset using a train/test split.
- Cross-dataset evaluation: train on one dataset and test on another using documented shared feature alignment.
- Leave-one-attack-class-out evaluation: hold out one attack category at a time to test generalization to unseen attacks.
- Threshold and false-alarm analysis: evaluate score thresholds and expected alert counts under low attack base rates.

## Output Files

Generated outputs are saved under `results/`:

- `results/tables/{experiment_name}_metrics.csv`
- `results/tables/{experiment_name}_per_attack_recall.csv`
- `results/tables/{experiment_name}_thresholds.csv`
- `results/figures/{experiment_name}_f1_comparison.png`
- `results/figures/{experiment_name}_fpr_comparison.png`
- `results/metadata/{experiment_name}_metadata.json`

Sample-mode outputs are marked as sample/demo in metadata and experiment names.

## AI Use

AI tools were used for planning, coding assistance, debugging support, and language improvement. Final code, experiments, interpretation, and thesis claims must be checked and accepted by the authors.

## GitHub Workflow

Use a `main` branch for stable work, feature branches for changes, pull requests for review, and meaningful commits. Never push raw datasets or generated experiment outputs.
