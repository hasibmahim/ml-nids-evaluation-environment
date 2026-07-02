# Google Colab Quickstart

## 1. Clone or Upload the Repository

Clone from GitHub:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
```

Or upload the repository ZIP and extract it in `/content`.

## 2. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

## 4. Run Sample Tests

```bash
python -m pytest
python src/run_all.py --configs configs --sample --pattern "within_*.yaml"
python src/run_all.py --configs configs --sample --pattern "cross_*.yaml"
```

Sample results are synthetic demo outputs only.

## 5. Copy Datasets from Drive

Example:

```bash
mkdir -p data/raw/cicids2017 data/raw/unsw_nb15
cp /content/drive/MyDrive/nids-data/cicids2017/*.csv data/raw/cicids2017/
cp /content/drive/MyDrive/nids-data/unsw_nb15/*.csv data/raw/unsw_nb15/
```

## 6. Run Limited Real-Data Experiments

```bash
python src/run_all.py --configs configs --max-rows 50000 --pattern "within_*.yaml"
python src/run_all.py --configs configs --max-rows 50000 --pattern "cross_*.yaml"
```

## 7. Copy Results Back to Drive

```bash
cp -r results /content/drive/MyDrive/nids-results-$(date +%Y%m%d-%H%M%S)
```
