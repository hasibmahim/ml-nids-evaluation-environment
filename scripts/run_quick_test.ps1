$ErrorActionPreference = "Stop"

python -m pytest
python src/run_all.py --configs configs --sample --pattern "within_*.yaml"
