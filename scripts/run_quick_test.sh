#!/usr/bin/env bash
set -euo pipefail

python -m pytest
python src/run_all.py --configs configs --sample --pattern "within_*.yaml"
