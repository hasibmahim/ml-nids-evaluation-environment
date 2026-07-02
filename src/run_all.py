"""Command-line runner for configured experiments."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import load_configs
from src.experiments import run_experiment
from src.utils import save_dataframe, set_random_seed, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML-NIDS evaluation experiments.")
    parser.add_argument("--configs", default="configs", help="Directory containing YAML configs.")
    parser.add_argument("--sample", action="store_true", help="Use generated synthetic demo data.")
    parser.add_argument("--max-rows", type=int, default=None, help="Override config row limit.")
    parser.add_argument("--pattern", default="*.yaml", help='Config filename pattern, e.g. "within_*.yaml".')
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--random-state", type=int, default=None, help="Override config random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed = args.random_state if args.random_state is not None else 42
    set_random_seed(seed)

    print("ML-NIDS evaluation runner")
    print(f"Configs: {args.configs}")
    print(f"Pattern: {args.pattern}")
    print(f"Sample mode: {args.sample}")
    if args.sample:
        print("WARNING: sample mode uses synthetic demo data only; outputs are not thesis results.")

    try:
        configs = load_configs(args.configs, args.pattern)
    except Exception as exc:
        print(f"ERROR: Could not load configs: {exc}")
        return 1

    summaries: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    for config in configs:
        name = config["experiment_name"]
        print(f"\nRunning {name} ({config['experiment_type']})")
        try:
            metrics_df = run_experiment(
                config,
                sample=args.sample,
                max_rows=args.max_rows,
                output_dir=args.output_dir,
                random_state=args.random_state,
            )
            summaries.append(metrics_df)
            print(f"Completed {name}: {len(metrics_df)} metric rows")
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"ERROR in {name}: {exc}")
            traceback.print_exc()

    if summaries:
        output_dir = Path(args.output_dir or "results")
        suffix = "sample" if args.sample else "real"
        summary_path = output_dir / "tables" / f"run_summary_{suffix}_{timestamp().replace(':', '')}.csv"
        save_dataframe(pd.concat(summaries, ignore_index=True), summary_path)
        print(f"\nSummary saved to {summary_path}")

    if failures:
        print("\nSome experiments failed:")
        for name, message in failures:
            print(f"- {name}: {message}")
        return 1

    print("\nAll experiments completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
