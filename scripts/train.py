"""
CLI entrypoint for a full training run.

Usage:
    python scripts/train.py
    python scripts/train.py --experiment MyExperiment
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Run ForecastOps training pipeline.")
    parser.add_argument(
        "--experiment",
        default="ForecastOps",
        help="MLflow experiment name (default: ForecastOps)",
    )
    args = parser.parse_args()

    from app.training.pipeline import run_pipeline

    result = run_pipeline(experiment_name=args.experiment)
    winner = result["winner"]
    print("\n" + "=" * 50)
    print(f"  Winner : {winner['model_name']}")
    print(f"  RMSE   : {winner['rmse']:.4f}")
    print(f"  MAE    : {winner['mae']:.4f}")
    print(f"  MAPE   : {winner['mape']:.2f}%")
    print(f"  Version: {winner.get('model_version', 'N/A')} → Production")
    print("=" * 50)


if __name__ == "__main__":
    main()
