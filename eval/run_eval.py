"""Evaluate both candidates using the same training and promotion pipeline."""

import json

import pandas as pd

from app.training.pipeline import run_pipeline


def main():
    result = run_pipeline(experiment_name="ForecastOps-Evaluation")
    table = pd.DataFrame(result["all_results"])[["model_name", "mae", "rmse", "mape"]]
    print(table.sort_values("rmse").to_string(index=False))
    print(
        json.dumps(
            {
                "winner": result["winner"]["model_name"],
                "protocol": "chronological 4-week holdout; recursive LightGBM",
            }
        )
    )


if __name__ == "__main__":
    main()
