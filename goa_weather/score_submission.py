"""Score a submission for the Goa Solar Irradiation Forecasting task.

Usage
-----
    python3 score_submission.py --submission-path submission.csv --solution-path solution.csv

Output
------
    A single floating-point number: the RMSE of the submission.
"""

import argparse
import sys

import numpy as np
import pandas as pd


def score(submission_path: str, solution_path: str) -> float:
    # --- load ----------------------------------------------------------------
    sub = pd.read_csv(submission_path)
    sol = pd.read_csv(solution_path)

    # --- column validation ---------------------------------------------------
    if "id" not in sub.columns or "solar_irradiation" not in sub.columns:
        raise ValueError(
            "Submission must have columns 'id' and 'solar_irradiation'."
        )
    if "id" not in sol.columns or "solar_irradiation" not in sol.columns:
        raise ValueError(
            "Solution must have columns 'id' and 'solar_irradiation'."
        )

    # --- row-count validation ------------------------------------------------
    if len(sub) != len(sol):
        raise ValueError(
            f"Row count mismatch: submission has {len(sub)} rows, "
            f"solution has {len(sol)} rows."
        )

    # --- merge on id ---------------------------------------------------------
    merged = sol.merge(sub, on="id", suffixes=("_true", "_pred"))
    if len(merged) != len(sol):
        raise ValueError(
            f"ID mismatch: only {len(merged)} of {len(sol)} IDs matched."
        )

    # --- value validation ----------------------------------------------------
    if merged["solar_irradiation_pred"].isna().any():
        raise ValueError("Submission contains NaN predictions.")

    # --- RMSE ----------------------------------------------------------------
    y_true = merged["solar_irradiation_true"].values
    y_pred = merged["solar_irradiation_pred"].values
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return rmse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a Goa Solar Irradiation submission."
    )
    parser.add_argument(
        "--submission-path", required=True, help="Path to the submission CSV."
    )
    parser.add_argument(
        "--solution-path", required=True, help="Path to the solution CSV."
    )
    args = parser.parse_args()

    try:
        result = score(args.submission_path, args.solution_path)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
