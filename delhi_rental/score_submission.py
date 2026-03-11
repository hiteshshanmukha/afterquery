# scoring for New Delhi rental price predictions
# usage: python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
# outputs a single RMSE float (lower = better, 0 is perfect)

import argparse
import sys
import pandas as pd
import numpy as np

REQUIRED_COLS = {"id", "monthly_rent"}


def load_and_validate(sub_path, sol_path):
    """Load submission + solution CSVs, bail out if anything looks off."""
    try:
        sub = pd.read_csv(sub_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: can't find submission file: {sub_path}")

    try:
        sol = pd.read_csv(sol_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: can't find solution file: {sol_path}")

    # columns present?
    for name, df in [("Submission", sub), ("Solution", sol)]:
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            sys.exit(f"ERROR: {name} is missing columns: {missing}")

    # same number of rows?
    if len(sub) != len(sol):
        sys.exit(
            f"ERROR: row count mismatch - submission has {len(sub)}, "
            f"solution has {len(sol)}"
        )

    # ids need to match
    sub_ids = set(sub["id"])
    sol_ids = set(sol["id"])
    if sub_ids != sol_ids:
        extra = sub_ids - sol_ids
        missing_ids = sol_ids - sub_ids
        msg = "ERROR: ID mismatch."
        if extra:
            msg += f" Extra in submission: {sorted(list(extra))[:5]}"
        if missing_ids:
            msg += f" Missing from submission: {sorted(list(missing_ids))[:5]}"
        sys.exit(msg)

    # align by id
    sub = sub.sort_values("id").reset_index(drop=True)
    sol = sol.sort_values("id").reset_index(drop=True)

    # coerce to float
    try:
        sub["monthly_rent"] = sub["monthly_rent"].astype(float)
    except (ValueError, TypeError):
        sys.exit("ERROR: monthly_rent column contains non-numeric values")

    if sub["monthly_rent"].isna().any():
        sys.exit("ERROR: monthly_rent column contains NaN values")

    # sanity check: rent should be positive
    if (sub["monthly_rent"] <= 0).any():
        sys.exit("ERROR: monthly_rent contains non-positive values")

    if (sub["monthly_rent"] > 500000).any():
        sys.exit(
            "ERROR: monthly_rent contains implausibly large values (>500000 INR)"
        )

    return sub, sol


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def score(sub_path, sol_path):
    sub, sol = load_and_validate(sub_path, sol_path)

    y_true = sol["monthly_rent"].astype(float).values
    y_pred = sub["monthly_rent"].astype(float).values

    return round(rmse(y_true, y_pred), 6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-path", required=True)
    ap.add_argument("--solution-path", required=True)
    args = ap.parse_args()

    print(score(args.submission_path, args.solution_path))
