# scoring for infrared thermography oral temp predictions
# usage: python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
# outputs a single RMSE float (lower = better)

import argparse
import sys
import math
import pandas as pd
import numpy as np

REQUIRED_COLS = {"id", "oral_temp"}


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

    # ids need to match up
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
        sub["oral_temp"] = sub["oral_temp"].astype(float)
    except (ValueError, TypeError):
        sys.exit("ERROR: oral_temp column contains non-numeric values")

    if sub["oral_temp"].isna().any():
        sys.exit("ERROR: oral_temp column contains NaN values")

    # sanity check: body temp should be somewhere in [30, 45]
    bad = sub[(sub["oral_temp"] < 30) | (sub["oral_temp"] > 45)]
    if len(bad) > 0:
        sys.exit(
            f"ERROR: {len(bad)} predictions outside plausible "
            f"body temperature range [30, 45] C"
        )

    return sub, sol


def score(sub_path, sol_path):
    sub, sol = load_and_validate(sub_path, sol_path)

    y_true = sol["oral_temp"].astype(float).values
    y_pred = sub["oral_temp"].astype(float).values

    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    return round(rmse, 6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-path", required=True)
    ap.add_argument("--solution-path", required=True)
    args = ap.parse_args()

    print(score(args.submission_path, args.solution_path))
