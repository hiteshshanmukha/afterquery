# scoring for online shoppers purchase intention predictions
# usage: python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
# outputs a single AUC-ROC float (higher = better, 1.0 is perfect)

import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

REQUIRED_COLS = {"id", "revenue"}
ALLOWED_VALUES = {0, 1}


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

    # coerce to int
    try:
        sub["revenue"] = sub["revenue"].astype(int)
    except (ValueError, TypeError):
        sys.exit("ERROR: revenue column contains non-integer values")

    if sub["revenue"].isna().any():
        sys.exit("ERROR: revenue column contains NaN values")

    # values must be 0 or 1
    bad_vals = set(sub["revenue"].unique()) - ALLOWED_VALUES
    if bad_vals:
        sys.exit(f"ERROR: invalid revenue values: {bad_vals} (allowed: {ALLOWED_VALUES})")

    return sub, sol


def score(sub_path, sol_path):
    sub, sol = load_and_validate(sub_path, sol_path)

    y_true = sol["revenue"].astype(int).values
    y_pred = sub["revenue"].astype(int).values

    # AUC-ROC needs at least one positive and one negative in y_true
    if len(set(y_true)) < 2:
        sys.exit("ERROR: solution has only one class — can't compute AUC-ROC")

    return round(float(roc_auc_score(y_true, y_pred)), 6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-path", required=True)
    ap.add_argument("--solution-path", required=True)
    args = ap.parse_args()

    print(score(args.submission_path, args.solution_path))
