"""
score_submission.py

Scores a submission for the EPA PM2.5 daily concentration prediction task.

Run:
    python3 score_submission.py --submission-path submission.csv --solution-path solution.csv

Metric: RMSE (root mean squared error) on the raw PM2.5 values in ug/m3.
Lower is better. Perfect score is 0.0.
"""

import argparse
import sys
import numpy as np
import pandas as pd

REQUIRED_COLS = {"id", "pm25"}


def load_and_validate(sub_path, sol_path):
    try:
        sub = pd.read_csv(sub_path)
    except FileNotFoundError:
        sys.exit("Submission file not found: " + sub_path)
    except Exception as e:
        sys.exit("Can't read submission: " + str(e))

    try:
        sol = pd.read_csv(sol_path)
    except FileNotFoundError:
        sys.exit("Solution file not found: " + sol_path)

    # columns
    missing = REQUIRED_COLS - set(sub.columns)
    if missing:
        sys.exit("Submission missing columns: " + str(missing))

    # row count
    if len(sub) != len(sol):
        sys.exit("Row count mismatch: submission has %d, expected %d" % (len(sub), len(sol)))

    # id match
    sub_ids = set(sub["id"].tolist())
    sol_ids = set(sol["id"].tolist())
    if sub_ids != sol_ids:
        extra = list(sub_ids - sol_ids)[:5]
        gone = list(sol_ids - sub_ids)[:5]
        msg = "ID mismatch."
        if extra:
            msg += " Extra in submission: " + str(extra)
        if gone:
            msg += " Missing from submission: " + str(gone)
        sys.exit(msg)

    # must be numeric
    try:
        sub["pm25"] = sub["pm25"].astype(float)
    except (ValueError, TypeError):
        sys.exit("pm25 column must be numeric")

    if sub["pm25"].isna().any():
        sys.exit("pm25 has %d NaN values" % sub["pm25"].isna().sum())

    # no negatives
    if (sub["pm25"] < 0).any():
        n_neg = (sub["pm25"] < 0).sum()
        sys.exit("pm25 has %d negative values" % n_neg)

    # sort by id
    sub = sub.sort_values("id").reset_index(drop=True)
    sol = sol.sort_values("id").reset_index(drop=True)

    return sub, sol


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def score(sub_path, sol_path):
    sub, sol = load_and_validate(sub_path, sol_path)
    y_true = sol["pm25"].values
    y_pred = sub["pm25"].values
    return round(rmse(y_true, y_pred), 6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score PM2.5 predictions")
    parser.add_argument("--submission-path", required=True)
    parser.add_argument("--solution-path", required=True)
    args = parser.parse_args()

    print(score(args.submission_path, args.solution_path))
