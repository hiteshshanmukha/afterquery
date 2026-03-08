"""
Scoring script for credit card default prediction.

Usage:
    python3 score_submission.py --submission-path submission.csv --solution-path solution.csv

Metric: AUC-ROC (higher = better, 0.5 = random, 1.0 = perfect)
Submission needs columns: id, default_prob (float in [0, 1])
"""

import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


REQUIRED_SUB_COLS  = {"id", "default_prob"}
REQUIRED_SOL_COLS  = {"id", "default_payment_next_month"}


def load_and_validate(submission_path: str, solution_path: str):
    # Load files
    try:
        sub = pd.read_csv(submission_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: Submission file not found: {submission_path}")
    except Exception as e:
        sys.exit(f"ERROR: Could not read submission file: {e}")

    try:
        sol = pd.read_csv(solution_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: Solution file not found: {solution_path}")

    # Check required columns
    missing = REQUIRED_SUB_COLS - set(sub.columns)
    if missing:
        sys.exit(f"ERROR: Submission missing required columns: {missing}. "
                 f"Expected columns: id, default_prob")

    # Row count check
    if len(sub) != len(sol):
        sys.exit(
            f"ERROR: Row count mismatch. Submission={len(sub)} rows, "
            f"Solution={len(sol)} rows."
        )

    # ID match
    sub_ids = set(sub["id"].tolist())
    sol_ids = set(sol["id"].tolist())
    if sub_ids != sol_ids:
        extra   = list(sub_ids - sol_ids)[:5]
        missing_ids = list(sol_ids - sub_ids)[:5]
        msg = "ERROR: ID mismatch between submission and solution."
        if extra:        msg += f" Unexpected IDs: {extra}"
        if missing_ids:  msg += f" Missing IDs: {missing_ids}"
        sys.exit(msg)

    # Validate probabilities
    try:
        probs = sub["default_prob"].astype(float)
    except (ValueError, TypeError):
        sys.exit("ERROR: default_prob column must contain numeric values.")

    if probs.isna().any():
        n_na = probs.isna().sum()
        sys.exit(f"ERROR: default_prob contains {n_na} NaN value(s).")

    out_of_range = ((probs < 0.0) | (probs > 1.0)).sum()
    if out_of_range > 0:
        sys.exit(
            f"ERROR: {out_of_range} value(s) in default_prob are outside [0, 1]. "
            f"All predicted probabilities must be in [0.0, 1.0]."
        )

    # Align by id
    sub = sub.sort_values("id").reset_index(drop=True)
    sol = sol.sort_values("id").reset_index(drop=True)

    return sub, sol


def score(submission_path: str, solution_path: str) -> float:
    sub, sol = load_and_validate(submission_path, solution_path)

    y_true = sol["default_payment_next_month"].astype(int).values
    y_score = sub["default_prob"].astype(float).values

    # All predictions identical -> AUC undefined, return 0.5
    if len(np.unique(y_score)) == 1:
        return 0.5

    auc = roc_auc_score(y_true, y_score)
    return round(float(auc), 6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score credit card default prediction submission."
    )
    parser.add_argument("--submission-path", required=True, help="Path to submission.csv")
    parser.add_argument("--solution-path",   required=True, help="Path to solution.csv")
    args = parser.parse_args()

    result = score(args.submission_path, args.solution_path)
    print(result)
