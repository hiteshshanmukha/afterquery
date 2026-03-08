# Scoring script for wine quality predictions.
# Usage: python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
# Prints a single float (macro F1).

import argparse
import sys
import pandas as pd
from sklearn.metrics import f1_score

REQUIRED_COLS = {"id", "quality"}
ALLOWED_QUALITY = {3, 4, 5, 6, 7, 8}


def load_and_validate(sub_path, sol_path):
    """Read both CSVs and run basic sanity checks."""
    try:
        sub = pd.read_csv(sub_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: can't find submission file: {sub_path}")

    try:
        sol = pd.read_csv(sol_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: can't find solution file: {sol_path}")

    # check columns
    for name, df in [("Submission", sub), ("Solution", sol)]:
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            sys.exit(f"ERROR: {name} is missing columns: {missing}")

    # row counts need to match
    if len(sub) != len(sol):
        sys.exit(
            f"ERROR: row count mismatch — submission has {len(sub)}, "
            f"solution has {len(sol)}"
        )

    # make sure IDs line up
    sub_ids = set(sub["id"])
    sol_ids = set(sol["id"])
    if sub_ids != sol_ids:
        extra = sub_ids - sol_ids
        missing = sol_ids - sub_ids
        msg = "ERROR: ID mismatch."
        if extra:
            msg += f" Extra in submission: {sorted(list(extra))[:5]}"
        if missing:
            msg += f" Missing from submission: {sorted(list(missing))[:5]}"
        sys.exit(msg)

    # sort both by id so rows align
    sub = sub.sort_values("id").reset_index(drop=True)
    sol = sol.sort_values("id").reset_index(drop=True)

    # quality values must be valid
    bad_vals = set(sub["quality"]) - ALLOWED_QUALITY
    if bad_vals:
        sys.exit(f"ERROR: invalid quality values: {bad_vals} (allowed: {ALLOWED_QUALITY})")

    return sub, sol


def score(sub_path, sol_path):
    sub, sol = load_and_validate(sub_path, sol_path)

    y_true = sol["quality"].astype(int).values
    y_pred = sub["quality"].astype(int).values

    result = f1_score(y_true, y_pred, average="macro", labels=sorted(ALLOWED_QUALITY))
    return round(float(result), 6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-path", required=True)
    ap.add_argument("--solution-path", required=True)
    args = ap.parse_args()

    print(score(args.submission_path, args.solution_path))
