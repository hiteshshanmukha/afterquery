# Scoring script for dry bean classification predictions.
# Usage: python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
# Prints a single float (macro F1).

import argparse
import sys
import pandas as pd
from sklearn.metrics import f1_score

REQUIRED_COLS = {"id", "Class"}
ALLOWED_CLASSES = {"BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"}


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
        missing_ids = sol_ids - sub_ids
        msg = "ERROR: ID mismatch."
        if extra:
            msg += f" Extra in submission: {sorted(list(extra))[:5]}"
        if missing_ids:
            msg += f" Missing from submission: {sorted(list(missing_ids))[:5]}"
        sys.exit(msg)

    # sort both by id so rows align
    sub = sub.sort_values("id").reset_index(drop=True)
    sol = sol.sort_values("id").reset_index(drop=True)

    # class values must be valid
    sub["Class"] = sub["Class"].astype(str).str.strip()
    bad_vals = set(sub["Class"]) - ALLOWED_CLASSES
    if bad_vals:
        sys.exit(f"ERROR: invalid Class values: {bad_vals} (allowed: {sorted(ALLOWED_CLASSES)})")

    return sub, sol


def score(sub_path, sol_path):
    sub, sol = load_and_validate(sub_path, sol_path)

    y_true = sol["Class"].astype(str).str.strip().values
    y_pred = sub["Class"].astype(str).str.strip().values

    result = f1_score(y_true, y_pred, average="macro", labels=sorted(ALLOWED_CLASSES))
    return round(float(result), 6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-path", required=True)
    ap.add_argument("--solution-path", required=True)
    args = ap.parse_args()

    print(score(args.submission_path, args.solution_path))
