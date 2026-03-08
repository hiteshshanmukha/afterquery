"""
score_submission.py
-------------------
Scores a submission for the NHTSA CRSS Crash Injury Severity Prediction task.

Usage:
    python3 score_submission.py --submission-path submission.csv --solution-path solution.csv

Metric: RATWGT-Weighted Macro F1
  Each crash carries a national sampling weight (RATWGT) representing how many
  real-world crashes it represents. The metric computes a weighted F1 per class,
  then averages across the 5 severity classes equally (macro).

  This ensures that rare but consequential crashes (fatal/serious) — which tend
  to have higher RATWGT values in CRSS sampling — are not diluted by the large
  number of low-severity crashes.

Output: a single float in [0.0, 1.0], higher is better.
"""

import argparse
import sys
import numpy as np
import pandas as pd

REQUIRED_SUB_COLS = {"id", "INJ_SEV"}
REQUIRED_SOL_COLS = {"id", "INJ_SEV", "RATWGT"}
VALID_CLASSES     = {0, 1, 2, 3, 4}


def load_and_validate(submission_path: str, solution_path: str):
    # ── Load ───────────────────────────────────────────────────────────────
    try:
        sub = pd.read_csv(submission_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: Submission file not found: {submission_path}")
    except Exception as e:
        sys.exit(f"ERROR reading submission: {e}")

    try:
        sol = pd.read_csv(solution_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: Solution file not found: {solution_path}")

    # ── Required columns ──────────────────────────────────────────────────
    missing = REQUIRED_SUB_COLS - set(sub.columns)
    if missing:
        sys.exit(f"ERROR: Submission missing columns: {missing}. "
                 f"Required: id, INJ_SEV")

    # ── Row count ─────────────────────────────────────────────────────────
    if len(sub) != len(sol):
        sys.exit(f"ERROR: Row count mismatch. Submission={len(sub)}, "
                 f"Solution={len(sol)}.")

    # ── ID match ──────────────────────────────────────────────────────────
    sub_ids = set(sub["id"].tolist())
    sol_ids = set(sol["id"].tolist())
    if sub_ids != sol_ids:
        extra   = list(sub_ids - sol_ids)[:5]
        missing_ids = list(sol_ids - sub_ids)[:5]
        msg = "ERROR: ID mismatch."
        if extra:        msg += f" Unexpected IDs in submission: {extra}"
        if missing_ids:  msg += f" Missing IDs: {missing_ids}"
        sys.exit(msg)

    # ── Valid class values ────────────────────────────────────────────────
    try:
        pred_classes = set(sub["INJ_SEV"].astype(int).tolist())
    except (ValueError, TypeError):
        sys.exit("ERROR: INJ_SEV column must contain integer values.")

    invalid = pred_classes - VALID_CLASSES
    if invalid:
        sys.exit(f"ERROR: Invalid INJ_SEV values: {invalid}. "
                 f"Allowed: 0, 1, 2, 3, 4")

    if sub["INJ_SEV"].isna().any():
        sys.exit(f"ERROR: INJ_SEV contains {sub['INJ_SEV'].isna().sum()} NaN(s).")

    # ── Align by id ───────────────────────────────────────────────────────
    sub = sub.sort_values("id").reset_index(drop=True)
    sol = sol.sort_values("id").reset_index(drop=True)

    return sub, sol


def weighted_macro_f1(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      sample_weights: np.ndarray) -> float:
    """
    Compute macro-averaged F1 where each sample contributes proportionally
    to its RATWGT sampling weight.

    For each class c:
        TP_c = sum of weights where y_true==c AND y_pred==c
        FP_c = sum of weights where y_true!=c AND y_pred==c
        FN_c = sum of weights where y_true==c AND y_pred!=c
        Precision_c = TP_c / (TP_c + FP_c)
        Recall_c    = TP_c / (TP_c + FN_c)
        F1_c        = 2 * P_c * R_c / (P_c + R_c)

    Macro F1 = mean(F1_c) over all 5 classes.
    If a class has no true or predicted instances, F1_c = 0.
    """
    w = sample_weights / sample_weights.sum()   # normalise weights
    classes = sorted(VALID_CLASSES)
    f1_per_class = []

    for c in classes:
        mask_true = (y_true == c)
        mask_pred = (y_pred == c)

        tp = w[mask_true & mask_pred].sum()
        fp = w[~mask_true & mask_pred].sum()
        fn = w[mask_true & ~mask_pred].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        f1_per_class.append(f1)

    return float(np.mean(f1_per_class))


def score(submission_path: str, solution_path: str) -> float:
    sub, sol = load_and_validate(submission_path, solution_path)

    y_true   = sol["INJ_SEV"].astype(int).values
    y_pred   = sub["INJ_SEV"].astype(int).values
    weights  = sol["RATWGT"].astype(float).values

    result = weighted_macro_f1(y_true, y_pred, weights)
    return round(result, 6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score CRSS crash severity submission."
    )
    parser.add_argument("--submission-path", required=True)
    parser.add_argument("--solution-path",   required=True)
    args = parser.parse_args()

    print(score(args.submission_path, args.solution_path))
