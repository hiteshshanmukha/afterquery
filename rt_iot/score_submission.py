# scoring for RT-IoT2022 network traffic classification
# usage: python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
# outputs a single macro-F1 float (higher = better)

import argparse
import sys
import pandas as pd
from sklearn.metrics import f1_score

REQUIRED_COLS = {"id", "Attack_type"}

VALID_CLASSES = {
    "DOS_SYN_Hping",
    "Thing_Speak",
    "ARP_poisioning",
    "MQTT_Publish",
    "NMAP_UDP_SCAN",
    "NMAP_XMAS_TREE_SCAN",
    "NMAP_OS_DETECTION",
    "NMAP_TCP_scan",
    "DDOS_Slowloris",
    "Wipro_bulb",
    "Metasploit_Brute_Force_SSH",
    "NMAP_FIN_SCAN",
}


def load_and_validate(sub_path, sol_path):
    """Load submission + solution CSVs and run basic sanity checks."""
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

    # strip whitespace from predictions
    sub["Attack_type"] = sub["Attack_type"].astype(str).str.strip()

    # check for invalid class labels
    bad_labels = set(sub["Attack_type"]) - VALID_CLASSES
    if bad_labels:
        sys.exit(f"ERROR: unrecognized Attack_type values: {bad_labels}")

    if sub["Attack_type"].isna().any():
        sys.exit("ERROR: Attack_type column contains NaN values")

    return sub, sol


def score(sub_path, sol_path):
    sub, sol = load_and_validate(sub_path, sol_path)

    y_true = sol["Attack_type"].values
    y_pred = sub["Attack_type"].values

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return round(macro_f1, 6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-path", required=True)
    ap.add_argument("--solution-path", required=True)
    args = ap.parse_args()

    print(score(args.submission_path, args.solution_path))
