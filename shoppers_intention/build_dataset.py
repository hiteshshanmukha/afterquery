"""
Build train/test/solution/sample/perfect CSVs from the Online Shoppers Purchasing Intention dataset.
Binary classification: predict whether a browsing session ends in a purchase (Revenue = 1 or 0).

UCI dataset id: 468
Source: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
License: CC-BY-4.0

Split: stratified random 80/20 on Revenue.
"""

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

TARGET = "revenue"
SEED = 42

print("Fetching Online Shoppers Purchasing Intention from UCI...")
ds = fetch_ucirepo(id=468)
df = pd.concat([ds.data.features, ds.data.targets], axis=1)
print(f"Raw shape: {df.shape}")
print(f"Raw columns: {list(df.columns)}")

# clean column names
rename = {
    "Administrative": "admin_pages",
    "Administrative_Duration": "admin_duration",
    "Informational": "info_pages",
    "Informational_Duration": "info_duration",
    "ProductRelated": "product_pages",
    "ProductRelated_Duration": "product_duration",
    "BounceRates": "bounce_rate",
    "ExitRates": "exit_rate",
    "PageValues": "page_values",
    "SpecialDay": "special_day",
    "Month": "month",
    "OperatingSystems": "operating_system",
    "Browser": "browser",
    "Region": "region",
    "TrafficType": "traffic_type",
    "VisitorType": "visitor_type",
    "Weekend": "weekend",
    "Revenue": "revenue",
}
df = df.rename(columns=rename)

# convert boolean target to int
df["revenue"] = df["revenue"].astype(int)
df["weekend"] = df["weekend"].astype(int)

# add derived features
# total pages visited across all categories
df["total_pages"] = df["admin_pages"] + df["info_pages"] + df["product_pages"]
# total time spent
df["total_duration"] = df["admin_duration"] + df["info_duration"] + df["product_duration"]
# product focus ratio (how much of the browsing was product-related)
df["product_focus"] = round(
    df["product_pages"] / (df["total_pages"] + 1), 4
)
# average time per page
df["avg_time_per_page"] = round(
    df["total_duration"] / (df["total_pages"] + 1), 4
)
# bounce-exit gap (how much bounce rate differs from exit rate)
df["bounce_exit_gap"] = round(df["exit_rate"] - df["bounce_rate"], 6)

# drop rows with NaN (there are a few)
before = len(df)
df = df.dropna().reset_index(drop=True)
after = len(df)
if before != after:
    print(f"Dropped {before - after} rows with NaN values")

# shuffle and add id
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df.insert(0, "id", range(len(df)))

print(f"Cleaned shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nTarget distribution:\n{df[TARGET].value_counts()}")
print(f"Revenue rate: {df[TARGET].mean():.3f}")
print()

# stratified 80/20 split
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df[TARGET], random_state=SEED
)
train_df = train_df.sort_values("id").reset_index(drop=True)
test_df = test_df.sort_values("id").reset_index(drop=True)

print(f"Train: {len(train_df)} rows (revenue rate: {train_df[TARGET].mean():.3f})")
print(f"Test:  {len(test_df)} rows (revenue rate: {test_df[TARGET].mean():.3f})")
print()
print(f"Train target distribution:\n{train_df[TARGET].value_counts()}")
print()

# write train.csv (all columns including target)
train_df.to_csv("train.csv", index=False)

# write test.csv (drop target)
test_df.drop(columns=[TARGET]).to_csv("test.csv", index=False)

# write solution.csv (id + target)
test_df[["id", TARGET]].to_csv("solution.csv", index=False)

# sample submission: predict 0 for everything (majority class)
sample = test_df[["id"]].copy()
sample[TARGET] = 0
sample.to_csv("sample_submission.csv", index=False)

# perfect submission: matches solution exactly
test_df[["id", TARGET]].to_csv("perfect_submission.csv", index=False)

print("Files written:")
import os
for fn in ["train.csv", "test.csv", "solution.csv", "sample_submission.csv", "perfect_submission.csv"]:
    size_kb = os.path.getsize(fn) / 1024
    rows = len(pd.read_csv(fn))
    print(f"  {fn}: {size_kb:.1f} KB, {rows} rows")
print("\nDone.")
