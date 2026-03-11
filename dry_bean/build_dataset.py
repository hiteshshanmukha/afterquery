"""
Build script for the Dry Bean Classification task.
Fetches the dataset from UCI, creates train/test split, and generates all required CSV files.
"""
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

SEED = 42
TEST_RATIO = 0.20

# 1. Fetch dataset
print("Fetching Dry Bean dataset from UCI...")
ds = fetch_ucirepo(id=602)
features = ds.data.features.copy()
targets = ds.data.targets.copy()

# Combine
df = pd.concat([features, targets], axis=1)
print(f"Total rows: {len(df)}, columns: {list(df.columns)}")
print(f"Class distribution:\n{df['Class'].value_counts().sort_index()}")

# 2. Shuffle and assign IDs
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df.insert(0, "id", range(1, len(df) + 1))

# 3. Stratified train/test split
train_df, test_df = train_test_split(
    df, test_size=TEST_RATIO, random_state=SEED, stratify=df["Class"]
)
train_df = train_df.sort_values("id").reset_index(drop=True)
test_df = test_df.sort_values("id").reset_index(drop=True)

print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")
print(f"Train class dist:\n{train_df['Class'].value_counts().sort_index()}")
print(f"Test class dist:\n{test_df['Class'].value_counts().sort_index()}")

# 4. Save train.csv (with target)
train_df.to_csv("train.csv", index=False)
print("Saved train.csv")

# 5. Save test.csv (no target)
test_cols = [c for c in test_df.columns if c != "Class"]
test_df[test_cols].to_csv("test.csv", index=False)
print("Saved test.csv")

# 6. Save solution.csv (id + target for test set)
test_df[["id", "Class"]].to_csv("solution.csv", index=False)
print("Saved solution.csv")

# 7. Save perfect_submission.csv (same as solution.csv)
test_df[["id", "Class"]].to_csv("perfect_submission.csv", index=False)
print("Saved perfect_submission.csv")

# 8. Save sample_submission.csv (all predict most common class)
most_common = train_df["Class"].mode()[0]
sample_sub = test_df[["id"]].copy()
sample_sub["Class"] = most_common
sample_sub.to_csv("sample_submission.csv", index=False)
print(f"Saved sample_submission.csv (all predictions = '{most_common}')")

print("\nDone! All CSV files generated.")
print(f"  train.csv:              {len(train_df)} rows")
print(f"  test.csv:               {len(test_df)} rows")
print(f"  solution.csv:           {len(test_df)} rows")
print(f"  perfect_submission.csv: {len(test_df)} rows")
print(f"  sample_submission.csv:  {len(test_df)} rows")
