"""
Build train/test/solution/sample/perfect CSVs from the Concrete Compressive Strength dataset.
Regression: predict concrete compressive strength (MPa) from mix proportions and age.

UCI dataset id: 165
Source: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
License: CC-BY-4.0

Split: random 80/20 with fixed seed.
"""

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

TARGET = "compressive_strength"
SEED = 42

print("Fetching Concrete Compressive Strength from UCI...")
ds = fetch_ucirepo(id=165)
df = pd.concat([ds.data.features, ds.data.targets], axis=1)
print(f"Raw shape: {df.shape}")

# clean column names
rename = {
    "Cement": "cement",
    "Blast Furnace Slag": "blast_furnace_slag",
    "Fly Ash": "fly_ash",
    "Water": "water",
    "Superplasticizer": "superplasticizer",
    "Coarse Aggregate": "coarse_aggregate",
    "Fine Aggregate": "fine_aggregate",
    "Age": "age",
    "Concrete compressive strength": "compressive_strength",
}
df = df.rename(columns=rename)

# add derived features that make the problem more interesting
# water-cement ratio is the most important factor in concrete engineering
df["water_cement_ratio"] = round(df["water"] / (df["cement"] + 0.01), 4)
# total binder = cement + slag + fly ash
df["total_binder"] = round(df["cement"] + df["blast_furnace_slag"] + df["fly_ash"], 2)
# aggregate ratio
df["coarse_fine_ratio"] = round(df["coarse_aggregate"] / (df["fine_aggregate"] + 0.01), 4)
# log of age (strength develops logarithmically with age)
df["log_age"] = round(np.log1p(df["age"]), 4)

# shuffle and add id
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df.insert(0, "id", range(len(df)))

print(f"Cleaned shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print()

# random 80/20 split
np.random.seed(SEED)
mask = np.random.rand(len(df)) < 0.8
train_df = df[mask].copy()
test_df = df[~mask].copy()

print(f"Train: {len(train_df)} rows")
print(f"Test:  {len(test_df)} rows")
print()
print(f"Train target stats:\n{train_df[TARGET].describe()}")
print()
print(f"Test target stats:\n{test_df[TARGET].describe()}")
print()

# write train.csv (all columns including target)
train_df.to_csv("train.csv", index=False)

# write test.csv (drop target)
test_df.drop(columns=[TARGET]).to_csv("test.csv", index=False)

# write solution.csv (id + target)
test_df[["id", TARGET]].to_csv("solution.csv", index=False)

# sample submission: predict the training mean for everything
train_mean = round(train_df[TARGET].mean(), 2)
sample = test_df[["id"]].copy()
sample[TARGET] = train_mean
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
