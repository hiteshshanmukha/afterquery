"""
Build script for Infrared Thermography Oral Temperature Prediction task.
Fetches the dataset from UCI, creates train/test split, and generates all required CSV files.
"""
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

SEED = 42
TEST_RATIO = 0.20

# 1. Fetch dataset
print("Fetching Infrared Thermography Temperature dataset from UCI (id=925)...")
ds = fetch_ucirepo(id=925)
features = ds.data.features.copy()
targets = ds.data.targets.copy()

# Combine features and a single target
# We predict aveOralM (average oral temperature, mixed measurement)
df = pd.concat([features, targets[["aveOralM"]]], axis=1)
df = df.rename(columns={"aveOralM": "oral_temp"})

# Drop rows with any NaN
n_before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"Dropped {n_before - len(df)} rows with NaN. Remaining: {len(df)}")

# Rename columns for clarity (remove spaces, make consistent)
col_renames = {
    "T_atm": "ambient_temp",
    "T_offset1": "temp_offset",
    "Max1R13_1": "max_right_inner_canthus",
    "Max1L13_1": "max_left_inner_canthus",
    "aveAllR13_1": "avg_right_inner_canthus",
    "aveAllL13_1": "avg_left_inner_canthus",
    "T_RC1": "right_cheek_temp",
    "T_RC_Dry1": "right_cheek_dry",
    "T_RC_Wet1": "right_cheek_wet",
    "T_RC_Max1": "right_cheek_max",
    "T_LC1": "left_cheek_temp",
    "T_LC_Dry1": "left_cheek_dry",
    "T_LC_Wet1": "left_cheek_wet",
    "T_LC_Max1": "left_cheek_max",
    "RCC1": "right_canthus_corrected",
    "LCC1": "left_canthus_corrected",
    "canthiMax1": "canthi_max",
    "canthi4Max1": "canthi_4_max",
    "T_FHCC1": "forehead_center",
    "T_FHRC1": "forehead_right",
    "T_FHLC1": "forehead_left",
    "T_FHBC1": "forehead_bottom",
    "T_FHTC1": "forehead_top",
    "T_FH_Max1": "forehead_max",
    "T_FHC_Max1": "forehead_center_max",
    "T_Max1": "face_max_temp",
    "T_OR1": "orbital_right",
    "T_OR_Max1": "orbital_right_max",
}
df = df.rename(columns=col_renames)

print(f"Total rows: {len(df)}, columns: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")
print(f"\nTarget (oral_temp) stats:")
print(df["oral_temp"].describe())

# 2. Shuffle and assign IDs
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df.insert(0, "id", range(1, len(df) + 1))

# 3. Stratified split via binning target for stratification
df["_temp_bin"] = pd.qcut(df["oral_temp"], q=5, labels=False, duplicates="drop")
train_df, test_df = train_test_split(
    df, test_size=TEST_RATIO, random_state=SEED, stratify=df["_temp_bin"]
)
train_df = train_df.drop(columns=["_temp_bin"]).sort_values("id").reset_index(drop=True)
test_df = test_df.drop(columns=["_temp_bin"]).sort_values("id").reset_index(drop=True)

print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")
print(f"Train oral_temp: mean={train_df['oral_temp'].mean():.3f}, std={train_df['oral_temp'].std():.3f}")
print(f"Test oral_temp:  mean={test_df['oral_temp'].mean():.3f}, std={test_df['oral_temp'].std():.3f}")

# 4. Save train.csv (with target)
train_df.to_csv("train.csv", index=False)
print("\nSaved train.csv")

# 5. Save test.csv (no target)
test_cols = [c for c in test_df.columns if c != "oral_temp"]
test_df[test_cols].to_csv("test.csv", index=False)
print("Saved test.csv")

# 6. Save solution.csv (id + target for test set)
test_df[["id", "oral_temp"]].to_csv("solution.csv", index=False)
print("Saved solution.csv")

# 7. Save perfect_submission.csv (same as solution)
test_df[["id", "oral_temp"]].to_csv("perfect_submission.csv", index=False)
print("Saved perfect_submission.csv")

# 8. Save sample_submission.csv (predict mean of training set)
train_mean = round(train_df["oral_temp"].mean(), 2)
sample_sub = test_df[["id"]].copy()
sample_sub["oral_temp"] = train_mean
sample_sub.to_csv("sample_submission.csv", index=False)
print(f"Saved sample_submission.csv (all predictions = {train_mean})")

print(f"\nDone! All CSV files generated.")
print(f"  train.csv:              {len(train_df)} rows")
print(f"  test.csv:               {len(test_df)} rows")
print(f"  solution.csv:           {len(test_df)} rows")
print(f"  perfect_submission.csv: {len(test_df)} rows")
print(f"  sample_submission.csv:  {len(test_df)} rows")
