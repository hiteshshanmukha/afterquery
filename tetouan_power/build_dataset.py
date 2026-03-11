"""
Build train/test/solution/sample/perfect CSVs from the Tetouan City Power Consumption dataset.
Time series regression: predict Zone 1 power consumption from weather + other zones.

UCI dataset id: 849
Source: https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city
License: CC-BY-4.0

Split: time-based. Train = Jan-Oct 2017, Test = Nov-Dec 2017.
"""

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

TARGET = "zone_1_power"

print("Fetching Tetouan Power Consumption from UCI...")
ds = fetch_ucirepo(id=849)
df = ds.data.original.copy()
print(f"Raw shape: {df.shape}")

# clean column names
rename = {
    "DateTime": "datetime",
    "Temperature": "temperature",
    "Humidity": "humidity",
    "Wind Speed": "wind_speed",
    "general diffuse flows": "general_diffuse_flows",
    "diffuse flows": "diffuse_flows",
    "Zone 1 Power Consumption": "zone_1_power",
    "Zone 2  Power Consumption": "zone_2_power",
    "Zone 3  Power Consumption": "zone_3_power",
}
df = df.rename(columns=rename)

# parse datetime
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# extract time features that a model might want
df["hour"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month
df["day_of_year"] = df["datetime"].dt.dayofyear

# add id
df.insert(0, "id", range(len(df)))

# keep datetime as string for the CSV
df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

print(f"Cleaned shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print()

# time-based split: Jan-Oct = train, Nov-Dec = test
# month column is already extracted
train_df = df[df["month"] <= 10].copy()
test_df = df[df["month"] > 10].copy()

print(f"Train: {len(train_df)} rows (Jan-Oct 2017)")
print(f"Test:  {len(test_df)} rows (Nov-Dec 2017)")
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
    size_mb = os.path.getsize(fn) / (1024 * 1024)
    rows = len(pd.read_csv(fn))
    print(f"  {fn}: {size_mb:.2f} MB, {rows} rows")
print("\nDone.")
