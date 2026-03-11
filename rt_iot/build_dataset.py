"""
Build train/test/solution/sample/perfect CSVs from the RT-IoT2022 UCI dataset.
Multi-class classification: predict Attack_type from network flow features.

UCI dataset id: 942
Source: https://archive.ics.uci.edu/dataset/942/rt-iot2022
License: CC-BY-4.0
"""

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

SEED = 42
TEST_FRAC = 0.20

# cap per-class samples to keep the CSV files under ~10 MB total
# rare classes keep all their data, large classes get downsampled
CLASS_CAPS = {
    "DOS_SYN_Hping": 8000,
    "Thing_Speak": 4000,
    "ARP_poisioning": 4000,
    "MQTT_Publish": 3000,
    "NMAP_UDP_SCAN": 2590,    # keep all
    "NMAP_XMAS_TREE_SCAN": 2010,  # keep all
    "NMAP_OS_DETECTION": 2000,    # keep all
    "NMAP_TCP_scan": 1002,        # keep all
    "DDOS_Slowloris": 534,        # keep all
    "Wipro_bulb": 253,            # keep all
    "Metasploit_Brute_Force_SSH": 37,  # keep all
    "NMAP_FIN_SCAN": 28,              # keep all
}

print("Fetching RT-IoT2022 from UCI...")
ds = fetch_ucirepo(id=942)
df = ds.data.original.copy()
print(f"Raw shape: {df.shape}")

# drop the UCI id column, we'll make our own
if "id" in df.columns:
    df = df.drop(columns=["id"])

# rename columns: replace dots with underscores for cleaner names
rename_map = {}
for c in df.columns:
    new = c.replace(".", "_")
    # also rename the port columns to be clearer
    if new == "id_orig_p":
        new = "src_port"
    elif new == "id_resp_p":
        new = "dst_port"
    rename_map[c] = new
df = df.rename(columns=rename_map)

print(f"Cleaned shape: {df.shape}")
print(f"Full target distribution:\n{df['Attack_type'].value_counts()}")
print()

# downsample large classes
dfs = []
for cls, cap in CLASS_CAPS.items():
    subset = df[df["Attack_type"] == cls]
    if len(subset) > cap:
        subset = subset.sample(n=cap, random_state=SEED)
    dfs.append(subset)
df = pd.concat(dfs, ignore_index=True)
print(f"After downsampling: {df.shape}")
print(f"Target distribution:\n{df['Attack_type'].value_counts()}")
print()

# add sequential id
df = df.reset_index(drop=True)
df.insert(0, "id", range(len(df)))

# stratified split on Attack_type
train_df, test_df = train_test_split(
    df, test_size=TEST_FRAC, random_state=SEED, stratify=df["Attack_type"]
)
train_df = train_df.sort_values("id").reset_index(drop=True)
test_df = test_df.sort_values("id").reset_index(drop=True)

print(f"Train: {len(train_df)} rows")
print(f"Test:  {len(test_df)} rows")
print(f"Train target dist:\n{train_df['Attack_type'].value_counts()}")
print(f"Test target dist:\n{test_df['Attack_type'].value_counts()}")
print()

# write train.csv (all columns including Attack_type)
train_df.to_csv("train.csv", index=False)

# write test.csv (drop target)
test_df.drop(columns=["Attack_type"]).to_csv("test.csv", index=False)

# write solution.csv (id + Attack_type)
test_df[["id", "Attack_type"]].to_csv("solution.csv", index=False)

# sample submission: predict the most common class for everything
most_common = train_df["Attack_type"].value_counts().index[0]
sample = test_df[["id"]].copy()
sample["Attack_type"] = most_common
sample.to_csv("sample_submission.csv", index=False)

# perfect submission: matches solution exactly
test_df[["id", "Attack_type"]].to_csv("perfect_submission.csv", index=False)

print("Files written:")
import os
for fn in ["train.csv", "test.csv", "solution.csv", "sample_submission.csv", "perfect_submission.csv"]:
    size_mb = os.path.getsize(fn) / (1024 * 1024)
    print(f"  {fn}: {size_mb:.2f} MB")
print("\nDone.")
