"""
Build train/test/solution/sample/perfect CSVs from the New Delhi Rental Listings dataset.
Regression: predict monthly rental price (INR) from property and location features.

Source: OpenML dataset 43837 (New-Delhi-Rental-Listings)
Upstream: Scraped New Delhi property rental data

Split: random 80/20 stratified on price quartile.
"""

import pandas as pd
import numpy as np
import openml

TARGET = "monthly_rent"
SEED = 42

print("Fetching New Delhi Rental Listings from OpenML...")
ds = openml.datasets.get_dataset(43837)
df = ds.get_data()[0]
print(f"Raw shape: {df.shape}")

# drop the unnamed index column
if "Unnamed:_0" in df.columns:
    df = df.drop(columns=["Unnamed:_0"])

# clean column names
rename = {
    "size_sq_ft": "size_sqft",
    "propertyType": "property_type",
    "bedrooms": "bedrooms",
    "latitude": "latitude",
    "longitude": "longitude",
    "localityName": "locality",
    "suburbName": "suburb",
    "cityName": "city",
    "price": "monthly_rent",
    "companyName": "listing_agency",
    "closest_mtero_station_km": "metro_dist_km",
    "AP_dist_km": "airport_dist_km",
    "Aiims_dist_km": "aiims_dist_km",
    "NDRLW_dist_km": "railway_station_dist_km",
}
df = df.rename(columns=rename)

# drop city column (all rows are Delhi)
if df["city"].nunique() <= 2:
    df = df.drop(columns=["city"])

# drop listing_agency  (potentially leaky - certain agents operate in premium areas)
df = df.drop(columns=["listing_agency"])

# clean outliers: remove extreme prices (above 99.5th percentile or below 1st percentile)
q_low = df[TARGET].quantile(0.005)
q_high = df[TARGET].quantile(0.995)
before = len(df)
df = df[(df[TARGET] >= q_low) & (df[TARGET] <= q_high)].copy()
print(f"Removed {before - len(df)} outlier rows (price outside {q_low:.0f}-{q_high:.0f})")

# remove implausible sizes
df = df[(df["size_sqft"] >= 100) & (df["size_sqft"] <= 10000)].copy()

# remove rows with implausible coordinates (clearly wrong)
df = df[(df["latitude"] > 28.0) & (df["latitude"] < 29.0)].copy()
df = df[(df["longitude"] > 76.5) & (df["longitude"] < 77.6)].copy()

# remove implausible distances (likely data errors: > 100km from landmarks)
for col in ["metro_dist_km", "airport_dist_km", "aiims_dist_km", "railway_station_dist_km"]:
    df = df[df[col] < 100].copy()

print(f"After cleaning: {len(df)} rows")

# feature engineering
# price per sqft (remove from test, this is derived from target)
df["rent_per_sqft"] = round(df[TARGET] / df["size_sqft"], 2)

# property density proxy: count localities in the same suburb
suburb_counts = df.groupby("suburb")["locality"].transform("nunique")
df["suburb_locality_count"] = suburb_counts

# distance to city center (approx Connaught Place: 28.6315, 77.2167)
df["center_dist_km"] = round(np.sqrt(
    ((df["latitude"] - 28.6315) * 111) ** 2 +
    ((df["longitude"] - 77.2167) * 85) ** 2
), 3)

# log transform of size for better modeling
df["log_size"] = round(np.log(df["size_sqft"]), 4)

# bedrooms per 100 sqft
df["bedrooms_per_100sqft"] = round(df["bedrooms"] / (df["size_sqft"] / 100), 4)

# average distance to all landmarks
df["avg_landmark_dist"] = round(
    (df["metro_dist_km"] + df["airport_dist_km"] + df["aiims_dist_km"] + df["railway_station_dist_km"]) / 4,
    3
)

# suburb encoding: compute median rent per suburb from training data later
# for now, encode property_type
prop_type_map = df["property_type"].value_counts().to_dict()
df["property_type_freq"] = df["property_type"].map(prop_type_map)

# Group rare localities into "Other"
locality_counts = df["locality"].value_counts()
rare_localities = locality_counts[locality_counts < 10].index
df["locality_grouped"] = df["locality"].apply(lambda x: "Other" if x in rare_localities else x)

# drop the rent_per_sqft (derived from target)
df = df.drop(columns=["rent_per_sqft"])

# shuffle and add id
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df.insert(0, "id", range(len(df)))

# Round floats for cleaner CSVs
for col in df.select_dtypes(include=[float]).columns:
    df[col] = df[col].round(4)

print(f"Final shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print()

# Stratified split on price quartile
df["_price_q"] = pd.qcut(df[TARGET], q=4, labels=False)
np.random.seed(SEED)
mask = np.zeros(len(df), dtype=bool)
for q in range(4):
    idx = df[df["_price_q"] == q].index
    selected = np.random.choice(idx, size=int(len(idx) * 0.8), replace=False)
    mask[selected] = True

train_df = df[mask].copy().drop(columns=["_price_q"])
test_df = df[~mask].copy().drop(columns=["_price_q"])

print(f"Train: {len(train_df)} rows (mean rent: {train_df[TARGET].mean():.0f})")
print(f"Test:  {len(test_df)} rows (mean rent: {test_df[TARGET].mean():.0f})")
print()
print(f"Train target stats:\n{train_df[TARGET].describe()}")
print()

# write train.csv (all columns including target)
train_df.to_csv("train.csv", index=False)

# write test.csv (drop target)
test_df.drop(columns=[TARGET]).to_csv("test.csv", index=False)

# write solution.csv (id + target)
test_df[["id", TARGET]].to_csv("solution.csv", index=False)

# sample submission: predict the training median
train_median = int(train_df[TARGET].median())
sample = test_df[["id"]].copy()
sample[TARGET] = train_median
sample.to_csv("sample_submission.csv", index=False)

# perfect submission
test_df[["id", TARGET]].to_csv("perfect_submission.csv", index=False)

print("Files written:")
import os
for fn in ["train.csv", "test.csv", "solution.csv", "sample_submission.csv", "perfect_submission.csv"]:
    size_kb = os.path.getsize(fn) / 1024
    rows = len(pd.read_csv(fn))
    print(f"  {fn}: {size_kb:.1f} KB, {rows} rows")
print("\nDone.")
