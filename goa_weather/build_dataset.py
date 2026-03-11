"""
Build train/test/solution/sample/perfect CSVs from the Historical Weather Data of Goa, India.
Time series regression: predict solar irradiation (Wh/m2) from weather conditions and temporal features.

Source: OpenML dataset 43409 (Historical-Weather-data-of-Goa-India)
Upstream: Meteorological data for Goa, India (15-minute intervals, July 2016 - July 2019)

Split: temporal. Train = Jul 2016 to Dec 2018, Test = Jan 2019 to Jul 2019.
"""

import pandas as pd
import numpy as np
import openml

TARGET = "solar_irradiation"
SEED = 42

print("Fetching Goa Weather Data from OpenML...")
ds = openml.datasets.get_dataset(43409)
df = ds.get_data()[0]
print(f"Raw shape: {df.shape}")

# clean column names
rename = {
    "Date": "date",
    "UT_time": "time",
    "Temperature_(K)": "temperature_k",
    "Relative_Humidity_(%)": "humidity",
    "Pressure_(hPa)": "pressure_hpa",
    "Wind_speed_(m/s)": "wind_speed",
    "Wind_direction": "wind_direction",
    "Rainfall_(kg/m2)": "rainfall",
    "Short-wave_irradiation_(Wh/m2)": "solar_irradiation",
}
df = df.rename(columns=rename)

# parse datetime - handle "24:00:00" entries (means midnight of next day)
# time column has mixed formats: "HH:MM" and "HH:MM:SS"
def fix_time_24(row):
    t = str(row["time"]).strip()
    d_str = str(row["date"]).strip()
    if t.startswith("24:"):
        t = "00:" + t[3:]
        d = pd.to_datetime(d_str, format="%d-%m-%Y") + pd.Timedelta(days=1)
        return pd.to_datetime(d.strftime("%Y-%m-%d") + " " + t, format="mixed")
    return pd.to_datetime(d_str + " " + t, format="mixed", dayfirst=True)

df["datetime"] = df.apply(fix_time_24, axis=1)
df = df.sort_values("datetime").reset_index(drop=True)

# convert temperature from Kelvin to Celsius (more intuitive for Indian context)
df["temperature_c"] = round(df["temperature_k"] - 273.15, 2)

# extract temporal features
df["hour"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month
df["day_of_year"] = df["datetime"].dt.dayofyear
df["year"] = df["datetime"].dt.year

# cyclical encoding of hour (important for solar patterns)
df["hour_sin"] = round(np.sin(2 * np.pi * (df["hour"] + df["minute"] / 60) / 24), 4)
df["hour_cos"] = round(np.cos(2 * np.pi * (df["hour"] + df["minute"] / 60) / 24), 4)

# cyclical encoding of month (monsoon seasonality)
df["month_sin"] = round(np.sin(2 * np.pi * df["month"] / 12), 4)
df["month_cos"] = round(np.cos(2 * np.pi * df["month"] / 12), 4)

# is_daytime flag (solar irradiation is zero at night)
df["is_daytime"] = (df["solar_irradiation"] > 0).astype(int)

# monsoon flag (June-September for Goa)
df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)

# wind components (u,v from speed and direction)
df["wind_u"] = round(df["wind_speed"] * np.sin(np.radians(df["wind_direction"])), 3)
df["wind_v"] = round(df["wind_speed"] * np.cos(np.radians(df["wind_direction"])), 3)

# humidity-temperature interaction (cloud formation proxy)
df["humidity_temp_interaction"] = round(df["humidity"] * df["temperature_c"] / 100, 2)

# pressure change (difference from rolling mean as a weather front indicator)
# We'll compute a simple lag-based feature
df["pressure_change"] = round(df["pressure_hpa"].diff(4).fillna(0), 2)  # change over 1 hour

# rolling averages (4 intervals = 1 hour)
df["rainfall_1h"] = round(df["rainfall"].rolling(4, min_periods=1).sum(), 4)
df["humidity_1h_avg"] = round(df["humidity"].rolling(4, min_periods=1).mean(), 2)

# drop raw date/time columns (we have datetime and extracted features)
df = df.drop(columns=["date", "time", "temperature_k"])

# keep datetime as string for CSV
df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

# add sequential id
df.insert(0, "id", range(len(df)))

# round floats
for col in df.select_dtypes(include=[float]).columns:
    if col not in ["latitude", "longitude"]:
        df[col] = df[col].round(4)

print(f"Cleaned shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print()

# temporal split: train = Jul 2016 to Dec 2018, test = Jan 2019 to Jul 2019
train_df = df[df["year"] < 2019].copy()
test_df = df[df["year"] >= 2019].copy()

print(f"Train: {len(train_df)} rows (Jul 2016 - Dec 2018)")
print(f"Test:  {len(test_df)} rows (Jan 2019 - Jul 2019)")
print()
print(f"Train target stats:\n{train_df[TARGET].describe()}")
print()
print(f"Test target stats:\n{test_df[TARGET].describe()}")
print()

# write train.csv
train_df.to_csv("train.csv", index=False)

# write test.csv (drop target)
test_df.drop(columns=[TARGET]).to_csv("test.csv", index=False)

# write solution.csv (id + target)
test_df[["id", TARGET]].to_csv("solution.csv", index=False)

# sample submission: predict training mean
train_mean = round(train_df[TARGET].mean(), 4)
sample = test_df[["id"]].copy()
sample[TARGET] = train_mean
sample.to_csv("sample_submission.csv", index=False)

# perfect submission
test_df[["id", TARGET]].to_csv("perfect_submission.csv", index=False)

print("Files written:")
import os
for fn in ["train.csv", "test.csv", "solution.csv", "sample_submission.csv", "perfect_submission.csv"]:
    size_mb = os.path.getsize(fn) / (1024 * 1024)
    rows = len(pd.read_csv(fn))
    print(f"  {fn}: {size_mb:.2f} MB, {rows} rows")
print("\nDone.")
