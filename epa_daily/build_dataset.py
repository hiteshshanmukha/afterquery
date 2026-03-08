"""
build_dataset.py
Build the EPA PM2.5 prediction dataset from raw AQS daily files.

Strategy:
- PM2.5 daily values (parameter 88101) are the target
- For each PM2.5 observation, join same-site same-day data for:
  - Temperature, Wind speed, Relative humidity, Barometric pressure
  - Co-pollutants: Ozone (44201), NO2 (42602), SO2 (42401), CO (42101)
- Also include site metadata: lat, lon, state, CBSA, elevation proxy
- Add temporal features: month, day of week, day of year

Join key: (State Code, County Code, Site Num, Date Local)
"""

import pandas as pd
import numpy as np
import zipfile
import os
from sklearn.model_selection import train_test_split


def load_from_zips(pattern, years, raw_dir='raw'):
    """Load and concat daily CSV data from EPA zip files."""
    frames = []
    for year in years:
        fname = pattern.format(year=year)
        zpath = os.path.join(raw_dir, fname)
        if not os.path.exists(zpath):
            print(f'  SKIP: {zpath} not found')
            continue
        with zipfile.ZipFile(zpath) as z:
            csvs = [n for n in z.namelist() if n.endswith('.csv')]
            with z.open(csvs[0]) as f:
                df = pd.read_csv(f)
                frames.append(df)
                print(f'  {fname}: {len(df)} rows')
    return pd.concat(frames, ignore_index=True)


def normalize_date(df):
    """Make sure Date Local is in YYYY-MM-DD format for consistent joins."""
    df['Date Local'] = pd.to_datetime(df['Date Local']).dt.strftime('%Y-%m-%d')
    return df


def extract_daily_value(df, value_col='Arithmetic Mean'):
    """
    Reduce each parameter file to one row per site-date.
    If data is hourly (multiple rows per site-date), average it.
    Pick POC=1 when multiple POCs exist (primary monitor).
    """
    df = normalize_date(df)
    key_cols = ['State Code', 'County Code', 'Site Num', 'Date Local']

    # first filter to POC=1 where available
    if 'POC' in df.columns:
        min_poc = df.groupby(key_cols)['POC'].transform('min')
        df = df[df['POC'] == min_poc].copy()

    # aggregate to daily mean (handles hourly data)
    lat_lon = df.groupby(key_cols)[['Latitude', 'Longitude']].first()
    daily = df.groupby(key_cols)[value_col].mean().reset_index()
    daily = daily.merge(lat_lon, on=key_cols, how='left')

    return daily


def main():
    years = [2020, 2021, 2022]

    # ---- Load PM2.5 (target) ----
    print('Loading PM2.5...')
    pm25 = load_from_zips('daily_88101_{year}.zip', years)

    # filter to 24-hour observations only, drop events like wildfires
    pm25 = pm25[pm25['Sample Duration'] == '24 HOUR'].copy()
    pm25 = pm25[pm25['Event Type'].isna() | (pm25['Event Type'] == 'None')].copy()

    # basic QC: drop negative and extreme values
    pm25 = pm25[(pm25['Arithmetic Mean'] >= 0) & (pm25['Arithmetic Mean'] <= 500)]

    key_cols = ['State Code', 'County Code', 'Site Num', 'Date Local']

    # normalize dates to YYYY-MM-DD for consistent merging
    pm25 = normalize_date(pm25)

    # reduce to one value per site-date
    pm25 = pm25.sort_values(key_cols + ['POC'])
    pm25 = pm25.drop_duplicates(subset=key_cols, keep='first')

    # keep useful columns
    pm25_base = pm25[key_cols + [
        'Arithmetic Mean', 'Latitude', 'Longitude',
        'AQI', 'State Name', 'County Name', 'CBSA Name'
    ]].copy()
    pm25_base = pm25_base.rename(columns={'Arithmetic Mean': 'pm25'})

    print(f'PM2.5 after QC: {len(pm25_base)} rows')

    # ---- Load meteorological data ----
    met_configs = [
        ('daily_TEMP_{year}.zip', 'temp'),
        ('daily_WIND_{year}.zip', 'wind_speed'),
        ('daily_RH_DP_{year}.zip', 'rel_humidity'),
        ('daily_PRESS_{year}.zip', 'pressure'),
    ]

    for pattern, col_name in met_configs:
        print(f'Loading {col_name}...')
        raw = load_from_zips(pattern, years)
        reduced = extract_daily_value(raw)
        reduced = reduced.rename(columns={'Arithmetic Mean': col_name})
        reduced = reduced.drop(columns=['Latitude', 'Longitude'], errors='ignore')

        pm25_base = pm25_base.merge(
            reduced[key_cols + [col_name]],
            on=key_cols, how='left'
        )
        matched = pm25_base[col_name].notna().sum()
        print(f'  Matched: {matched}/{len(pm25_base)} ({100*matched/len(pm25_base):.1f}%)')

    # ---- Load co-pollutants ----
    copoll_configs = [
        ('daily_44201_{year}.zip', 'ozone'),
        ('daily_42602_{year}.zip', 'no2'),
        ('daily_42401_{year}.zip', 'so2'),
        ('daily_42101_{year}.zip', 'co'),
    ]

    for pattern, col_name in copoll_configs:
        print(f'Loading {col_name}...')
        raw = load_from_zips(pattern, years)
        reduced = extract_daily_value(raw)
        reduced = reduced.rename(columns={'Arithmetic Mean': col_name})
        reduced = reduced.drop(columns=['Latitude', 'Longitude'], errors='ignore')

        pm25_base = pm25_base.merge(
            reduced[key_cols + [col_name]],
            on=key_cols, how='left'
        )
        matched = pm25_base[col_name].notna().sum()
        print(f'  Matched: {matched}/{len(pm25_base)} ({100*matched/len(pm25_base):.1f}%)')

    # ---- Temporal features ----
    pm25_base['date'] = pd.to_datetime(pm25_base['Date Local'])
    pm25_base['month'] = pm25_base['date'].dt.month
    pm25_base['day_of_week'] = pm25_base['date'].dt.dayofweek
    pm25_base['day_of_year'] = pm25_base['date'].dt.dayofyear
    pm25_base['year'] = pm25_base['date'].dt.year

    # ---- Encode state as integer ----
    # use FIPS code which is already there
    pm25_base = pm25_base.rename(columns={'State Code': 'state_code'})

    # ---- Build site_id for grouping ----
    pm25_base['site_id'] = (
        pm25_base['state_code'].astype(str).str.zfill(2) + '_' +
        pm25_base['County Code'].astype(str).str.zfill(3) + '_' +
        pm25_base['Site Num'].astype(str).str.zfill(4)
    )

    # ---- CBSA encoding (metro area) ----
    # encode as integer: frequent CBSAs get their own code, rare ones get 0
    cbsa_counts = pm25_base['CBSA Name'].value_counts()
    top_cbsas = cbsa_counts[cbsa_counts >= 500].index.tolist()
    cbsa_map = {name: i+1 for i, name in enumerate(top_cbsas)}
    pm25_base['cbsa_code'] = pm25_base['CBSA Name'].map(cbsa_map).fillna(0).astype(int)

    # ---- Select final columns ----
    feature_cols = [
        'Latitude', 'Longitude', 'state_code', 'cbsa_code',
        'month', 'day_of_week', 'day_of_year', 'year',
        'temp', 'wind_speed', 'rel_humidity', 'pressure',
        'ozone', 'no2', 'so2', 'co',
    ]
    target_col = 'pm25'

    df = pm25_base[['site_id'] + feature_cols + [target_col]].copy()

    # drop rows where ALL met features are missing (useless)
    met_cols = ['temp', 'wind_speed', 'rel_humidity', 'pressure']
    all_met_missing = df[met_cols].isna().all(axis=1)
    print(f'Dropping {all_met_missing.sum()} rows with no met data at all')
    df = df[~all_met_missing].copy()

    print(f'Final dataset: {len(df)} rows, {len(feature_cols)} features')
    print(f'Missing rates:')
    for c in feature_cols:
        miss = df[c].isna().mean()
        if miss > 0:
            print(f'  {c}: {100*miss:.1f}%')

    # ---- Target stats ----
    print(f'\nTarget (pm25):')
    print(f'  min={df[target_col].min():.2f}, max={df[target_col].max():.1f}')
    print(f'  mean={df[target_col].mean():.2f}, median={df[target_col].median():.2f}')
    print(f'  std={df[target_col].std():.2f}')

    # ---- Train/test split ----
    # stratified on binned pm25 to keep distribution balanced
    bins = [0, 5, 10, 15, 25, 50, 500]
    df['pm25_bin'] = pd.cut(df[target_col], bins=bins, labels=False, include_lowest=True)

    # add id
    df = df.reset_index(drop=True)
    df.insert(0, 'id', range(len(df)))

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['pm25_bin']
    )
    train_df = train_df.sort_values('id').reset_index(drop=True)
    test_df = test_df.sort_values('id').reset_index(drop=True)

    # drop helper columns
    drop_helper = ['site_id', 'pm25_bin']

    train_out = train_df.drop(columns=drop_helper)
    test_out = test_df.drop(columns=drop_helper + [target_col])
    solution = test_df[['id', target_col]].copy()
    perfect = test_df[['id', target_col]].copy()
    sample = test_df[['id']].copy()
    sample[target_col] = train_df[target_col].mean()

    # save
    train_out.to_csv('train.csv', index=False)
    test_out.to_csv('test.csv', index=False)
    solution.to_csv('solution.csv', index=False)
    perfect.to_csv('perfect_submission.csv', index=False)
    sample.to_csv('sample_submission.csv', index=False)

    print(f'\nSaved:')
    print(f'  train.csv: {len(train_out)} rows x {train_out.shape[1]} cols')
    print(f'  test.csv:  {len(test_out)} rows x {test_out.shape[1]} cols')
    print(f'  solution.csv: {len(solution)} rows')
    print(f'  perfect_submission.csv: {len(perfect)} rows')
    print(f'  sample_submission.csv: {len(sample)} rows')


if __name__ == '__main__':
    main()
