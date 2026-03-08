"""Debug why the merge returns 0 matches."""
import pandas as pd
import zipfile

def load_one(zpath):
    with zipfile.ZipFile(zpath) as z:
        csvs = [n for n in z.namelist() if n.endswith('.csv')]
        with z.open(csvs[0]) as f:
            return pd.read_csv(f, low_memory=False)

pm = load_one('raw/daily_88101_2022.zip')
pm = pm[pm['Sample Duration'] == '24 HOUR']

temp = load_one('raw/daily_TEMP_2022.zip')

# check types
print("PM key types:")
for c in ['State Code', 'County Code', 'Site Num']:
    print(f"  {c}: dtype={pm[c].dtype}, sample={pm[c].iloc[0]}")

print("TEMP key types:")
for c in ['State Code', 'County Code', 'Site Num']:
    print(f"  {c}: dtype={temp[c].dtype}, sample={temp[c].iloc[0]}")

# date format
print(f"\nPM Date Local sample: '{pm['Date Local'].iloc[0]}'")
print(f"TEMP Date Local sample: '{temp['Date Local'].iloc[0]}'")

# check if same sites exist
pm_sites = set(zip(pm['State Code'], pm['County Code'], pm['Site Num']))
temp_sites = set(zip(temp['State Code'], temp['County Code'], temp['Site Num']))
print(f"\nPM sites: {len(pm_sites)}")
print(f"TEMP sites: {len(temp_sites)}")
print(f"Overlap: {len(pm_sites & temp_sites)}")

# check TEMP sample durations
print(f"\nTEMP Sample Duration values: {temp['Sample Duration'].unique()}")
