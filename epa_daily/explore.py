import pandas as pd
import zipfile
import os

def get_csv_from_zip(zpath):
    """Open the actual csv inside the zip, skipping directory entries."""
    z = zipfile.ZipFile(zpath)
    csvs = [n for n in z.namelist() if n.endswith('.csv')]
    return z, z.open(csvs[0])

# peek at PM2.5 structure
z, f = get_csv_from_zip('raw/daily_88101_2022.zip')
df = pd.read_csv(f, nrows=5)
print('PM25 columns:', df.columns.tolist())
print()
print(df.head(2).to_string())
f.close(); z.close()

print()

# load full pm25 2022
z, f = get_csv_from_zip('raw/daily_88101_2022.zip')
full = pd.read_csv(f)
print('PM25 2022 shape:', full.shape)
pnames = full["Parameter Name"].unique()
print('Parameter Name unique:', pnames)
am = full["Arithmetic Mean"]
print('Arithmetic Mean range:', am.min(), 'to', am.max())
f.close(); z.close()

print()

# check temp
z, f = get_csv_from_zip('raw/daily_TEMP_2022.zip')
tdf = pd.read_csv(f, nrows=100)
print('TEMP parameters:', tdf["Parameter Name"].unique())
f.close(); z.close()

# check wind
z, f = get_csv_from_zip('raw/daily_WIND_2022.zip')
wdf = pd.read_csv(f, nrows=100)
print('WIND parameters:', wdf["Parameter Name"].unique())
f.close(); z.close()

# check rh
z, f = get_csv_from_zip('raw/daily_RH_DP_2022.zip')
rdf = pd.read_csv(f, nrows=100)
print('RH_DP parameters:', rdf["Parameter Name"].unique())
f.close(); z.close()

# check pressure
z, f = get_csv_from_zip('raw/daily_PRESS_2022.zip')
pdf_data = pd.read_csv(f, nrows=100)
print('PRESS parameters:', pdf_data["Parameter Name"].unique())
f.close(); z.close()
