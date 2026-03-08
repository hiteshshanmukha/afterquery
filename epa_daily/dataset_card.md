# EPA Daily PM2.5 Dataset

## Overview

Daily average PM2.5 (fine particulate matter, diameter <= 2.5 micrometers) concentrations measured at EPA Air Quality System (AQS) monitoring stations across the US, from 2020 through 2022. PM2.5 is the pollutant most strongly linked to respiratory and cardiovascular health effects. Predicting it from meteorological conditions and co-pollutant readings is a real operational problem -- filling temporal gaps in the monitoring network, nowcasting for health advisories, etc.

I joined the PM2.5 daily data with same-site, same-day observations of temperature, wind speed, relative humidity, barometric pressure, ozone, nitrogen dioxide, sulfur dioxide, and carbon monoxide. The joins are on the EPA site identifier (state FIPS + county code + site number) and date, so all features come from the same physical location and day as the PM2.5 reading.

After filtering to 24-hour PM2.5 averages only (no 1-hour snapshots), removing wildfire event flags, and dropping quality-suspect readings, there are about 93,700 observations across ~480 sites.

## Source

EPA Air Quality System (AQS), pre-generated daily summary files.

Downloads from: https://aqs.epa.gov/aqsweb/airdata/download_files.html

Specific files used:
- PM2.5 FRM/FEM (parameter 88101): `daily_88101_{2020,2021,2022}.zip`
- Outdoor Temperature: `daily_TEMP_{2020,2021,2022}.zip`
- Wind Speed: `daily_WIND_{2020,2021,2022}.zip`
- Relative Humidity / Dew Point: `daily_RH_DP_{2020,2021,2022}.zip`
- Barometric Pressure: `daily_PRESS_{2020,2021,2022}.zip`
- Ozone (44201): `daily_44201_{2020,2021,2022}.zip`
- NO2 (42602): `daily_42602_{2020,2021,2022}.zip`
- SO2 (42401): `daily_42401_{2020,2021,2022}.zip`
- CO (42101): `daily_42101_{2020,2021,2022}.zip`

EPA AQS documentation: https://aqs.epa.gov/aqsweb/documents/about_aqs_data.html

## License

Public-Domain-US-Gov

EPA AQS data is produced by the U.S. Environmental Protection Agency, a federal agency. The data is public domain under US government works.

https://aqs.epa.gov/aqsweb/airdata/download_files.html

## Features

| Column | Type | Notes |
|--------|------|-------|
| `id` | int | Row identifier |
| `Latitude` | float | Decimal degrees, station location |
| `Longitude` | float | Decimal degrees, station location |
| `state_code` | int | FIPS state code (1-56) |
| `cbsa_code` | int | Core-based statistical area (metro area). 0 = rural or rare metro. Integer-encoded. |
| `year` | int | 2020, 2021, or 2022 |
| `month` | int | 1-12 |
| `day_of_week` | int | 0=Monday through 6=Sunday |
| `day_of_year` | int | 1-366 |
| `temp` | float | Outdoor temperature, Fahrenheit. Missing ~5%. |
| `wind_speed` | float | Resultant wind speed, knots. Missing ~29%. |
| `rel_humidity` | float | Relative humidity, percent. Missing ~32%. |
| `pressure` | float | Barometric pressure, millibars. Missing ~47%. |
| `ozone` | float | Daily mean ozone, ppm. Missing ~30%. |
| `no2` | float | Nitrogen dioxide, ppb. Missing ~43%. |
| `so2` | float | Sulfur dioxide, ppb. Missing ~51%. |
| `co` | float | Carbon monoxide, ppm. Missing ~49%. |
| `pm25` | float | **Target.** Daily average PM2.5 concentration, ug/m3. Only in train.csv. |

The missingness is a core challenge. Not every EPA station measures all parameters. PM2.5-only sites in rural areas often lack co-pollutant instruments entirely. The missingness is MNAR (missing not at random) -- it's driven by which instruments a station has, which correlates with site type (urban/rural), funding, and the kinds of pollution sources nearby. Sites with co-located NO2 and CO monitors tend to be near roadways, which has different PM2.5 behavior than a background rural monitor.

## Splitting & Leakage

Stratified random split, 80% train / 20% test. Stratification is on binned PM2.5 concentration (6 bins: 0-5, 5-10, 10-15, 15-25, 25-50, 50+).

No temporal split -- dates from all three years appear in both train and test. This means single-site autocorrelation can help: if site X on January 5th is in train, nearby dates from the same site might be in test, and PM2.5 doesn't change much day-to-day at the same location. This isn't leakage per se (all features are from the same day), but it does make the task easier than predicting at truly novel sites or times.

`cbsa_code` comes from the EPA data. It's a legitimate geographic feature (metro-area-level grouping). Not leakage, but aware that encoding it as an integer creates an implicit ordinal structure that doesn't exist in reality -- these are just metro ID numbers.
