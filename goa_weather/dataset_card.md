# Goa Weather Solar Irradiation — Dataset Card

## Overview

| Property | Value |
|----------|-------|
| **Name** | Goa Weather — Solar Irradiation |
| **Domain** | Solar energy / meteorology |
| **Task** | Regression (predict solar irradiation Wh/m²) |
| **Rows (total)** | 108,096 |
| **Rows (train)** | 87,744 (Jul 2016 – Dec 2018) |
| **Rows (test)** | 20,352 (Jan – Jul 2019) |
| **Columns** | 27 (including id, datetime, target) |

## Source

OpenML dataset 43409 ("Historical-Weather-data-of-Goa-India"): https://www.openml.org/search?type=data&id=43409

Derived from NASA MERRA-2 reanalysis data for Goa, India, downloaded via [SoDa](http://www.soda-pro.com/web-services/meteo-data/merra). The raw data covers 15-minute interval meteorological observations from July 2016 to July 2019.

## License

AQ Internal

## Splitting & Leakage

A temporal split was used: training data covers July 2016 through December 2018, and test data covers January 2019 through July 2019. This is a proper time-series split that avoids data leakage from future weather to past predictions. Notably, the test set includes a wider range of dry-season months (January-May) relative to the same period in training, making generalisation important.

The target column (`solar_irradiation`) is not present in `test.csv`. The `is_daytime` feature was derived from training-period targets, so it does not leak test-time labels. Rolling and lag features (`pressure_change`, `rainfall_1h`, `humidity_1h_avg`) were computed over the full chronological series before splitting, which means the first few rows of the test set carry a small amount of information from the end of the training period — this is an acceptable and realistic edge effect for a time-series forecast task.

## Target

`solar_irradiation` — short-wave solar irradiation measured at the surface in Wh/m² per 15-minute interval.

| Statistic | Train | Test |
|-----------|-------|------|
| Mean | 56.49 | 61.56 |
| Std | 76.49 | 80.99 |
| Min | 0.0 | 0.0 |
| 25% | 0.0 | 0.0 |
| 50% | 0.0 | 0.0 |
| 75% | ~99 | ~110 |
| Max | ~265 | ~265 |

The distribution is highly skewed: roughly half the intervals are nighttime with zero irradiation. Daytime values follow a bell-shaped curve peaking around solar noon, with significant variance due to cloud cover.

## Features

### Raw weather measurements
- **temperature_c**: Air temperature in °C (converted from Kelvin)
- **humidity**: Relative humidity, percent
- **pressure_hpa**: Barometric pressure in hPa
- **wind_speed**: Wind speed in m/s
- **wind_direction**: Meteorological wind direction, 0-360°
- **rainfall**: Precipitation intensity in kg/m²

### Temporal features
- **hour, minute, day_of_week, month, day_of_year, year**: Calendar components
- **hour_sin, hour_cos**: Cyclical encoding of hour
- **month_sin, month_cos**: Cyclical encoding of month

### Engineered features
- **is_daytime**: Binary flag (1 = irradiation was positive in training data)  
- **is_monsoon**: 1 if June-September (the Indian southwest monsoon)
- **wind_u, wind_v**: Cartesian wind vector decomposition
- **humidity_temp_interaction**: humidity × temperature / 100 (cloud formation proxy)
- **pressure_change**: Pressure delta over the last hour
- **rainfall_1h**: Cumulative rainfall over past hour
- **humidity_1h_avg**: Rolling 1-hour mean humidity

## Known Issues

1. The `is_daytime` feature in test was derived from training-period patterns; actual sunrise/sunset varies and may not align exactly at dawn/dusk transition.
2. Some rare timestamps had "24:00:00" formatting (equivalent to midnight next day); these were sanitised during preprocessing.
3. Rainfall values are extremely sparse — mostly zeros with occasional heavy spikes during monsoon.
4. Pressure change and rolling features may have edge effects at the start of each year's data.

## Complexity & Difficulty

- **Temporal split** introduces distribution shift: the test period has more high-solar dry months than proportionally present in training.
- **Zero-inflated target**: ~50% of values are exactly zero (nighttime). Models that don't handle this bimodality will waste capacity.
- **Monsoon regime change**: Very different weather patterns between Jun-Sep (cloudy, rainy, low solar) and Oct-May (clear, sunny, high solar). The test set spans both regimes.
- **High frequency**: 15-minute intervals mean that lag features and temporal modeling matter, but pure autoregressive approaches can't use test-time target values.
- **Feature interactions**: Solar irradiation depends on complex interactions between time of day, cloud cover (proxied by humidity and rainfall), season, and atmospheric stability.
