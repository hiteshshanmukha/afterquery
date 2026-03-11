# Goa Solar Irradiation Forecasting

## Objective

Predict the short-wave solar irradiation (in Wh/m²) at 15-minute intervals for Goa, India, using concurrent weather measurements and temporal features. This is a regression task over a time series dataset with strong daily and seasonal patterns.

Goa sits at ~15.5°N on India's western coast. Solar irradiation follows a clear daily cycle (zero at night, bell-shaped during the day) modulated by cloud cover, humidity, and the monsoon. The southwest monsoon (June-September) brings heavy cloud cover and rainfall, dramatically reducing solar output. The post-monsoon and pre-monsoon dry months have strong, consistent solar irradiation.

The data covers July 2016 to July 2019. Training runs through December 2018, and you need to predict January through July 2019. This temporal split means the test period includes both the dry season (Jan-May, high solar) and the onset of the 2019 monsoon (Jun-Jul, dropping solar).

## Inputs

| File | Contents |
|------|----------|
| `train.csv` | 87,744 records (Jul 2016 – Dec 2018) with all features + `solar_irradiation` |
| `test.csv`  | 20,352 records (Jan – Jul 2019), same features but no `solar_irradiation` |

### Features

**Weather measurements:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Row ID (sequential across the full dataset) |
| `datetime` | string | Timestamp, format "YYYY-MM-DD HH:MM:SS", every 15 minutes |
| `temperature_c` | float | Air temperature in Celsius. Goa ranges ~22-36°C. |
| `humidity` | float | Relative humidity (%). High in monsoon (>85%), lower in dry season. |
| `pressure_hpa` | float | Atmospheric pressure in hectopascals. |
| `wind_speed` | float | Wind speed (m/s) |
| `wind_direction` | float | Wind direction in degrees (0-360, meteorological convention) |
| `rainfall` | float | Rainfall intensity (kg/m²). Extremely spiky — 0 most of the time, heavy bursts during monsoon. |

**Temporal features:**

| Column | Type | Description |
|--------|------|-------------|
| `hour` | int | Hour of day (0-23) |
| `minute` | int | Minute (0, 15, 30, or 45) |
| `day_of_week` | int | 0=Monday through 6=Sunday |
| `month` | int | 1-12 |
| `day_of_year` | int | 1-365/366 |
| `year` | int | 2016-2019 |
| `hour_sin` | float | sin(2π × fractional_hour / 24) — cyclical hour encoding |
| `hour_cos` | float | cos(2π × fractional_hour / 24) |
| `month_sin` | float | sin(2π × month / 12) — cyclical month encoding |
| `month_cos` | float | cos(2π × month / 12) |

**Derived features:**

| Column | Type | Description |
|--------|------|-------------|
| `is_daytime` | int | 1 if solar_irradiation > 0 in training data (useful as a learned proxy, not available for test) |
| `is_monsoon` | int | 1 if month is June-September |
| `wind_u` | float | East-west wind component (speed × sin(direction)) |
| `wind_v` | float | North-south wind component (speed × cos(direction)) |
| `humidity_temp_interaction` | float | humidity × temperature / 100 — cloud formation proxy |
| `pressure_change` | float | Pressure change over the past hour — weather front indicator |
| `rainfall_1h` | float | Cumulative rainfall over past hour (sum of 4 intervals) |
| `humidity_1h_avg` | float | Average humidity over past hour |

## Output

Produce a `submission.csv` with 20,352 rows (plus header).

```
id,solar_irradiation
87744,0.0
87745,0.0
87746,12.34
...
```

| Column | Type | Details |
|--------|------|--------|
| `id` | int | Must match IDs in `test.csv` |
| `solar_irradiation` | float | Predicted solar irradiation in Wh/m² (must be ≥ 0) |

## Metric

RMSE (Root Mean Squared Error). Lower is better.

```
RMSE = sqrt(mean((y_true - y_pred)²))
```

Predicting the training mean for every row gives RMSE ~76. A model that just predicts zero at night and a bell curve during the day should get below 50. A good model that conditions on weather and season should push below 30. Anything under 20 is excellent.

Score locally:

```bash
python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
```

## Constraints

- Don't use `id` as a feature — it's just a row identifier.
- All other columns are fair game.
- Predictions must be non-negative (solar irradiation can't be negative).
- `is_daytime` in the test set was computed from training data and may not perfectly reflect actual dawn/dusk — don't blindly trust it for masking.
