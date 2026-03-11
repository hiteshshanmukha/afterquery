# Tetouan City Power Consumption Forecasting

## Objective

Predict the power consumption of Zone 1 in Tetouan, a city in northern Morocco, at 10-minute intervals. You're given weather conditions (temperature, humidity, wind, solar radiation proxies), the concurrent power consumption of two other distribution zones, and temporal information.

The data covers all of 2017. Training data runs from January through October, and you need to predict November and December. This is a proper temporal split, so you can't use future data to predict past values.

## Inputs

| File | Contents |
|------|----------|
| `train.csv` | 43,776 records (Jan-Oct 2017) with all features + `zone_1_power` |
| `test.csv`  | 8,640 records (Nov-Dec 2017), same features but no `zone_1_power` |

### Features

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Row ID (sequential across the full year) |
| `datetime` | string | Timestamp, format "YYYY-MM-DD HH:MM:SS", every 10 minutes |
| `temperature` | float | Air temperature in Tetouan (Celsius) |
| `humidity` | float | Relative humidity (%) |
| `wind_speed` | float | Wind speed |
| `general_diffuse_flows` | float | General diffuse solar radiation. Proxy for cloud cover. |
| `diffuse_flows` | float | Diffuse solar radiation component |
| `zone_2_power` | float | Power consumption of Zone 2 (different distribution network) |
| `zone_3_power` | float | Power consumption of Zone 3 (different distribution network) |
| `hour` | int | Hour of day (0-23) |
| `minute` | int | Minute of hour (0 or 10 or 20 ... 50) |
| `day_of_week` | int | 0=Monday through 6=Sunday |
| `month` | int | Month number (1-12) |
| `day_of_year` | int | Day of year (1-365) |

## Output

Produce a `submission.csv` with 8,640 rows (plus header).

```
id,zone_1_power
43776,28500.12
43777,29012.55
43778,27800.00
...
```

| Column | Type | Details |
|--------|------|--------|
| `id` | int | Must match IDs in `test.csv` |
| `zone_1_power` | float | Predicted power consumption for Zone 1 (must be positive) |

## Metric

SMAPE (Symmetric Mean Absolute Percentage Error), on a 0-100 scale. Lower is better. 0 is perfect.

```
SMAPE = mean( |y_true - y_pred| / ((|y_true| + |y_pred|) / 2) ) * 100
```

SMAPE penalizes over- and under-prediction somewhat symmetrically, and it's scale-independent. Predicting the training mean for every row gives a SMAPE around 18-19. A model that picks up the daily cycle and seasonal trend should get below 10, and a good one should push under 5.

Score locally with:

```bash
python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
```

## Constraints

- `id` is a row identifier, not a feature (but the ordering is meaningful since IDs are sequential in time).
- This is a time series task. The train/test split is temporal: train = Jan-Oct, test = Nov-Dec. Don't shuffle the data and pretend it's i.i.d.
- `zone_2_power` and `zone_3_power` in the test set are the actual values from the same timestamp. They're available at prediction time because the three zones are metered independently and simultaneously. Using them is fair game.
- All predictions must be positive (power consumption can't be negative).
- `datetime` can be parsed for additional features (holidays, Ramadan timing, etc).
