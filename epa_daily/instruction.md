# EPA Daily PM2.5 Concentration Prediction

## Objective

Predict daily fine particulate matter (PM2.5) concentrations at EPA monitoring stations across the United States. Each row is one day at one monitoring site. You have meteorological conditions and co-pollutant measurements from the same site and day, plus the station's location.

Target column: `pm25` (float, micrograms per cubic meter, ug/m3).

Most values are in the 0-20 range (clean-to-moderate air quality), but there's a heavy right tail. Wildfire smoke events and industrial incidents push occasional readings above 100, sometimes above 300. The median is 7 ug/m3, the mean is 8.5, and the max is 395. That gap between mean and median tells you the distribution is skewed.

The data spans 2020-2022 and covers ~480 monitoring sites across all 50 states. About 93,700 total observations after QC.

---

## Inputs

| File | Rows | What's in it |
|------|------|--------------|
| `train.csv` | 74,956 | 16 features + `pm25` target + `id` |
| `test.csv`  | 18,740 | 16 features + `id`, no target |

### Features

**Location:** `Latitude`, `Longitude` (decimal degrees), `state_code` (FIPS code, integer), `cbsa_code` (metro area, integer -- 0 means rural or uncommon metro)

**Time:** `year` (2020-2022), `month` (1-12), `day_of_week` (0=Monday through 6=Sunday), `day_of_year` (1-366)

**Meteorology (same site, same day):**
- `temp` -- outdoor temperature in Fahrenheit. Missing ~5%.
- `wind_speed` -- resultant wind speed. Missing ~29%.
- `rel_humidity` -- relative humidity percentage. Missing ~32%.
- `pressure` -- barometric pressure in millibars. Missing ~47%.

**Co-pollutants (same site, same day):**
- `ozone` -- ozone concentration in ppm. Missing ~30%.
- `no2` -- nitrogen dioxide in ppb. Missing ~43%.
- `so2` -- sulfur dioxide in ppb. Missing ~51%.
- `co` -- carbon monoxide in ppm. Missing ~49%.

The missingness is real. Not all monitoring sites measure all parameters. A PM2.5 station in a rural area probably doesn't have NO2 or CO sensors co-located. The missingness pattern itself is informative -- sites with full met+pollutant coverage tend to be large urban monitoring stations that see different PM2.5 levels than rural sites.

---

## Output

Generate `submission.csv` with exactly 18,740 rows plus a header:

```
id,pm25
0,7.2
4,12.1
9,5.8
...
```

| Column | Type | Details |
|--------|------|---------|
| `id` | int | Must match the IDs in `test.csv` exactly |
| `pm25` | float | Predicted PM2.5 concentration in ug/m3, must be >= 0 |

---

## Metric

RMSE (Root Mean Squared Error). Lower is better, 0.0 is perfect.

```
RMSE = sqrt( mean( (predicted - actual)^2 ) )
```

Predicting the training mean for everything gives ~6.81. A decent model should get well below 5.

RMSE penalizes large errors quadratically. Since this dataset has a heavy right tail (some days with PM2.5 > 100 from wildfire smoke), getting those extreme events right matters a lot. A model that nails the 5-10 ug/m3 bulk but blows the high-PM2.5 days will score badly.

```bash
python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
```

---

## Constraints

- Predictions must be non-negative (PM2.5 >= 0). Negative values fail validation.
- No NaN or non-numeric predictions.
- `id` must match test.csv exactly.
