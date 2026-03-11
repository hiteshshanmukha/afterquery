# Infrared Thermography Oral Temperature Prediction

## Objective

Given infrared thermal camera readings taken from different parts of a person's face (inner canthus, cheeks, forehead, orbital area), plus some environmental info and basic demographics, predict their oral temperature in degrees Celsius.

The target column is `oral_temp`. It's a continuous variable. Most values sit in the 36.5-37.5 range, but you'll see a handful of febrile subjects (above 38) and a few on the low end (below 36). Standard deviation is around 0.51, so to be useful a model needs to get within about half a degree.

## Inputs

| File | Contents |
|------|----------|
| `train.csv` | 814 subjects with all features + the `oral_temp` label |
| `test.csv`  | 204 subjects, same features but no `oral_temp` |

### Features

33 features total: 3 categorical and 30 numeric.

**Demographics:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Row ID |
| `Gender` | string | "Male" or "Female" |
| `Age` | string | Age bracket, e.g. "21-30", "31-40", "41-50", "51-60", "61+" |
| `Ethnicity` | string | Self-reported, e.g. "White", "Black or African-American", "Asian", "Hispanic or Latino" |

**Environment:**

| Column | Type | Description |
|--------|------|-------------|
| `ambient_temp` | float | Room temperature in C (roughly 21-26) |
| `Humidity` | float | Relative humidity (%) |
| `Distance` | float | Camera-to-subject distance in meters |
| `temp_offset` | float | Calibration offset for the thermal camera |

**Thermal readings (all in C) -- inner canthus (tear duct area):**

| Column | Type | Description |
|--------|------|-------------|
| `max_right_inner_canthus` | float | Max temp, right inner canthus |
| `max_left_inner_canthus` | float | Max temp, left inner canthus |
| `avg_right_inner_canthus` | float | Average temp, right inner canthus region |
| `avg_left_inner_canthus` | float | Average temp, left inner canthus region |
| `right_canthus_corrected` | float | Right canthus after environmental correction |
| `left_canthus_corrected` | float | Left canthus after environmental correction |
| `canthi_max` | float | Max of left and right canthus |
| `canthi_4_max` | float | Max across 4 canthus measurement zones |

**Thermal readings -- cheeks:**

| Column | Type | Description |
|--------|------|-------------|
| `right_cheek_temp` | float | Right cheek avg surface temp |
| `right_cheek_dry` | float | Right cheek, dry-skin region |
| `right_cheek_wet` | float | Right cheek, moisture-exposed region |
| `right_cheek_max` | float | Right cheek max |
| `left_cheek_temp` | float | Left cheek avg surface temp |
| `left_cheek_dry` | float | Left cheek, dry-skin region |
| `left_cheek_wet` | float | Left cheek, moisture-exposed region |
| `left_cheek_max` | float | Left cheek max |

**Thermal readings -- forehead and orbital:**

| Column | Type | Description |
|--------|------|-------------|
| `forehead_center` | float | Forehead center |
| `forehead_right` | float | Forehead right quadrant |
| `forehead_left` | float | Forehead left quadrant |
| `forehead_bottom` | float | Forehead bottom |
| `forehead_top` | float | Forehead top |
| `forehead_max` | float | Max forehead temp |
| `forehead_center_max` | float | Max of forehead center ROI |
| `face_max_temp` | float | Highest temp anywhere on the face |
| `orbital_right` | float | Right orbital region |
| `orbital_right_max` | float | Right orbital max |

## Output

Produce a `submission.csv` with 204 rows (plus header).

```
id,oral_temp
5,36.89
12,37.23
17,36.54
...
```

| Column | Type | Details |
|--------|------|--------|
| `id` | int | Must match IDs in `test.csv` |
| `oral_temp` | float | Predicted oral temperature (Celsius) |

## Metric

RMSE (root mean squared error). Lower is better.

```
RMSE = sqrt(mean((y_true - y_pred)^2))
```

For reference, just predicting the mean every time gives RMSE of about 0.53. A decent model should get below 0.35, and anything under 0.3 starts to be clinically useful (that's roughly the threshold where the prediction is close enough to trust).

Score locally with:

```bash
python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
```

## Constraints

- `id` is just a row identifier, don't use it as a feature.
- All other columns are fair game, including demographics.
- Predictions need to be in [30, 45] C.
- Everything is in Celsius. Don't convert to Fahrenheit.
