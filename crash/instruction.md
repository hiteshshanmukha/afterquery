# Crash Injury Severity Prediction

## Objective

You have real motor vehicle crash records from NHTSA's Crash Report Sampling System (CRSS), years 2018 through 2022. Each row is one crash built by joining the accident-level, vehicle, driver, and distraction tables. Predict the injury severity.

Target column: `INJ_SEV`, coded on the KABCO scale:

| Class | Meaning |
|-------|---------|
| 0 | No Apparent Injury |
| 1 | Possible Injury |
| 2 | Suspected Minor Injury |
| 3 | Suspected Serious Injury |
| 4 | Fatal Injury |

The distribution is heavily skewed. Class 0 is about 49% of the data, class 4 (fatal) is only around 2%. The metric weights by survey sampling weight (RATWGT), so getting the rare severe crashes right matters way more than you'd expect from their frequency alone.

---

## Inputs

| File | Rows | What's in it |
|------|------|--------------|
| `train.csv` | 207,236 | All features + the `INJ_SEV` target |
| `test.csv`  | 51,810 | Same features minus `INJ_SEV`. Includes `RATWGT`. |

### Features

31 columns from the CRSS database. All integer-coded. Most of them are categorical codes, not real numbers, even though they look numeric. Lots of columns have sentinel values like 97, 98, 99, 997, 998, 999 that mean "unknown" or "not applicable." If you treat these as actual numbers your model will do something dumb.

The worst offender is `VSPD_EST` (travel speed). Over half the rows have value 998, which means "not applicable" -- speed wasn't estimated for that crash. That's not 998 mph.

Quick rundown of what's in the data:

**Time/place:** `CRASH_YEAR` (2018-2022), `CRASH_MONTH` (1-12), `CRASH_HOUR` (0-23, 99=Unknown), `REGION` (1-4 for NE/MW/S/W), `RURAL_URBAN` (1=Urban, 2=Rural)

**Road conditions:** `ROAD_CLASS` (trafficway type, 0-9), `ROAD_ALIGN` (straight vs curve, 0-9), `ROAD_SURF_COND` (dry/wet/snow/ice, 1-11 plus 98/99), `LIGHT_COND` (daylight/dark, 1-9), `WEATHER` (clear/rain/snow etc, 1-12 plus 98/99), `SPEED_LIMIT` (posted limit in mph, 98/99 = unknown)

**Crash config:** `MAN_COLL` (manner of collision -- rear-end, head-on, angle, etc), `TYP_INT` (intersection type), `NUM_VEHICLES` (1-6), `NUM_PERSONS` (1-12)

**Vehicle (for vehicle #1):** `BODY_TYP` (~67 distinct CRSS body codes -- very granular, has separate codes for 2-door sedan, 4-door sedan, hatchback, etc), `VEH_MAKE` (~69 manufacturer codes), `VEH_MODEL_YEAR`, `VEH_AGE` (= crash year minus model year), `VSPD_EST` (travel speed -- 997/998/999 are sentinels), `ROLLOVER` (0/1/2/3 plus unknowns -- **leakage, drop this**), `FIRE_EXP` (0=no, 1=fire/explosion)

**Driver (vehicle #1):** `DRIVER_AGE` (998/999 = unknown), `DRIVER_SEX` (1=Male, 2=Female, 8/9 = unknown), `DRINKING` (0/1), `DRUG_INVOLVEMENT` (0/1)

**Distraction:** `DISTRACTED` (~21 codes, 0=not distracted, 10=cell phone, etc)

**Occupant protection:** `RESTRAINT_USE` (~14 codes), `AIRBAG_DEPLOY` (~9 codes, 20=not deployed -- **leakage, drop this**), `EJECTED` (0=no, 1=total, 2=partial -- **leakage, drop this**)

**Weight:** `RATWGT` (float, survey sampling weight -- used in scoring, not a crash attribute)

---

## Output

Generate `submission.csv` with exactly 51,810 rows plus a header:

```
id,INJ_SEV
3,0
5,1
8,3
...
```

| Column | Type | Details |
|--------|------|---------|
| `id` | int | Must match the IDs in `test.csv` exactly |
| `INJ_SEV` | int | Prediction, one of 0, 1, 2, 3, 4 |

---

## Metric

RATWGT-weighted macro F1. Higher is better (0 to 1).

Each crash's contribution to precision and recall for its class gets scaled by its RATWGT value (the survey weight), then F1 is macro-averaged across all 5 classes equally. A fatal crash with weight 400 matters 40x more than a minor crash with weight 10 in the metric math.

This makes minority classes (especially class 4) critical. Models that only predict the common classes will score poorly.

```bash
python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
```

---

## Constraints

- **Drop `ROLLOVER`, `AIRBAG_DEPLOY`, and `EJECTED`.** These are during-crash outcomes, not pre-crash conditions. Rollover, airbag deployment, and occupant ejection are all determined by the crash itself, so using them is target leakage.
- **Handle sentinel codes properly.** The CRSS database uses numeric codes like 997, 998, 999 for "unknown" and "not applicable." Treating them as real numbers will wreck your model. `VSPD_EST` is the biggest one (53% of values are 998).
- **Don't use `RATWGT` as a feature.** It's a survey design variable. The scorer needs it, your model shouldn't touch it.
- Predictions must be integers in {0, 1, 2, 3, 4}. Floats, NaN, or out-of-range values will fail validation.
