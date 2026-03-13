# CRSS Crash Injury Severity Dataset

## Overview

Real police-reported motor vehicle crash data from the United States, 2018 through 2022. Sourced from NHTSA's Crash Report Sampling System (CRSS), which is a nationally representative probability sample -- each record has a survey weight (RATWGT) representing how many real-world crashes it stands for.

I joined four CRSS tables per year to build each row: the crash-level accident table, vehicle attributes for vehicle #1, driver demographics for that vehicle's driver, and their distraction record. Then concatenated all five years, filtered to cases with known injury severity (KABCO classes 0-4 only), and split 80/20 for train and test.

The target is `INJ_SEV`, the maximum injury severity on the KABCO scale (0=no injury, 4=fatal). It's quite imbalanced -- roughly half the data is class 0 and fatals are only about 2%. But the scoring metric is weighted by RATWGT, which makes those rare high-severity crashes count for a lot.

## Source

NHTSA Crash Report Sampling System (CRSS), 2018-2022.

Downloaded directly from NHTSA's static file server:
`https://static.nhtsa.gov/nhtsa/downloads/CRSS/{year}/CRSS{year}CSV.zip`

Documentation and codebook: https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system

Citation: National Highway Traffic Safety Administration. Crash Report Sampling System (CRSS), 2018-2022. Washington, DC: U.S. Department of Transportation.

## License

Public-Domain-US-Gov

CRSS is produced by NHTSA, a U.S. federal agency. The data is in the public domain under U.S. government works policy and free to use for any purpose.

https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system

## Features

| Column | Type | Notes |
|--------|------|-------|
| `id` | int | Row identifier |
| `CRASH_YEAR` | int | 2018-2022 |
| `CRASH_MONTH` | int | 1-12 |
| `CRASH_HOUR` | int | 0-23, 99=Unknown |
| `REGION` | int | 1=NE, 2=MW, 3=South, 4=West |
| `RURAL_URBAN` | int | 1=Urban, 2=Rural |
| `ROAD_CLASS` | int | Trafficway description (0-9) |
| `ROAD_ALIGN` | int | Straight vs curves (0-9) |
| `ROAD_SURF_COND` | int | Surface condition (1-11, 98/99=unknown) |
| `LIGHT_COND` | int | Lighting (1-9) |
| `WEATHER` | int | Weather conditions (1-12, 98/99=unknown) |
| `SPEED_LIMIT` | int | Posted speed limit in mph, 98/99=unknown |
| `MAN_COLL` | int | Manner of collision -- rear-end, head-on, etc |
| `TYP_INT` | int | Intersection type |
| `NUM_VEHICLES` | int | Vehicle count, capped at 6 |
| `NUM_PERSONS` | int | Person count, capped at 12 |
| `BODY_TYP` | int | CRSS body type code (~67 distinct values) |
| `VEH_MAKE` | int | Manufacturer code (~69 values) |
| `VEH_MODEL_YEAR` | int | Model year |
| `VEH_AGE` | int | `CRASH_YEAR - VEH_MODEL_YEAR` |
| `VSPD_EST` | int | Travel speed in mph. 997/998/999 = unknown/NA. ~53% are 998. |
| `FIRE_EXP` | int | Fire or explosion (0=no, 1=yes) |
| `DRIVER_AGE` | int | Age in years, 998/999=unknown |
| `DRIVER_SEX` | int | 1=Male, 2=Female, 8/9=unknown |
| `DRINKING` | int | Alcohol involvement (0/1, 8/9=unknown) |
| `DRUG_INVOLVEMENT` | int | Drug involvement (0/1, 8/9=unknown) |
| `DISTRACTED` | int | Distraction type (~21 codes, 0=not distracted) |
| `RESTRAINT_USE` | int | Restraint/helmet type (~14 codes) |

| `RATWGT` | float | CRSS survey weight (~7-800). Used in scoring, not a crash attribute. |
| `INJ_SEV` | int | **Target.** KABCO injury severity, 0-4. Only in train.csv. |

Most columns are categorical codes even though they're stored as integers. The CRSS codebook defines what each integer means. A lot of them have high-value sentinels (97, 98, 99, 997, 998, 999) that represent "unknown," "not reported," or "not applicable." These are the biggest data quality issue -- if your model sees 998 in `VSPD_EST` as a real speed, everything breaks.

`BODY_TYP` is probably the most granular column. It has 67 distinct values because CRSS has separate codes for 2-door sedans, 4-door sedans, hatchbacks, station wagons, and so on within just the "passenger car" group. Grouping these into broader categories (cars, SUVs, trucks, motorcycles) tends to help.

## Splitting & Leakage

Stratified random split (80% train / 20% test), stratified on `INJ_SEV` with `random_state=42`. No time-based split -- all five years are mixed together in both train and test. No group-based split either.

`ROLLOVER`, `AIRBAG_DEPLOY`, and `EJECTED` were during-crash outcomes — rollover, airbag deployment, and occupant ejection are all determined by the crash event itself, not before it. All three have been removed from the dataset.

`RATWGT` is a survey design weight, not a physical crash attribute. The scorer uses it to weight the metric, but it shouldn't be used as a feature.
