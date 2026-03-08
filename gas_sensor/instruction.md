# Gas Identification from Chemical Sensor Array

## The problem

An array of 8 metal-oxide semiconductor (MOX) gas sensors was exposed to six different gases at varying concentrations. Given the sensor readings for a sample, classify which gas it is.

The six gases (mapped to integer labels):

| Label | Gas |
|-------|-----|
| 1 | Ethanol |
| 2 | Ethylene |
| 3 | Ammonia |
| 4 | Acetaldehyde |
| 5 | Acetone |
| 6 | Toluene |

The classes are moderately balanced. Acetone (5) is the most common at ~22%, Ammonia (3) is the least at ~12%. No extreme imbalance, but the macro F1 metric still penalizes ignoring any single class.

## Data files

`train.csv` has 11,128 samples with labels. `test.csv` has 2,782 samples without labels.

## Features

128 numeric columns from 8 sensors (labeled `s1` through `s8`), each producing 16 derived features (`f1` through `f16`). Column names follow the pattern `s{sensor}_f{feature}`, so `s3_f12` is the 12th feature from the 3rd sensor.

The 16 per-sensor features capture both steady-state conductance changes and transient dynamics (rise time, decay characteristics, etc). All values are continuous floats. No missing data.

There is also an `id` column. Don't use it as a feature.

## What to submit

A CSV file with two columns: `id` and `gas_type`.

```
id,gas_type
4,2
9,1
15,6
```

`gas_type` must be an integer in {1, 2, 3, 4, 5, 6}. One row per test sample, matching the IDs in `test.csv`.

## Scoring

Macro-averaged F1:

```python
f1_score(y_true, y_pred, average='macro', labels=[1, 2, 3, 4, 5, 6])
```

Run it locally:

```bash
python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
```

## Notes

- The data was collected over 36 months with gradual sensor drift. The train/test split is random (stratified on `gas_type`), **not temporal**, so drift is mixed into both sets. This makes it standard multi-class classification rather than a drift-compensation problem.
- Gas concentration isn't an explicit feature, but it's baked into the sensor readings (especially the transient features). The model may partly learn concentration-dependent patterns alongside gas identity.
- Within-sensor feature correlations are very high (16 features from one conductance curve). Effective dimensionality is well below 128. Consider PCA or regularization.
- The 80/20 split is stratified on `gas_type`.
