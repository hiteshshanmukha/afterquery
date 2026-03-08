# Gas Sensor Array Drift Dataset

## Overview

Readings from an array of 8 metal-oxide semiconductor (MOX) gas sensors exposed to 6 different gases at varying concentration levels. The experiment ran over 36 months in a controlled lab environment, producing 13,910 total measurements. Sensor drift (gradual change in sensor baseline and sensitivity over time) is a known phenomenon in this data, which is part of what makes it interesting for classification.

MOX sensors work by measuring changes in electrical conductance when gas molecules interact with the sensor surface. Different gases cause different conductance patterns across different sensor materials, and the combination of responses across the 8-sensor array creates a "fingerprint" for each gas. The 16 features per sensor capture both the steady-state response and dynamic characteristics (how fast the conductance changes, overshoot, settling behavior).

## Source

> Vergara, A., Vembu, S., Ayhan, T., Ryan, M.A., Homer, M.L., & Huerta, R. (2012). Chemical gas sensor drift compensation using classifier ensembles. Sensors and Actuators B: Chemical, 166, 320-329.

> Rodriguez-Lujan, I., Fonollosa, J., Vergara, A., Homer, M.L., & Huerta, R. (2014). On the calibration of sensor arrays for pattern recognition using the minimal number of experiments. Chemometrics and Intelligent Laboratory Systems, 130, 123-134.

UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset

## License

CC-BY-4.0

https://creativecommons.org/licenses/by/4.0/

Original data: https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset

**What I changed:**
- The original dataset ships as 10 separate batch files (one per collection period). I merged them into a single frame.
- Added an `id` column since the original has none.
- Renamed generic column names to a structured `s{sensor}_f{feature}` convention.
- Stratified 80/20 split on `gas_type` to create train and test sets.
- Dropped `gas_type` from test.csv.

## Features

| Column pattern | Count | Description |
|---------------|-------|-------------|
| `id` | 1 | Row identifier |
| `s1_f1` through `s1_f16` | 16 | Sensor 1 features (TGS2600 model) |
| `s2_f1` through `s2_f16` | 16 | Sensor 2 features (TGS2602 model) |
| `s3_f1` through `s3_f16` | 16 | Sensor 3 features (TGS2610 model) |
| `s4_f1` through `s4_f16` | 16 | Sensor 4 features (TGS2620 model) |
| `s5_f1` through `s5_f16` | 16 | Sensor 5 features (TGS2611 model) |
| `s6_f1` through `s6_f16` | 16 | Sensor 6 features (TGS2612 model) |
| `s7_f1` through `s7_f16` | 16 | Sensor 7 features (TGS2620-2 model) |
| `s8_f1` through `s8_f16` | 16 | Sensor 8 features (TGS2602-2 model) |
| `gas_type` | 1 | Target: 1=Ethanol, 2=Ethylene, 3=Ammonia, 4=Acetaldehyde, 5=Acetone, 6=Toluene |

Within each sensor's 16 features: features 1-2 are steady-state conductance measurements, features 3-16 capture transient dynamics (rise and decay characteristics at different time windows).

All features are continuous floats. No missing values.

**Note on gas concentration:** There's no concentration column in the data, but concentration is baked into the sensor readings. The transient features shift with concentration, and so do the raw magnitudes. So the model can pick up on concentration-dependent patterns even though it's not an explicit input. Not a problem for the classification task, just something to be aware of.

## Splitting & Leakage

Stratified random split, 80% train / 20% test, stratified on `gas_type`.

The data was collected across 36 months in 10 batches with increasing sensor drift. Because the split is random (not temporal), drift effects end up in both train and test. That means this is really just standard multi-class classification: the model trains on a mix of drifted and non-drifted samples, so it doesn't need to generalize across a temporal gap. A temporal split (train on batches 1-7, test on 8-10) would be a much harder problem, but that's not what we're doing here. No test-set information leaks into training.

**Feature correlation:** Each sensor's 16 features come from the same conductance curve, so within-sensor correlations are very high (often 0.9+). There's also moderate cross-sensor correlation since all 8 sensors see the same gas sample at the same time. Naive Bayes and similar independence-assuming models will struggle. Tree models and regularized linear models are fine, but keep in mind that 128 columns is not 128 independent dimensions of signal.

Features are purely physical sensor measurements, completely independent of the label assignment process. No leakage concerns.
