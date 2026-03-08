# Golden Workflow -- EPA Daily PM2.5 Prediction

## Data sanity checks

Start by profiling missingness. The co-pollutant columns (ozone, no2, so2, co) are missing 30-51% of values. Meteorological columns (temp, wind, humidity, pressure) range from 5% to 47%. Before doing anything else, verify that missingness rates are similar between train and test, because if they aren't, the split might have introduced bias.

Check the target distribution. PM2.5 has a heavy right tail -- mean around 8.5 ug/m3, median around 7, but with occasional spikes >100 (wildfire events, inversions). Since the metric is RMSE, those tail values carry outsized weight. Quantiles to look at: p50, p90, p95, p99, and max. Decide early whether you think the extreme values are learnable from the features or whether they're just noise you'll have to eat.

Look at geographic coverage. Latitude/Longitude will give you the spatial extent -- these are real EPA sites across the continental US plus Alaska, Hawaii, Puerto Rico, etc. Check whether `state_code` and `cbsa_code` have reasonable cardinalities and whether rare codes appear in test but not train (or vice versa).

Verify temporal features make sense: year should be 2020-2022, month 1-12, day_of_week 0-6, day_of_year 1-366.

## Validation strategy

5-fold cross-validation on the training set. Stratified on binned PM2.5 to keep the target distribution similar across folds (same bin edges used in the train/test split: 0-5, 5-10, 10-15, 15-25, 25-50, 50+).

Because the same site can appear in both train and test (it's a random not spatial split), there's no strict need for group-based CV by site. But be careful: if you build site-level features like site means, those need to be computed within each CV fold to avoid contamination.

Watch for overfitting on cbsa_code -- it's a categorical with potentially hundreds of levels, so regularization or target-encoding it inside CV is important.

## Preprocessing plan

Missingness handling is the most impactful preprocessing step here. Several approaches, roughly in order of complexity:

1. **Simple imputation**: Fill missing values with training-set medians. Quick, works as a baseline, but ignores that missingness is informative (a site missing SO2/CO likely has different PM2.5 patterns than one with full instrumentation).

2. **Missingness indicators + imputation**: Add binary columns for each feature's missingness, then fill with medians or zeros. Random forest models can split on the indicator directly. This tends to help quite a bit given the MNAR structure.

3. **Grouped imputation**: Compute fill values by state_code or cbsa_code. Sites in the same metro area share similar meteorology and co-pollutant relationships, so local medians are better than global ones.

4. **Iterative imputation (MICE / IterativeImputer)**: More principled but slow and may not outperform missingness indicators for tree models.

Scaling doesn't matter for tree-based models. If you're using linear models or neural nets, standardize after imputation.

For `cbsa_code`, either target-encode it or embed it. Don't one-hot encode it with hundreds of levels in a linear model -- you'll overfit badly.

## Baseline model

Train a LightGBM or XGBoost regressor with default hyperparameters on all 16 features, using NaN natively (both support missing value handling out of the box). This should beat the constant-mean baseline (RMSE ~6.81) substantially. Expect something in the 2.5-4.0 range on CV, depending on how well the spatial autocorrelation helps.

If you want a non-gradient-boosting sanity check, a Ridge regression on the numeric features (after imputation) should land somewhere around RMSE 5-6. Useful as a floor.

## Iteration plan

Starting from the tree baseline, things to try in roughly priority order:

1. **Missingness indicators.** Even though LightGBM handles NaN internally, explicit binary indicators for each imputed feature give the model a direct signal about site instrumentation. This is probably worth 0.1-0.3 RMSE.

2. **Cyclical encoding of temporal features.** `month` and `day_of_year` are cyclical -- December 31 is close to January 1, but the integer encoding says they're far apart. Sine/cosine encoding helps linear models; for trees it's less critical but still reasonable.

3. **Spatial features.** Latitude/Longitude interactions, distance to coast, elevation (would need an external lookup). Or just let gradient boosting learn axis-aligned splits on lat/lon.

4. **Target-encoded cbsa_code.** With proper in-fold encoding to avoid data leakage. This turns the high-cardinality categorical into a smooth numeric that captures the metro-area-level PM2.5 baseline.

5. **Lag-like features (careful).** You can't use yesterday's PM2.5 because it's the target. But grouping by site and computing site-level statistics from the training set (mean PM2.5 at that site, 75th percentile, etc.) is valid -- it's just a static site characteristic. Don't compute these from test data.

6. **Log-transform the target.** PM2.5 is right-skewed; predicting log(pm25+1) and back-transforming can reduce the influence of extreme values. But the metric is RMSE on the raw scale, so the back-transformed predictions need to be good in the original units, and Jensen's inequality means exp(E[log(y)]) underestimates E[y]. May need bias correction.

7. **Hyperparameter tuning.** Once features are set, Optuna or similar to tune learning rate, max depth, min child weight, subsample, colsample. Important to tune on your CV folds, not a single holdout.

8. **Blending / stacking.** Combine LightGBM, XGBoost, and maybe CatBoost. Diminishing returns compared to the items above, but often good for another 0.05-0.1 RMSE.

## Error analysis

Sort the test predictions by absolute error and look at the worst ones. They'll almost certainly be the extreme PM2.5 events (>50 ug/m3). Check what the features look like for those rows -- are the met features available, or are they all NaN? Is the model underpredicting all extremes, or only some?

Look at error by state and by month. PM2.5 has strong seasonality (winter inversions in the Mountain West, summer wildfire smoke on the West Coast). Some months might have much higher RMSE than others.

Check whether errors correlate with missingness. If rows with more missing features have systematically higher errors, your imputation strategy might need work. Or it might just mean that sites with fewer instruments are harder to predict, which is expected.

Residuals vs predicted values: if there's curvature, a log-transform or different loss function (Huber) might help. If residuals fan out at higher predicted values, heteroscedasticity is in play and quantile regression or a variance-stabilizing transform could be useful.

## Leaderboard safety

A constant prediction of the training mean gives RMSE ~6.81. Any reasonable regression model should substantially beat that. If your public score is worse than 6.8, something is fundamentally wrong (wrong column order, predictions shifted, etc).

Watch for train-test distribution shift. The test set was sampled from the same period and sites, so there shouldn't be major distribution differences. But if your CV score and leaderboard score diverge by more than about 10%, investigate. Common causes: accidental leakage in feature engineering (using test data to compute site means), or instability from overfitting to a handful of extreme values.

Don't overfit to the public leaderboard. With 18,740 test rows and RMSE in the 2-4 range, random variance in the public/private split can move your score by a few hundredths. Trusting your CV is better.

## Submission checks

Before submitting:

1. Verify the output has exactly 2 columns: `id` and `pm25`.
2. Row count = 18,740 (same as test.csv).
3. All `id` values match test.csv exactly (merge and check for NaNs).
4. `pm25` values are all non-negative. The scorer enforces this and will reject submissions with negatives.
5. No NaN or inf in the `pm25` column.
6. Predicted values should be in a reasonable range. If your max prediction is 500 when the training max is 395, double-check.

Run `score_submission.py --submission-path your_predictions.csv --solution-path solution.csv` locally to verify formatting before uploading. If it runs without error, formatting is correct.
