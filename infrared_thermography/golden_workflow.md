# How I'd work through this

## 1. Look at the data first

Only 814 training rows with 33 features, so the ratio of features to samples is not great. Worth spending time understanding what's going on before throwing models at it.

Thermal camera data can have calibration weirdness, and skin surface temp is a noisy proxy for core body temperature. You want to make sure the data makes physical sense before building anything.

What I'd check:
- No NaNs (the 2 rows with missing data were already dropped).
- `oral_temp` should be in the 35.5-40.5 range. There's one subject at 40.34 which is high -- verify that their facial thermal readings are also elevated. If the canthus reads 34 but oral says 40, something's wrong.
- All thermal features should land somewhere in 30-40 C. Anything outside that is probably a sensor glitch.
- `ambient_temp` ranges from about 21 to 26. Room temperature affects every thermal reading on the face, and that relationship is real physics, not just noise in the data. Worth plotting ambient vs. the main thermal features.
- `Distance` is 0.8m for most people. If it varies, that matters because it changes the camera's effective resolution.
- `temp_offset` has a pretty narrow range. Check whether it adds any predictive value on top of the raw temperatures.
- Run `df.describe()`, look for anything weird -- zero-variance columns, crazy outliers, etc.
- Demographics: about 55/45 male/female, ages skew toward 21-40, ethnicity is mostly White. These do matter since gender affects vasodilation and melanin affects infrared emissivity a little.

## 2. Validation

Small dataset means CV estimates are going to be noisy. RMSE can swing 0.05-0.10 between folds just from sampling variance.

Use 5-fold CV, stratified on quantile-binned `oral_temp` (5 bins). You want each fold to have some of the rare high-temperature subjects, otherwise the fold that gets them all will look artificially different from the others.

Track per-fold RMSE, not just the mean. If fold 3 has RMSE of 0.15 and fold 5 has 0.55, investigate what's going on. Probably a few extreme subjects landed in one fold.

KFold(10) would only give ~80 test samples per fold which is pretty thin. 5-fold is a reasonable middle ground. Could also try LOOCV as a sanity check since N is small enough.

## 3. Preprocessing

30 thermal features are extremely correlated because they're all measuring the same face at the same time. Without careful handling, models will either overfit on redundant features or spread coefficients around randomly.

For categoricals:
- `Gender`: just map to 0/1.
- `Age`: ordinal encode 0 through 4. Age affects thermoregulation -- older people tend to run a bit cooler.
- `Ethnicity`: one-hot encode. Melanin does affect emissivity in the infrared band these cameras use (8-14 um), so it's a real feature. But some groups are small, so regularization matters here.

Feature engineering ideas (all motivated by the physics):
- `canthus_asymmetry = max_right_inner_canthus - max_left_inner_canthus`. Big asymmetry usually means the subject wasn't facing the camera straight, which makes the reading less reliable.
- `canthus_cheek_gradient = canthi_max - (right_cheek_temp + left_cheek_temp) / 2`. The temperature drop from canthus (warmest point) to cheeks. This gradient changes with ambient temp but the ratio is more stable than raw values.
- `forehead_uniformity = forehead_max - forehead_top`. If the top of the forehead is way cooler than the max, hair might be interfering with the reading.
- `ambient_adjustment = canthi_max - ambient_temp`. A canthus reading of 35 in a 22 degree room is different from 35 in a 26 degree room.
- `left_right_cheek_diff = abs(left_cheek_temp - right_cheek_temp)`. Another symmetry check.

If using linear models or SVR, fit a StandardScaler on train only. Trees don't care but it doesn't hurt.

PCA on just the thermal features might be worth trying. With 30+ correlated features, you can probably capture most of the variance in 5-8 components. The first few PCs will likely correspond to "overall facial temperature" and "canthus vs. periphery gradient."

## 4. Baselines

Start simple, make sure the pipeline works.

- Predicting the mean (37.03) every time gives RMSE around 0.53. That's the floor.
- Single feature: fit a linear regression with just `canthi_max` or `face_max_temp`. In the thermal screening literature, the inner canthus is considered the best non-contact proxy for core temp. Should get around 0.35-0.40 RMSE on its own. If it doesn't, check your pipeline.
- Quick RF: `RandomForestRegressor(n_estimators=100, max_depth=8)` with all features. Expect something around 0.30-0.35. Look at feature importances. `face_max_temp`, `canthi_max`, `forehead_max` should be at the top. If `Distance` or a demographic feature ranks high, figure out why.

## 5. Model iteration

The goal is to beat the single-feature baseline (~0.38 RMSE). The tricky part is squeezing signal from 30+ redundant features on only 814 rows without overfitting.

What I'd try, roughly in order:

1. **Ridge or Lasso with all features.** Regularization handles the multicollinearity. Lasso will zero out the redundant columns, which is informative. Fast to run and easy to interpret.

2. **LightGBM or XGBoost.** Keep it constrained: `max_depth` 3-5, at least 10 samples per leaf, learning rate 0.05-0.1, 100-300 trees. The advantage here is capturing interactions between ambient conditions and thermal readings (the mapping from canthus to oral temp isn't the same at 22 C and 26 C ambient).

3. **SVR (RBF kernel).** Scale features first. Tune C and gamma with GridSearchCV. SVR tends to do well on small dense numeric datasets, and the epsilon-insensitive loss gives some built-in outlier handling.

4. **Stacking.** Out-of-fold predictions from Ridge + LightGBM + SVR, combined with a simple Ridge meta-learner. Usually worth 0.01-0.03 improvement.

5. **Feature selection with RFECV.** With this many correlated features on so few rows, cutting down to 8-12 features might actually help.

One thing to keep in mind: on 814 samples, if two models differ by less than ~0.02 RMSE, you can't really tell them apart. Don't over-tune.

## 6. Error analysis

Look at where the model gets it wrong. In medical temperature prediction, 0.5 C off is the difference between "normal" and "send them to secondary screening."

- Plot predicted vs. actual. Look for systematic bias: does the model under-predict febrile subjects (oral_temp > 38)? These are the cases that matter most and there aren't many in training, so regression to the mean is a real risk.
- Younger subjects (21-30) tend to have warmer facial skin but that doesn't always mean a higher core temp. Check if over-prediction correlates with age.
- Plot residuals against `ambient_temp`. If errors track with room temperature, the model isn't properly correcting for the environment. Bad sign for generalization.
- Check residuals by `Ethnicity`. Skin emissivity varies with melanin, and if the model consistently under-predicts for darker-skinned subjects, that's both a technical and an equity problem.
- Look at the 10 worst predictions. Are they from the same subgroup? Same ambient conditions? Or just random noise?

## 7. Trust your CV

204 test rows is small, so the test RMSE has real sampling variance. Two models 0.01 apart are basically indistinguishable.

- CV and test RMSE should be within about 0.03-0.05 of each other. If test is way better than CV, you got lucky. If way worse, check for distribution issues between splits.
- The split was stratified on temperature, but that doesn't guarantee the demographics are balanced. If the test set happens to be overwhelmingly male or overwhelmingly young, that could shift things.
- Don't repeatedly submit and tune against the test score. With only 204 samples you can overfit to the test set surprisingly fast.

## 8. Submission checklist

1. 204 data rows + header
2. Two columns: `id` and `oral_temp`
3. IDs match test.csv
4. All `oral_temp` values are floats in [30, 45]. No NaN, no strings.
5. Most predictions should land in 36.0-38.0. If you see 25 or 42, something broke.
6. Run the scorer locally: `python3 score_submission.py --submission-path submission.csv --solution-path solution.csv`
7. Spot-check: someone with `face_max_temp` of 36.5 should predict oral_temp around 37.0-37.3 (skin reads about 0.5-1.0 below oral). If the model says 35.0 for that case, debug it.
