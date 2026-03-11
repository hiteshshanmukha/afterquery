# How I'd work through this

## 1. Understand the domain first

Concrete strength prediction is one of the most studied problems in civil engineering ML, and there's real domain knowledge that helps. The key insight: **the water-cement ratio is the most important single predictor**. This is Abrams' Law from 1918. Lower water/cement ratio = less porosity = higher strength. Everything else modulates this basic relationship.

What to look at first:
- Plot `compressive_strength` vs `water_cement_ratio`. You should see a clear negative trend, but it's not perfectly linear — it follows an exponential decay curve. Samples with very low w/c ratios (< 0.3) are high-performance concrete and can reach 70-80 MPa.
- Plot `compressive_strength` vs `age`. This is the classic logarithmic curing curve. Concrete gains most of its strength in the first 28 days, then continues slowly for months. But the rate depends on the binder composition — slag and fly ash mixes are much slower to develop strength than pure cement mixes.
- Look at the correlation matrix. You'll see that `cement` and `water` aren't as strongly correlated to strength individually as `water_cement_ratio` is. That's the interaction effect. A model that only sees the raw features needs to learn this interaction from data; the derived ratio gives it the answer directly.
- Check `age` distribution. It's highly concentrated at standard testing ages: 3, 7, 14, 28, 56, 90, 180, 365 days. These aren't evenly distributed — 28 and 56 days are overrepresented. The model needs to generalize between these gaps.

## 2. The age problem

This is the trickiest part of the dataset. `age` creates a grouped structure — the same recipe tested at different ages gives different strengths. If you build a model that only cares about age, you'd predict "28-day concrete is ~35 MPa" which is roughly the mean. But the variance within each age group is enormous because different recipes have wildly different strengths.

What to do:
- Consider whether to treat age as a continuous variable or an ordinal/categorical one. The standard testing ages are discrete, but the underlying process is continuous. I'd keep it numeric and let `log_age` handle the non-linearity.
- For tree-based models, the raw `age` is fine — they can learn the step-function-like pattern at standard ages plus the continuous trend.
- The interaction between age and binder composition matters a lot. Pure cement mixes gain strength fast (strong at 7 days), while slag/fly ash mixes are weak early but can overtake cement mixes by 90+ days. If your model doesn't capture this interaction, it'll systematically mispredict old slag/fly ash mixes.

## 3. Validation strategy

The train set is only 818 rows. Validation strategy matters a lot at this size because a single fold can have high variance.

- **5-fold CV** with a fixed random seed. Not stratified (it's regression, not classification), but you could try binning the target into quantiles and stratifying on that to ensure each fold has a similar strength distribution.
- **Be wary of recipe leakage in CV.** Since the same recipe appears at multiple ages, a naive random split lets the model see the recipe in training and just interpolate on age for the validation fold. This inflates CV scores. Ideally, do a **group-based split on recipe** (group all rows with identical cement/water/slag/fly_ash/superplasticizer/coarse_aggregate/fine_aggregate into the same fold). This is harder to implement but gives a more honest estimate of generalization.
- For quick iteration, a simple 80/20 holdout (16% of 818 = ~130 rows) is fine. Just be aware that results are noisy at this size.

## 4. Feature engineering

The pre-computed derived features (`water_cement_ratio`, `total_binder`, `coarse_fine_ratio`, `log_age`) are a good start. More ideas:

**Interaction terms:**
- `cement_x_age = cement * log_age` — captures how cement content interacts with curing time.
- `slag_ratio = blast_furnace_slag / total_binder` — fraction of binder that is slag. High slag mixes behave differently.
- `fly_ash_ratio = fly_ash / total_binder` — same idea for fly ash.
- `superplasticizer_per_binder = superplasticizer / total_binder` — normalized admixture dosage.

**Polynomial features on key variables:**
- `wc_squared = water_cement_ratio ** 2` — captures the exponential nature of Abrams' law.
- `age_squared = age ** 2` or `sqrt_age = sqrt(age)` — alternative age transforms if log_age isn't enough.

**Aggregate features:**
- `total_aggregate = coarse_aggregate + fine_aggregate` — total inert volume.
- `binder_to_aggregate = total_binder / total_aggregate` — richness of the mix.
- `water_to_binder = water / total_binder` — like w/c ratio but includes supplementary cementitious materials.

**Binary flags:**
- `has_slag = (blast_furnace_slag > 0).astype(int)` — whether the mix uses slag.
- `has_fly_ash = (fly_ash > 0).astype(int)` — whether fly ash is present.
- `has_superplasticizer = (superplasticizer > 0).astype(int)` — these mixes tend to be high-performance.

For tree models, most of these aren't necessary — the model can learn splits on the raw features. But for linear models, the interactions and polynomial terms are critical.

## 5. Baselines

- **Training mean (35.7 MPa):** RMSE ~16.8. That's the floor.
- **Linear regression on water_cement_ratio + log_age:** Should get RMSE ~12-13. Two features that capture the two main effects.
- **Ridge regression with all features + interaction terms:** Expect RMSE ~8-10. The non-linearities hurt linear models here.
- **Random Forest (default hyperparams):** Should get RMSE ~5-7 without any tuning. Trees handle the non-linear interactions between w/c ratio and age naturally.

## 6. Models

**Order of priority:**

1. **LightGBM / XGBoost.** This is a small tabular dataset — gradient boosting is the natural choice. Start with conservative settings: max_depth 5, learning_rate 0.05, 200-500 trees, min_child_samples 10. The small dataset size means overfitting is a real risk. Use early stopping on your validation set.

2. **Tuned gradient boosting.** Hyperparameter search with Optuna or RandomizedSearchCV. Key parameters to tune: learning_rate (0.01-0.1), max_depth (3-8), min_child_samples (5-30), subsample (0.6-1.0), colsample_bytree (0.6-1.0), num_leaves (15-63), reg_alpha (0-10), reg_lambda (0-10). With only 818 training rows, regularization matters.

3. **Stacking / blending.** Combine predictions from LightGBM + XGBoost + Ridge + maybe SVR with an RBF kernel. SVR is actually quite competitive on small datasets with well-scaled features. Use out-of-fold predictions for the meta-learner. A simple Ridge meta-learner is usually enough.

4. **Neural network (optional, likely won't beat trees).** The dataset is too small for deep learning to shine. If you try, use a small MLP (2-3 hidden layers, 64-128 neurons each) with strong dropout (0.3-0.5) and weight decay. BatchNorm helps stabilize training on small data.

**What to skip:**
- KNN — performs poorly because the feature space is relatively high-dimensional (12 features) and not all features are equally informative.
- Kernel methods with custom concrete-science kernels — interesting in theory but overkill for this data size and task.

## 7. Error analysis

Where will models struggle?

- **High-performance concrete (strength > 60 MPa).** These are rare in the dataset and have unusual mix designs (very low w/c, superplasticizer, often slag but no fly ash). Models tend to underpredict them because there aren't enough examples.
- **Very young samples (age 1-3 days).** Strength at early ages is highly variable and depends a lot on curing conditions (temperature, moisture) that aren't in the dataset. Models will have higher errors here.
- **Slag/fly ash mixes at old ages.** These mixes develop strength on a fundamentally different curve. If the model doesn't capture the binder_type × age interaction, it'll systematically underestimate old slag/fly ash concrete.
- **Outlier recipes.** Some samples have unusual proportions (e.g., very high fly ash content). The model has few similar examples to learn from.

Plot predictions vs. actuals. A good model should cluster around the 45-degree line. Typical failure: predictions get "squeezed" toward the mean — underpredicting strong concrete and overpredicting weak concrete. This happens with over-regularized models.

Compute residuals stratified by `age` and by `water_cement_ratio` bins. If errors are systematic in any bin, the model is missing a structural pattern.

## 8. Submission checklist

1. 212 data rows + header.
2. Two columns: `id` and `compressive_strength`.
3. IDs match test.csv.
4. All predictions are positive floats. No NaN, no strings, no negatives.
5. Predictions should mostly fall in the 5-80 MPa range. If you see values below 0 or above 100, something's wrong.
6. Run the scorer: `python3 score_submission.py --submission-path submission.csv --solution-path solution.csv`
7. Spot-check: a sample with high cement, low water, old age should predict 50+ MPa. A young sample with high water should be 10-20 MPa. If your model says otherwise, debug it.
