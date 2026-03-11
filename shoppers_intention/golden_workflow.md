# How I'd work through this

## 1. The page_values elephant in the room

Before doing any real modeling, look at `page_values`. Plot a histogram split by `revenue`. You'll see that virtually all purchases come from sessions where `page_values > 0`, and the vast majority of non-purchases have `page_values = 0`. A single threshold on `page_values` gets you most of the way to a good AUC.

Why this matters for the workflow:
- **Rationale:** If you don't understand how dominant `page_values` is, you'll waste time tuning features that barely matter. It's the first thing to investigate because it determines your modeling strategy.
- **What's specific to this dataset:** `page_values` is a Google Analytics metric that bakes in historical conversion information. It's not "future leakage" — it's legitimately computed from past sessions — but it already encodes the signal you're trying to predict. The question becomes: can you squeeze additional AUC from the other features on top of what `page_values` gives you?

Other things to check:
- `bounce_rate` and `exit_rate` are highly correlated (r > 0.9). Both measure tendency to leave, but exit_rate is always ≥ bounce_rate. The gap between them (`bounce_exit_gap`) captures whether visitors interact before leaving.
- Duration columns are extremely skewed. Most sessions have `admin_duration = 0` and `info_duration = 0`. When non-zero, they can be huge (hours). Consider log transforms.
- `month` distribution isn't uniform. November has the most sessions (likely holiday shopping), while some months have very few. May has a lot too, possibly Mother's Day traffic.
- `visitor_type` is ~85% returning visitors. New vs returning likely has different conversion patterns that interact with `page_values`.

## 2. Class imbalance is real but moderate

15.5% positive rate isn't crippling (unlike, say, fraud detection at 0.1%), but it's enough that a model can get lazy and predict "no purchase" most of the time.

**Rationale:** AUC-ROC is threshold-invariant, so extreme class imbalance mainly hurts through insufficient positive examples for the model to learn from, not through metric distortion. With ~1,500 positive labels in training, we have enough to learn good patterns. But the model still needs to rank positives above negatives, and if it puts too much probability mass near 0, the ranking quality degrades.

How I'd handle it:
- Start with `class_weight='balanced'` in tree models. This upweights positive examples by ~5.5x so the model pays attention to them.
- For gradient boosting, use `scale_pos_weight = n_neg / n_pos ≈ 5.5` in XGBoost, or `is_unbalance=True` in LightGBM.
- SMOTE or other oversampling is overkill here — there are enough positive examples that the model can learn the pattern without synthesis. Oversampling often hurts AUC in my experience.
- If you're stacking models, make sure the base models output well-calibrated probabilities. AUC doesn't require calibration, but stacking does.

## 3. Validation strategy

Stratified 5-fold cross-validation on `revenue`. The test set was also stratified, so CV should track test AUC fairly well.

**Rationale:** Stratification ensures each fold has ~15.5% positives. Without it, you could get a fold with 12% positive or 19% positive, which adds noise to your AUC estimate. With ~2,000 rows per fold, the variance is already noticeable.

What to track per fold:
- AUC-ROC (the official metric)
- Precision-recall AUC (more sensitive to class imbalance effects)
- Calibration curve (reliability diagram) — are the predicted probabilities well-calibrated?
- Feature importances — are they consistent across folds?

Avoid time-based splits even though the data has a `month` column. The random split matches how the test set was constructed.

## 4. Feature engineering

The derived features in the dataset are a start. Here's what else to try.

**Handle the page_values distribution:**
- `has_page_value = (page_values > 0).astype(int)` — binary flag. This alone is incredibly predictive.
- `log_page_values = log1p(page_values)` — compresses the heavy tail for sessions that do have page values.
- Consider binning `page_values` into 0, low (1-20), medium (20-50), high (50+). Different bins have very different conversion rates.

**Session engagement features:**
- `admin_share = admin_pages / (total_pages + 1)` — fraction of session spent on admin (account, settings). Higher admin share might signal intent (updating address, checking order status).
- `info_share = info_pages / (total_pages + 1)` — early-stage browsing, less purchase intent.
- `product_depth = product_pages * product_duration` — raw engagement volume on products. Log-transform this.
- `zero_browsing = (total_pages == 0).astype(int)` — sessions with no page views at all. These never convert.

**Categorical encoding:**
- `month` → ordinal or one-hot. Nov and Dec should be strong purchase months. Consider grouping into seasons.
- `visitor_type` → one-hot (3 categories). Or just `is_returning = (visitor_type == 'Returning_Visitor').astype(int)`.
- `operating_system`, `browser`, `region`, `traffic_type` → target encoding or frequency encoding. Too many categories for one-hot with this dataset size. Target encoding has leakage risk — do it within CV folds using the training fold only.

**Interaction features:**
- `page_values_x_returning = page_values * is_returning` — do returning visitors with high page values convert at different rates?
- `weekend_x_special_day = weekend * special_day` — weekend near special day might be different from weekday near special day.

**For tree models:** Most of this engineering is optional — trees can discover thresholds and interactions. But the binary flags (`has_page_value`, `zero_browsing`) and log transforms definitely help by making the model more efficient.

## 5. Baselines

- **All-zeros (predict no purchase):** AUC = 0.5 exactly. That's the floor.
- **Single feature: `page_values > 0`:** AUC around 0.89-0.91. Seriously. One feature.
- **Logistic regression on page_values + bounce_rate + exit_rate + product_duration:** AUC around 0.90-0.92.
- **Random Forest with all features, class_weight='balanced':** AUC around 0.90-0.93 without tuning.

## 6. Models

**Order of priority:**

1. **LightGBM with balanced class weights.** Best first model for tabular data. Start: learning_rate=0.05, max_depth=6, num_leaves=31, n_estimators=500, is_unbalance=True. Use early stopping. Expect AUC 0.91-0.93.

2. **Careful hyperparameter tuning.** Key parameters: learning_rate (0.01-0.1), num_leaves (15-63), min_child_samples (10-50), subsample (0.6-1.0), colsample_bytree (0.5-1.0), reg_alpha (0-5), reg_lambda (0-5). Use Optuna with 100+ trials, optimizing AUC on CV.

3. **Feature selection experiment.** Try removing `page_values` entirely and see how much AUC drops. If the other features can't get above 0.80 without it, they're mostly noise for this task. This is informative even if you keep the feature in the final model — it tells you whether your engineering is adding real signal.

4. **XGBoost as an alternative.** Different regularization paths than LightGBM, sometimes finds slightly different optimal regions. Try both, ensemble if they disagree.

5. **Stacking.** Out-of-fold predictions from LightGBM + XGBoost + Logistic Regression + maybe a simple MLP. Meta-learner: logistic regression with low C (0.01-0.1). Usually worth 0.5-1.0% AUC improvement.

6. **Threshold optimization.** After getting probability predictions, find the threshold that maximizes F1 or some business-relevant metric. AUC doesn't depend on threshold, but if you're outputting hard 0/1 labels, the threshold matters. Default 0.5 is almost certainly not optimal when the base rate is 15.5%.

**What I'd skip:**
- Neural networks. Not enough data, too many categorical features with few levels, no architecture advantage. Trees dominate here.
- SVM with RBF kernel. Can work but is slow to tune and doesn't output calibrated probabilities without extra work.

## 7. Error analysis

Where will the model struggle?

- **Sessions with `page_values = 0` that actually convert.** These are the hardest cases — about 5% of `page_values = 0` sessions result in a purchase. The model needs to use other signals (high product browsing, returning visitor, November, etc.) to catch them. Most false negatives will come from this group.

- **High page_values but no purchase.** About 50% of sessions with `page_values > 0` still don't convert. The model will want to predict "purchase" for all of these but half are wrong. Browsing time and bounce/exit rates might help distinguish.

- **"Other" visitor type.** Only ~1% of sessions. Too few to learn a reliable pattern. The model will probably just treat them like returning visitors.

- **Rare traffic types and browsers.** Some traffic_type and browser values appear fewer than 20 times in training. Target encoding on these is unreliable. Consider grouping rare categories into an "other" bucket.

- **Month-specific effects.** November has the most sessions and likely the highest conversion rate (holiday shopping). If the model overfits to November patterns, it may not generalize to lighter months. Check per-month AUC in CV.

Look at the confusion matrix at your chosen threshold. The most common error type will be false negatives (predicting 0 when it's 1), because the model is biased toward the majority class. Adjusting the threshold down from 0.5 will catch more true positives at the cost of more false positives.

## 8. Submission checklist

1. 2,466 data rows + header.
2. Two columns: `id` and `revenue`.
3. IDs match test.csv.
4. Every value is exactly 0 or 1. No floats, no NaN, no strings.
5. Run the scorer: `python3 score_submission.py --submission-path submission.csv --solution-path solution.csv`
6. Sanity check: the overall positive rate in your predictions should be somewhere in the 10-25% range. If it's 0% (all zeros) or 50%+, something's off.
7. If you have access to probabilities, plot the ROC curve. It should be smooth and convex. If it's jagged or dips below the diagonal, there's a bug.
