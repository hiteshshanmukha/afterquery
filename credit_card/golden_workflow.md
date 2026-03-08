# Golden Workflow — Credit Card Default Prediction

## 1. Sanity-check the data first

Before doing anything, poke around the data to catch encoding issues early. Things that trip people up on this dataset:

- `EDUCATION` has values 0, 5, 6 that aren't in the docs. They're rare (~5% total) but they exist in both train and test. Just bucket them together as "unknown" — don't drop or impute.
- Some `BILL_AMT` values are negative. That's from overpayment — the customer has a credit balance. Don't clip to zero, the sign carries information.
- `PAY_X = -2` (card not used that month) and `PAY_X = -1` (paid in full) get treated as the same thing a lot, but they have different default rates. Check them separately.
- Note: the PAY columns are PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 — there's no PAY_1. That's inherited from the original UCI dataset naming. Don't assume it exists.
- The financial columns range from about -10k to 500k+ NTD. Run `describe()`, make sure nothing is wildly off vs `LIMIT_BAL`.
- Default rate in train should be around 22-23%. If it's way off something went wrong with the split.
- There shouldn't be any missing values — confirm with `df.isna().sum().sum()` anyway.

## 2. Validation setup

5-fold stratified CV on the target, using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

Stratification matters here because 22% positive rate means unstratified folds can swing between 18-27% positive, which makes the AUC estimates noisy.

Worth tracking calibration alongside AUC — a model can rank well but produce garbage probabilities. `calibration_curve` from sklearn is fine for this.

Don't bother optimizing a classification threshold. AUC doesn't care about thresholds.

## 3. Preprocessing

The columns are a mix of categorical codes, ordinal delay statuses, and financial amounts. They need different handling.

Handle the weird category codes:
- EDUCATION: map {0, 5, 6} → single "unknown" bucket (e.g. all → 5)
- MARRIAGE: map {0} → 3

For PAY_X: tree models handle the raw integers fine. If using logistic regression, worth adding `PAY_X_any_delay = (PAY_X > 0)` and `PAY_X_delay_months = max(PAY_X, 0)` as separate features.

Useful engineered features:
- `pay_to_bill_ratio_i = PAY_AMTi / (BILL_AMTi + 1)` — how much of the bill they actually paid
- `utilisation = BILL_AMT1 / (LIMIT_BAL + 1)` — credit utilisation, strong default signal
- `avg_delay_last_3m = mean(PAY_0, PAY_2, PAY_3).clip(0)` — recent delay severity
- `balance_trend = BILL_AMT1 - BILL_AMT6` — growing balance = trouble
- `total_paid_6m = sum of PAY_AMT1..6` — proxy for financial capacity
- `num_late_months` — count of months where PAY_X > 0

Scaling: only needed for linear models. Fit scaler on train folds only. Don't standardize PAY_X before computing avg_delay or it loses meaning.

## 4. Baseline

Start simple to make sure the pipeline works.

LogisticRegression with `class_weight='balanced'` on the raw features (label-encode EDUCATION/MARRIAGE, leave PAY_X as is) should get roughly AUC 0.72-0.75.

A RandomForest (200 trees, balanced weights) should land around 0.77-0.79 without any tuning.

If you're below 0.70, something is broken — probably `id` leaking in as a feature, or BILL_AMT negatives got clipped, or the ID join is wrong.

Quick sanity check: look at RF feature importances. PAY_0 and PAY_2 should be top features. If SEX or AGE are dominating, something's off.

## 5. Improving the model

Work through these roughly in order, running full CV each time:

1. Add the engineered features from section 3. Usually worth +0.01-0.02 AUC with trees.

2. Switch to LightGBM. Set `scale_pos_weight = N_neg/N_pos` (about 3.45). Tune `num_leaves` (31-127), `learning_rate` (0.01-0.1), `min_child_samples` (20-200), `feature_fraction` (0.6-0.9), `lambda_l1` (0-1). Should get 0.80-0.83.

3. Try probability calibration — `CalibratedClassifierCV(method='isotonic', cv='prefit')` on a 10% held-out calibration set. LightGBM probabilities tend to be overconfident.

4. Blend or stack with XGBoost. Simple average of LightGBM + XGBoost predictions works surprisingly well at this dataset size. If stacking, use logistic regression (C=0.5) on the two OOF prediction columns.

5. Optional: monotonicity constraints on PAY_0, PAY_2, utilisation, num_late_months. Both LightGBM and XGBoost support this. Helps generalization, prevents weird non-monotone patterns.

## 6. Error analysis

Look at the hardest predictions — sort OOF by `|pred_prob - true_label|` and look at the top 200 or so.

Common failure mode: customers with PAY_0 = 0 (made minimum payment) who still default. They look fine by payment status but their balance is climbing fast. The `balance_trend` feature helps catch these.

Other things to check:
- Calibration curve — if the model is underconfident in the 0.3-0.6 range, isotonic calibration should help
- AUC broken out by EDUCATION group, especially the undocumented codes (0/5/6). If it's way lower for those, add an explicit unknown-education flag
- AUC by LIMIT_BAL quintile — high-limit customers who default are rare and the model often misses them. An interaction like LIMIT_BAL × utilisation can help

## 7. Avoiding overfit

Since the test set is a stratified random split (no temporal shift), CV AUC should track test AUC pretty closely. If the gap is more than ~0.01, you're probably overfitting — usually from too many tuning rounds on one train/val split.

Use 5-fold CV as the main signal rather than a single hold-out. With only ~5500 positives in 25k rows, a single split has high variance.

All feature selection has to happen inside the CV loop — no peeking at test performance to pick features.

## 8. Final checks before submitting

- 5000 data rows + header
- Columns are `id` and `default_prob` (exact names)
- IDs match test.csv: `set(sub.id) == set(test.id)`
- All probs in [0, 1], no NaNs
- Run the scorer: `python3 score_submission.py --submission-path submission.csv --solution-path solution.csv` — should get a number above 0.5
- Mean predicted probability should be around 0.22-0.25. If it's near 0.5 the model isn't learning the class distribution
