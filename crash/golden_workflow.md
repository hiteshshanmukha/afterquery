# How I'd approach this task

---

## 1. Data sanity checks

Before building anything, get familiar with what you're actually working with. This dataset has some real quirks that will bite you if you skip this step.

Things to check right away:
- Confirm `INJ_SEV` only has values in {0, 1, 2, 3, 4}. It does, but verify. The class split is roughly 49/22/16/11/2 -- that last class (fatal, class 4) is only about 2% of the data. Since the metric is RATWGT-weighted macro F1 and those fatal crashes carry high weights, you need to take the imbalance seriously from the start.
- Look at `VSPD_EST` value counts. You'll see that over 53% of the values are 998, which means "Not Applicable" (speed wasn't recorded). Another ~5% are 999 (Unknown). Only about 40% of rows have actual speed values. This is the single most important thing to catch -- if your model treats 998 as a real speed, it'll build splits around a phantom 998 mph cluster and everything goes sideways.
- Scan other columns for sentinel codes. `DRIVER_AGE` has 998/999, `CRASH_HOUR` has 99, `SPEED_LIMIT` has 98/99. Many of the categorical columns use 97/98/99 for unknown/not reported. Run `.value_counts()` on every column and look for suspicious spikes at round numbers near the top of each code range.
- Check `BODY_TYP` -- it has 67 distinct values. That's the full CRSS vehicle body classification, which is very granular (separate codes for 2-door sedan vs 4-door sedan vs hatchback). Tree models will handle this fine with enough data, but you might want to group these later.
- Look at `ROLLOVER`, `AIRBAG_DEPLOY`, and `EJECTED`. All three have strong correlations with `INJ_SEV` because they're during-crash outcomes, not pre-crash conditions. Rollover, airbag deployment, and occupant ejection are all determined by the crash event itself. These are leakage and must be dropped.
- Eyeball `RATWGT` -- it ranges from about 7 to 800 with a mean around 114. It's a survey weight, not a crash feature. The scorer uses it but your model shouldn't.

**Why this matters here:** CRSS is a government survey database, not a clean Kaggle dataset. The integer codes look like real numbers but aren't. Missing data is encoded as special sentinel values, not as NaN. If you skip this step you'll train on garbage.

---

## 2. Validation strategy

Macro F1 weighted by RATWGT is an unusual metric. You need your CV setup to reflect it or you'll optimize for the wrong thing.

- Use stratified 5-fold cross-validation on `INJ_SEV`. The stratification is important because class 4 is only ~2% of the data. With unstratified folds you could get folds with very few fatal crashes, making your per-fold F1 estimates noisy and unreliable.
- Always compute per-class F1 alongside the overall macro score. I've seen models that report 0.35 macro F1 but have literally 0.0 for class 4 -- you can't tell if you only look at the aggregate. Since the metric is macro, a zero on any class drags the whole score down by 20% of what it could be.
- Ideally, compute the RATWGT-weighted version of macro F1 on your validation set, not the unweighted one. The scorer weights by RATWGT, and the two can disagree. You can copy the logic from `score_submission.py` for this.
- 5 folds is plenty for 207k rows. More folds means more compute for marginal stability gains.

**Why this matters here:** The RATWGT weighting means that a few high-weight fatal crashes can swing your score more than thousands of minor ones. If your validation doesn't account for this, you might pick a model that looks good on unweighted CV but scores poorly on the actual metric.

---

## 3. Preprocessing plan

The main preprocessing challenge is dealing with the CRSS sentinel codes. Tree models don't need scaling or normalization, but they do need the sentinel values handled.

**Sentinel replacement:** For every column that uses sentinel codes, replace them with -1 (or NaN if your model handles missing values natively -- LightGBM does). The key ones:

```python
SENTINELS = {
    "VSPD_EST": [997, 998, 999],
    "DRIVER_AGE": [998, 999],
    "CRASH_HOUR": [99],
    "SPEED_LIMIT": [98, 99],
    "VEH_MAKE": [97, 98, 99],
    "BODY_TYP": [98, 99],
    "VEH_MODEL_YEAR": [9998, 9999],
    "RESTRAINT_USE": [97, 98, 99],
    "DISTRACTED": [96, 97, 98, 99],
    "DRINKING": [8, 9],
    "DRUG_INVOLVEMENT": [8, 9],
    "DRIVER_SEX": [8, 9],
    "MAN_COLL": [98, 99],
    "TYP_INT": [98, 99],
    "ROAD_ALIGN": [8, 9],
    "ROAD_SURF_COND": [98, 99],
    "LIGHT_COND": [8, 9],
    "WEATHER": [98, 99],
    "ROAD_CLASS": [8, 9],
}
```

Before replacing, create binary "is_missing" indicator columns for the important ones (at minimum `VSPD_EST`, `DRIVER_AGE`, `SPEED_LIMIT`). The missingness pattern is informative -- crashes where speed wasn't estimated tend to be different from those where it was.

**Drop `ROLLOVER`, `AIRBAG_DEPLOY`, and `EJECTED`.** These are during-crash outcomes. Using them gives you artificially good validation scores that won't reflect real predictive power.

**Feature grouping:** `BODY_TYP` has 67 codes. Group them into something like: passenger cars (1-12), SUVs (14-19), vans (20-29), pickups (30-39), buses (40-49), heavy trucks (50-69), motorcycles (78-89), other/unknown. Add the grouped version as a new feature, keep the original too for tree models that can handle high cardinality.

**Other features worth engineering:**
- Night flag from `CRASH_HOUR` (say, hours 21-5)
- Speed-missing interaction: `VSPD_EST` being 998 combined with `BODY_TYP` group might separate "minor fender bender where speed wasn't bothered with" from "motorcycle crash where speed wasn't recorded"
- `VEH_AGE` bins (0-2 = new, 3-7, 8-15, 15+, unknown)

**Why this matters here:** The CRSS sentinel codes are by far the biggest preprocessing hurdle. The code 998 in `VSPD_EST` covers 53% of the data. If you feed that to a tree model it'll happily split on "speed > 500" and create a massive leaf that mixes all kinds of crashes. Getting this right is worth more than any amount of hyperparameter tuning.

---

## 4. Baseline model

Get a working pipeline first, then iterate.

- A `DummyClassifier` predicting all class 0 scores about 0.168 on the weighted macro F1. That's the floor.
- A quick `RandomForestClassifier(n_estimators=200, class_weight='balanced')` with the sentinels replaced should get you in the 0.25-0.35 range. Set `class_weight='balanced'` from the start or the model will completely ignore classes 3 and 4.
- Sanity check feature importances. `RESTRAINT_USE`, `MAN_COLL`, `BODY_TYP`, and `VSPD_EST` should be among the top features. If `VSPD_EST` dominates everything, you probably still have the 998 sentinel in there. If `AIRBAG_DEPLOY` or `EJECTED` show up, you forgot to drop the leakage columns.
- Make sure your pipeline writes a valid submission.csv and the scorer runs on it. Better to find format bugs now than after a bunch of modeling work.

**Why this matters here:** The baseline tells you whether your preprocessing is sane. If the random forest gets below 0.20 with balanced class weights, something is wrong with your data pipeline -- probably sentinel codes still being treated as numbers.

---

## 5. Iteration plan

Once the baseline works, there are a few things that move the needle the most on this particular task. In rough priority order:

**1. Switch to LightGBM or XGBoost with sample weights.** Gradient boosting handles the integer categoricals well and generally beats random forests here. Use inverse-frequency class weights so the model pays more attention to rare classes. Something like:

```python
class_counts = np.bincount(y, minlength=5)
weights = len(y) / (5 * class_counts)
sample_weight = np.array([weights[label] for label in y])
```

With `num_leaves=63`, `learning_rate=0.05`, 1000-1500 rounds and early stopping, this should get you into the 0.35-0.40 range.

**2. Threshold tuning.** This is the single biggest lever after getting a decent model. The metric is macro F1, which means each class counts equally. A model trained with logloss will naturally bias toward the common classes. After training, adjust the predicted probability thresholds: multiply the logits for class 3 and class 4 by factors > 1 (try 1.5-3.0 for class 3, 2.0-5.0 for class 4) and argmax. Tune these multipliers on your OOF predictions using the actual weighted macro F1 metric. This alone can add 3-5 points.

**3. Feature engineering.** The body-type grouping, missingness indicators, and night flag mentioned in preprocessing. Also try target-encoding for `VEH_MAKE` (69 values) -- compute the mean INJ_SEV per make on the training data, with smoothing. Rare makes like Maserati have very different crash profiles than Toyota.

**4. Model stacking.** If you want to push further, collect OOF probability predictions from LightGBM, a simpler logistic regression (needs scaling + one-hot encoding), and maybe a KNN. Feed those into a simple meta-learner. Keep the meta-learner heavily regularized.

**Why this matters here:** The RATWGT-weighted macro F1 metric punishes you hard for neglecting rare classes. Threshold tuning exploits the mismatch between the logloss training objective and the macro F1 evaluation metric. On this dataset, the gap between raw argmax and tuned thresholds is often bigger than the gap between a mediocre model and a good model.

---

## 6. Error analysis

Once you have a model scoring in the 0.35+ range, look at where it fails.

- Print the confusion matrix normalized by true class. You'll see a lot of 0/1/2 confusion (the low-severity classes blur together), which is expected -- the distinction between "no injury" and "possible injury" is inherently fuzzy in crash reports.
- The critical failure: check if the model ever predicts class 4 at all. If recall for class 4 is near zero, your macro F1 has a hard ceiling. Fatal crashes are rare but they have distinctive signatures (high speed, motorcycle, no restraint). If your model isn't picking those up, the sentinel codes probably aren't handled right, or you need stronger class weighting.
- Look at high-RATWGT misclassifications. A single wrong prediction on a case with RATWGT=700 hurts the score more than getting 50 low-weight cases wrong. Sort your OOF errors by RATWGT and see what the worst ones look like.
- Check if certain `BODY_TYP` codes or `VEH_MAKE` values are disproportionately misclassified. Motorcycle crashes (body types 78-89) have very different severity profiles -- they're either minor or fatal with less middle ground.

**Why this matters here:** The weighted metric means a handful of high-weight misclassifications can dominate your score. Understanding which cases carry the most weight and why they're being misclassified tells you where to focus effort.

---

## 7. Leaderboard safety

The train/test split is a stratified random split across all 5 years. No temporal component, no group structure. So your CV should track the test score pretty closely.

Things to watch for:
- Don't overfit to the specific RATWGT distribution in your CV folds. The weights are survey design artifacts, not features. If your model secretly learns the weight pattern it won't generalize.
- Be careful with `CRASH_YEAR`. The 2020 data has fewer crashes (COVID), and the class distribution shifts slightly across years. If you're overfitting to year-specific patterns, your model might be less robust, though since train and test both contain all years this isn't a huge risk.
- If you're tuning threshold multipliers, tune them on OOF predictions from your full CV, not on a single held-out fold. Single-fold estimates of per-class F1 are noisy given how few class 4 samples there are.

**Why this matters here:** The main risk isn't train/test distribution shift (they're randomly split). The risk is overfitting to the particular class 4 samples in your training data, since there are only about 4,500 of them. Robust CV and stable threshold tuning matter more than squeezing another 0.5 points on a single fold.

---

## 8. Submission checks

Before submitting, run through these:

1. Exactly 51,810 rows plus the header. Not 51,809, not 51,811.
2. Columns are exactly `id` and `INJ_SEV`, nothing else.
3. Every `id` in test.csv appears exactly once in the submission.
4. Every `INJ_SEV` value is an integer in {0, 1, 2, 3, 4}. No floats, no NaN, no -1.
5. Run the scorer: `python3 score_submission.py --submission-path submission.csv --solution-path solution.csv` and verify it prints a number.
6. Spot-check some predictions. A single-vehicle motorcycle crash with no helmet and VSPD_EST around 70 should probably be class 3 or 4. A low-speed rear-end collision in daylight should be 0 or 1. If your model predicts otherwise, something's off.
7. Check your class distribution. If you're predicting < 1% class 4, your model probably needs stronger threshold adjustment for the rare class.
