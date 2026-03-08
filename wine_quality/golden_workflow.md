# How I'd approach this task

---

## 1. Look at the data first

Before touching any models, spend some time just understanding what we're working with. Macro F1 is the metric here, and it weighs all classes equally — so if the model completely ignores class 3 or class 8, the score tanks. That makes the class distribution really important to understand upfront.

Things to check:
- Confirm `quality` only has values in {3, 4, 5, 6, 7, 8}. Classes 3 and 8 combined are around 7% of the data, which is pretty severe imbalance, not just "slightly skewed".
- Look at `wine_type` counts — whites outnumber reds roughly 3:1. And the quality distributions are different between them. Reds cluster around 5-6 and rarely hit 7-8.
- Peek at the correlation between `density`, `alcohol`, and `residual_sugar`. Especially for whites, these are heavily correlated (more sugar = denser). Might want to engineer something there or at least be aware of it for regularization.
- Run `df.describe()` and scan for anything weird. The original UCI data is clean (no NaNs), but double check. `total_sulfur_dioxide` has some high values in white wines (300+) — those are real, don't clip them.

---

## 2. Validation setup

Since we're scored on macro F1 and the rare classes matter a lot, the validation setup needs to preserve class proportions. If you do a naive random split, you might get folds with zero class-3 samples and your CV estimate becomes useless.

- Stratified 5-fold on `quality` is the way to go. If your framework lets you stratify on multiple columns, create a composite key like `quality` + `wine_type` and stratify on that. The reason is that red and white wines have different feature distributions, so a fold that's 90% white could look artificially good.
- Always print per-class F1 alongside the macro score. I've seen models that report macro F1 of 0.65 but have literally 0.0 F1 for class 3 — you won't catch that if you just look at the aggregate number.

---

## 3. Preprocessing

Tree models don't care much about scaling, but a few things still matter.

**Encoding `wine_type`:** Just map it to 0/1. It's binary, no need for anything fancier.

**Feature engineering — these actually help:**
- `free_so2_ratio = free_sulfur_dioxide / (total_sulfur_dioxide + 1)` — what fraction of the SO₂ is in the active (free) form. More meaningful than the raw numbers for predicting stability.
- `acid_ratio = fixed_acidity / (volatile_acidity + 0.01)` — wines where volatile acidity is high relative to fixed acidity tend to taste off. The tasters penalize this.
- `alcohol_density_ratio = alcohol / density` — teases apart whether density comes from alcohol or from leftover sugar.

**Don't clip outliers.** Seriously. Some white wines have `residual_sugar` over 60 g/dm³. Those are real sweet wines and they tend to get specific quality ratings. Clipping throws away useful signal.

**If using linear models:** StandardScaler, fit on train only, apply to both. The usual drill.

---

## 4. Baseline

Get something running fast so you know the pipeline works.

- Sanity floor: a `DummyClassifier` predicting the most common class (6) scores around 0.05-0.08 macro F1. Basically useless since it gets zero recall on most classes.
- A quick `RandomForestClassifier(n_estimators=100, class_weight='balanced')` should land in the 0.40-0.50 range for macro F1 without any tuning. Make sure to set `class_weight='balanced'` from the start or the rare classes get completely ignored.
- Sanity check the feature importances — `alcohol`, `volatile_acidity`, `sulphates` should be near the top. If they're not, something's off in the pipeline.

---

## 5. Improving the model

The main challenge is macro F1 on imbalanced classes. Optimizing accuracy won't help here — a model that just predicts 5 and 6 can get 70%+ accuracy but terrible macro F1.

Here's the order I'd try things:

1. **Switch to gradient boosting with class weights.** XGBoost or LightGBM with balanced sample weights. LightGBM is usually faster for this size of data. Set `num_class=6`.

2. **Tune per-class prediction thresholds.** After you have a model that outputs probabilities, don't just argmax. Add small offsets to the logit scores for underrepresented classes (3 and 8) and tune those offsets on your validation set. This is one of the biggest levers for macro F1 in my experience.

3. **Try ordinal-aware approaches.** Quality is ordered — predicting 6 when the truth is 5 isn't as bad as predicting 8 when the truth is 5. The `mord` library or a custom ordinal loss for XGBoost can exploit this. Not always a big win, but worth a try.

4. **Stacking.** Collect out-of-fold probability predictions from a couple of different models (say LightGBM, logistic regression, and maybe a KNN). Feed those into a simple logistic regression meta-learner. Keep the meta-learner simple (low C, like 0.1) so it doesn't overfit.

5. **Separate models for red vs white.** Red and white wines really do behave differently — different feature ranges, different quality distributions. Training separate models and merging predictions can help, though it also halves the red wine training set which is already small.

---

## 6. Error analysis

Once you have a decent model, look at where it's getting confused.

- Plot the confusion matrix (normalized by true class). You'll see a lot of 5↔6 confusion, which is fine — those classes are adjacent and hard to separate. 7↔8 confusion is also expected.
- The killer failure mode: the model never predicts class 3 at all. If recall for class 3 is zero, your macro F1 has a hard ceiling no matter how well you do on everything else. Check for this specifically.
- See if errors are concentrated in one wine type. Red wines with quality 7 or 8 are pretty rare in training, so those tend to get misclassified more.

---

## 7. Trusting your CV score

The test set was a stratified random split, so there shouldn't be any distribution shift between CV and test. Your offline macro F1 should track the leaderboard score pretty closely.

One thing to watch out for: a model might bump CV from 0.55 to 0.58 by getting better at classes 5/6, but actually get worse on the rare classes. Track per-class F1 across folds to make sure improvements are broad-based.

And obviously, don't use `id` as a feature.

---

## 8. Before submitting

Quick checklist:

1. 1,300 rows (plus the header row).
2. Columns are `id` and `quality`, nothing else.
3. IDs match test.csv exactly.
4. Every `quality` value is an integer in {3, 4, 5, 6, 7, 8}. No floats, no NaN.
5. Run the scorer: `python3 score_submission.py --submission-path submission.csv --solution-path solution.csv` — make sure it prints a number.
6. Eyeball a few predictions. A wine with high alcohol and low volatile acidity should probably be 7 or 8. If your model says 4, something's wrong.
