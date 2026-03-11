# How I'd approach this task

---

## 1. Data sanity checks

Before modelling, get a feel for the data and confirm nothing is broken. The metric is macro F1, so every class matters equally — understanding the class distribution is critical.

Things to check:
- Confirm `Class` only has the 7 expected values: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA. No nulls, no typos.
- Look at class counts. DERMASON is ~26% of training data, BOMBAY is ~4%. That 6:1 ratio means a naive majority-class predictor will score poorly on macro F1.
- Run `df.describe()` and check for NaN or zero values. The original UCI data is clean, but verify. Area and ConvexArea should be strictly positive. ShapeFactor values should be in reasonable ranges.
- Plot pairwise scatter of `Area` vs `MajorAxisLength` coloured by class. You'll immediately see that BOMBAY beans form a tight cluster at very high Area (>150K pixels) — that class is trivially separable. The hard part is SEKER vs DERMASON vs SIRA which overlap heavily in size and shape.
- Check the correlation matrix. You'll see many features are highly correlated (Area-ConvexArea ~0.99, MajorAxisLength-EquivDiameter ~0.95). This dataset has severe multicollinearity by design since all features come from the same grain image geometry.

---

## 2. Validation strategy

Macro F1 penalises per-class failures equally. A model that nails the big classes but misses BOMBAY entirely will score badly. So the validation setup must preserve class proportions.

- **Stratified 5-fold CV** on `Class`. This ensures each fold has roughly the same class distribution as the full training set. With only ~418 BOMBAY samples in training, you'd get ~84 per fold — enough to get a reasonable F1 estimate for that class.
- **Print per-class F1** alongside the macro score after every fold. I've seen models that report macro F1 = 0.88 but have 0.60 F1 on SEKER because the model confuses it with DERMASON. You need visibility into per-class performance to catch this.
- The test set was a stratified random split, no temporal or group structure, so CV scores should track test performance closely. No need for time-based splits or group-aware folds.

---

## 3. Preprocessing plan

All 16 features are numeric, so there's no categorical encoding needed. But the feature space has specific properties worth handling.

**Scaling:** Tree-based models don't need it, but if you try SVM, KNN, or logistic regression, the features span wildly different ranges (Area is ~20K–250K, Eccentricity is 0–1). StandardScaler fitted on train only.

**Feature engineering — what's worth trying:**
- `size_category`: Bucket Area into 3-4 bins. BOMBAY is huge, DERMASON is small, others are in between. This gives tree models an easy split.
- `elongation_ratio = (MajorAxisLength - MinorAxisLength) / MajorAxisLength`: More interpretable than AspectRatio and captures how oval vs round a bean is. HOROZ beans are very elongated; SEKER is round.
- `convexity_deficit = (ConvexArea - Area) / ConvexArea`: How much of the convex hull is NOT filled — captures irregular bean shapes. BARBUNYA tends to have higher values.
- `perimeter_to_area = Perimeter / np.sqrt(Area)`: Normalised perimeter, captures surface roughness independently of size.

**What NOT to do:**
- Don't apply PCA to reduce dimensionality. The multicollinearity is informative — different classes cluster differently along correlated axes, and tree models handle this naturally. PCA would blend those signals.
- Don't remove "redundant" features like ConvexArea just because it's correlated with Area. The gap between them (Solidity) carries class signal — letting the model see both is better than pre-deciding which to drop.

---

## 4. Baseline model

Get a pipeline working fast with a quick model:

- **Sanity floor:** `DummyClassifier(strategy='most_frequent')` predicts DERMASON for everything → macro F1 around 0.04–0.05 (1/7 classes correct, zero recall on the other 6).
- **First real baseline:** `RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)`. Should land around **0.90–0.92 macro F1** without tuning. This dataset is relatively clean and the features are informative.
- Sanity check feature importances — `Area`, `Perimeter`, `ShapeFactor4`, and `Compactness` should rank high. If `Extent` or `Solidity` dominate, something might be off.
- Don't skip the `class_weight='balanced'` parameter. Without it, BOMBAY and BARBUNYA recall will drop noticeably.

---

## 5. Iteration plan

This dataset is cleaner than most real-world data, so the gap between baseline and ceiling is smaller. But there are still gains to be had:

1. **Gradient boosting.** Switch to LightGBM or XGBoost. For multi-class, use `objective='multiclass'`, `num_class=7`. LightGBM with `is_unbalance=True` or per-class sample weights handles the class imbalance. Expect macro F1 around **0.92–0.94** with default hyperparameters.

2. **Hyperparameter tuning.** Key knobs for LightGBM on this data:
   - `num_leaves`: try 31, 63, 127. The data isn't huge so deeper trees can overfit.
   - `min_child_samples`: increase from default (20) to 50-100 to regularize — the minority class BOMBAY has very clean boundaries and doesn't need deep splits.
   - `learning_rate` + `n_estimators`: use early stopping with patience ~50 on the CV macro F1. Start with lr=0.05, 1000 estimators.
   - `colsample_bytree`: try 0.7–0.9. With 16 highly correlated features, random column subsampling adds helpful diversity.

3. **Threshold tuning post-hoc.** Train the model to output probabilities, then adjust per-class thresholds on validation. For macro F1, this can help the minority classes. Specifically, lower the prediction threshold for BOMBAY and BARBUNYA so borderline cases get classified correctly. Use Optuna or grid search over threshold offsets.

4. **Stacking / blending.** Out-of-fold predictions from 3 diverse models:
   - LightGBM (tree-based, captures non-linear boundaries)
   - SVM with RBF kernel (captures smooth decision boundaries — works well for the overlapping classes)
   - KNN with k=5-15 (locality-based — BOMBAY is easily separable in feature space)
   Feed the 3×7 probability vectors into a logistic regression meta-learner. Keep it simple (C=0.1) to avoid overfitting the meta-features.

5. **Hard classes analysis.** The main confusion axis is SEKER ↔ DERMASON ↔ SIRA. These three are medium-sized, roundish beans with overlapping feature distributions. If the overall macro F1 is stuck, focus specifically on distinguishing these three:
   - Add pairwise interaction features between shape factors for these classes.
   - Try a two-stage classifier: first separate BOMBAY/HOROZ/CALI/BARBUNYA (easier), then run a specialist model on the SEKER/DERMASON/SIRA sub-problem.

---

## 6. Error analysis

Once a strong model is in place, examine the failure patterns:

- **Confusion matrix** (normalised by true class). Expected pattern:
  - BOMBAY: near-perfect — it's a size outlier, trivially separable.
  - HOROZ: high accuracy — most elongated bean, high AspectRatio.
  - CALI and BARBUNYA: some mutual confusion — both are large, kidney-shaped beans.
  - SEKER, DERMASON, SIRA: the main confusion cluster. These three are the performance bottleneck.

- **Misclassified samples.** Pull out the misclassified rows and check if they're genuinely borderline (feature values in the overlap zone) or if there's a pattern the model is missing. Sometimes a specific ShapeFactor value cleanly separates a subset that the model misses.

- **BOMBAY false negatives.** If any BOMBAY beans are misclassified, check if they're unusually small for their variety (smaller beans near the CALI size range). With only ~104 test samples, even 2-3 misclassifications drop BOMBAY F1 noticeably.

---

## 7. Leaderboard safety

This is a stratified random split with no temporal component, so there's minimal risk of train-test distribution shift. CV scores should match test scores closely.

Watch for:
- Overfitting to the validation set through excessive threshold tuning — use a held-out calibration set or nested CV for thresholds.
- The small BOMBAY test set (104 samples) means BOMBAY F1 has high variance — a single misclassification changes BOMBAY F1 by ~2%. Don't over-optimise for it.

---

## 8. Submission checks

Quick checklist:

1. 2,723 rows (plus the header row).
2. Columns are `id` and `Class`, nothing else.
3. IDs match test.csv exactly.
4. Every `Class` value is one of: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA. Case-sensitive, no extra whitespace.
5. Run the scorer: `python3 score_submission.py --submission-path submission.csv --solution-path solution.csv` — must print a number.
6. Spot-check a few predictions: if a bean has Area > 150,000, it should almost certainly be BOMBAY. If AspectRatio > 1.8, it's probably HOROZ. If these obvious cases are wrong, debug the pipeline.
