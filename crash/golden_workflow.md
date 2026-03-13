# How I'd approach this task

---

## 0. Understand what you're actually predicting

Before touching any code, internalize the fundamental challenge here. You're predicting **injury severity from pre-crash conditions alone**. The features describe what was happening *before* impact — speed, road surface, vehicle type, restraint use, collision geometry. But injury severity is largely determined by **crash dynamics** that aren't in the data: delta-V (change in velocity on impact), intrusion depth, where the occupant's body hit the interior, whether the steering column collapsed.

In injury biomechanics, delta-V is the gold standard predictor of injury severity. It's not directly available here. What you can do is **approximate crash energy** from what you have: `VSPD_EST` gives pre-impact travel speed, `MAN_COLL` tells you the collision geometry (head-on vs rear-end vs sideswipe — head-on transfers far more energy), `BODY_TYP` tells you vehicle mass and structural rigidity (a motorcycle absorbs zero impact energy for the rider; an SUV has crumple zones). The model needs to learn these physics-based interactions from the data, since you can't hand it delta-V directly.

The KABCO scale itself introduces noise. It's a police officer's assessment at the scene, not a clinical diagnosis. The boundary between class 0 (no apparent injury) and class 1 (possible injury) is subjective — it depends on what the officer observed, whether the person complained of pain, whether they were transported to a hospital. Classes 2-4 are more reliable because visible injuries and fatalities are harder to misjudge. Expect the model to struggle most with the 0/1/2 boundary, and that's partly irreducible.

**Why this framing matters:** If you treat this as a generic classification problem, you'll build a generic model. If you understand that you're reconstructing crash physics from indirect measurements, you'll engineer features and interpret errors very differently.

---

## 1. Data sanity checks

CRSS is a probability sample of police-reported crashes, not a census. Every crash in the dataset was selected from a sampling frame, and its `RATWGT` tells you how many real-world crashes it represents. Fatal crashes are sampled at nearly 100% (low weights, ~7-30), while minor property-damage crashes are sampled at maybe 1-2% (high weights, 400-800). This is why the training set has ~2% fatals even though fatals are far rarer than that in the real crash population — they're overrepresented by design.

This sampling structure has direct modeling implications:
- The training data **overrepresents severe crashes** relative to reality. Your model sees proportionally more fatal and serious-injury cases than it would in a truly random sample. This is actually helpful for learning rare-class patterns, but it means raw class frequencies in the training set don't reflect real-world base rates.
- The scoring metric weights by `RATWGT` to undo this sampling bias and evaluate performance as if you'd seen the full national crash population. A minor crash with `RATWGT=700` represents 700 real-world crashes; getting it wrong costs you 700x as much as a single unweighted error.
- **Don't use `RATWGT` as a feature.** It's a survey design artifact, not a crash characteristic. But you *should* understand its structure to inform your loss function and validation setup.

Things to check:
- Confirm `INJ_SEV` is in {0, 1, 2, 3, 4}. The class split is roughly 49/22/16/11/2 — but remember this is the *sampled* split, not the population split. The real-world distribution is even more skewed toward class 0.
- Look at `VSPD_EST` value counts. 53% are 998 ("Not Applicable") and another ~5% are 999 ("Unknown"). Only ~40% have actual speed values. Speed tends to be "Not Applicable" for specific reasons: parking lot incidents, very low-speed crashes where the officer didn't bother estimating, or non-motorist crashes. The missingness is informative — it correlates with crash type and severity.
- Scan all columns for sentinel codes: `DRIVER_AGE` (998/999), `CRASH_HOUR` (99), `SPEED_LIMIT` (98/99), plus 97/98/99 patterns across most categoricals. Run `.value_counts()` on every column.
- Check `BODY_TYP` — 67 distinct values. CRSS separates 2-door sedans from 4-door sedans from hatchbacks. This granularity matters because vehicle structure determines occupant protection, but you'll want to also group into broad categories.
- Note that data quality varies systematically by severity. Fatal crashes (class 4) get thorough investigation — an officer spends hours documenting the scene, measurements are precise, fields are rarely missing. Minor crashes (class 0) get a quick report — speed is often "Not Applicable," restraint use may be assumed, distraction is rarely coded. This means **missingness itself is a severity signal**, which is a subtle but real source of information.

**Why this matters here:** CRSS is a government survey database with domain-specific encoding, not a standard ML dataset. The integer codes look numeric but aren't. Missing data is encoded as high-value sentinels, not NaN. And the sampling design means the data you see isn't a simple random draw from reality.

---

## 2. Validation strategy

Macro F1 weighted by RATWGT is an unusual metric. You need your CV to reflect both the macro averaging and the weight structure.

- Use stratified 5-fold cross-validation on `INJ_SEV`. Stratification matters because class 4 is only ~2% of the data — unstratified folds could have too few fatals for reliable per-class F1 estimates.
- **Always compute per-class F1.** A model reporting 0.35 macro F1 might have 0.0 for class 4 — you can't tell from the aggregate. Since the metric is macro, a zero on any class costs you a full 20% of the possible score.
- Compute the **RATWGT-weighted** version of macro F1, not just the unweighted one. The scorer weights by RATWGT, and these can disagree significantly. Copy the logic from `score_submission.py` for your validation function. If you only track unweighted F1, you might select a model that handles the overrepresented severe crashes well but fumbles the high-weight minor crashes that dominate the real-world evaluation.
- Consider whether to **train with RATWGT as sample weights** in the loss function. This makes the model optimize toward the same distribution the scorer evaluates on. The tradeoff: you lose the benefit of the overrepresented severe cases. In practice, try both weighted and unweighted training and compare on RATWGT-weighted validation F1. Often, training unweighted (to exploit the overrepresentation of rare classes) but evaluating weighted is the best combo.
- 5 folds is plenty for 207k rows. More folds means more compute for marginal stability.

**Why this matters here:** The RATWGT weighting means a few high-weight minor crashes can swing your score more than thousands of unweighted fatal crashes. Your validation must account for this or you'll optimize for the wrong thing.

---

## 3. Preprocessing plan

Two preprocessing challenges: CRSS sentinel codes, and engineering features that approximate the crash physics you don't directly observe.

### Sentinel replacement

For every column with sentinel codes, replace them with NaN (LightGBM handles native missing values well) or -1. The key ones:

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

**Before** replacing, create binary "is_missing" indicator columns for at least `VSPD_EST`, `DRIVER_AGE`, `SPEED_LIMIT`, and `RESTRAINT_USE`. As noted in section 1, missingness correlates with investigation thoroughness, which correlates with crash severity. A crash where every field is filled in was probably investigated carefully — likely a serious one. A crash with speed "Not Applicable" and restraint "Unknown" was probably minor and quickly documented.

### Feature engineering — approximating crash physics

Since you don't have delta-V, approximate crash energy from what you do have:

- **Kinetic energy proxy:** When `VSPD_EST` is known, `speed^2 * body_type_mass_estimate` approximates kinetic energy at impact. Head-on collisions (`MAN_COLL`) transfer nearly all of this as deformation energy; sideswipes transfer very little. Create an interaction: `VSPD_EST^2` × a collision-severity multiplier derived from `MAN_COLL` (head-on=1.0, angle=0.7, rear-end=0.4, sideswipe=0.1). This is a crude delta-V proxy.
- **Vehicle vulnerability grouping:** Group `BODY_TYP` (67 codes) by occupant protection level, not just size. Motorcycles (78-89) provide zero structural protection — injury is almost entirely a function of speed and what the rider hits. Large SUVs and trucks have high mass and crumple zones. Passenger cars are in between. A useful grouping: motorcycles, passenger cars (1-12), SUVs/crossovers (14-19), vans (20-29), pickups (30-39), buses (40-49), heavy trucks (50-69), other/unknown. Keep the original code too — trees can use both.
- **Motorcycle flag:** Motorcycle crashes have fundamentally different injury mechanics than enclosed-vehicle crashes. There's no cabin, no airbag, no crumple zone. A motorcycle rider at 40 mph with no helmet hitting a fixed object is likely class 3-4. The same speed in a modern sedan with seatbelt and airbags is likely class 0-1. A binary `is_motorcycle` flag and its interactions with speed and restraint use will help the model separate these two very different crash populations.
- **Unrestrained × speed interaction:** `RESTRAINT_USE` indicating no seatbelt/no helmet combined with high `VSPD_EST` is a strong fatality signal. Restraint use alone has ambiguous severity implications (people might unbuckle at low speeds), but unrestrained at highway speed is extremely dangerous.
- **Speed-missing reason:** `VSPD_EST=998` doesn't mean speed was zero — it means the officer didn't estimate it. This happens in low-speed parking lot crashes (speed wasn't relevant), pedestrian-involved crashes, and some multi-vehicle pileups. Cross `VSPD_EST_missing` with `MAN_COLL` and `BODY_TYP` group to distinguish "minor fender bender, speed irrelevant" from "complex crash, speed not recorded."
- **Night + rural + high speed:** Nighttime rural high-speed crashes have disproportionately severe outcomes due to delayed EMS response, higher speeds, and more frequent fixed-object impacts. Combine `CRASH_HOUR` (night flag, 21-5), `RURAL_URBAN`, and `SPEED_LIMIT`.
- **Driver risk profile:** Young drivers (16-25) and elderly drivers (75+) have different injury vulnerability. Young drivers crash at higher speeds; elderly occupants sustain worse injuries at the same delta-V due to frailty. Create age bins with biomechanically meaningful cutoffs: 16-25, 26-64, 65-74, 75+, unknown.
- **Vehicle age and safety equipment:** `VEH_AGE` correlates with safety technology. Vehicles from 2012+ have electronic stability control mandated. 2018+ have better crash structures. Very old vehicles (15+ years) lack modern safety features. Bin `VEH_AGE` as: 0-3 (newest, best safety), 4-7, 8-14, 15+ (oldest), unknown.
- **Target encoding for `VEH_MAKE`:** 69 manufacturer codes with very different crash profiles. Compute smoothed mean `INJ_SEV` per make on training folds. Luxury brands tend to have lower severity (better vehicle safety); motorcycles brands (Harley-Davidson, etc.) have higher severity. Use K-fold target encoding to avoid leakage.

**Why this matters here:** The sentinel codes are the biggest data-quality trap, but the bigger modeling challenge is that you're predicting a physics outcome from indirect measurements. The features that move the needle most are the ones that approximate crash energy and occupant vulnerability — not generic column transformations.

---

## 4. Baseline model

Get a working pipeline first, then iterate.

- A `DummyClassifier` predicting all class 0 scores about 0.168 on weighted macro F1. That's the floor.
- A quick `RandomForestClassifier(n_estimators=200, class_weight='balanced')` with sentinels replaced should get 0.25-0.35. `class_weight='balanced'` from the start or the model ignores classes 3 and 4 entirely.
- Sanity check feature importances. `RESTRAINT_USE`, `MAN_COLL`, `BODY_TYP`, and (cleaned) `VSPD_EST` should be among the top. If `VSPD_EST` dominates everything, the 998 sentinel is probably still being treated as a real speed.
- Write a valid `submission.csv` and run the scorer. Find format bugs now, not after modeling.

**Why this matters here:** The baseline tells you if your preprocessing is sane. If the RF gets below 0.20 with balanced class weights, something is wrong — probably sentinel codes still treated as numbers.

---

## 5. Iteration plan

Once the baseline works, here's what moves the needle on this particular task.

### 5a. Gradient boosting with cost-sensitive learning

LightGBM is the right tool here for specific reasons: it handles NaN natively (critical when 53% of your speed column is missing), it learns on integer categoricals without one-hot encoding (important with 67-value `BODY_TYP`), and it supports custom loss functions for the cost-sensitive approach you'll need.

Don't just use generic inverse-frequency class weights. The CRSS sampling design means your training set already overrepresents severe crashes. Instead, derive sample costs from the **scoring metric**:

```python
# Option A: Train with RATWGT-derived costs that mirror the evaluation metric
# This makes the training loss approximate what the scorer actually measures
sample_weight = train_df['RATWGT'].values

# Option B: Combine class weights with RATWGT
# Upweight rare classes AND weight by survey importance
class_counts = np.bincount(y, minlength=5)
class_weight = len(y) / (5 * class_counts)
sample_weight = np.array([class_weight[label] for label in y]) * train_df['RATWGT'].values
```

Try both and compare on RATWGT-weighted validation F1. Option A aligns training with evaluation; Option B additionally compensates for class imbalance. Which works better depends on how much the overrepresentation of severe crashes in the sample already handles the imbalance for you.

For a more principled approach, use **focal loss** instead of standard cross-entropy. Focal loss downweights easy-to-classify examples (the vast majority of class 0 cases that the model gets right quickly) and focuses learning on the hard boundary cases. LightGBM supports custom objectives:

```python
# Focal loss with gamma=2.0 focuses on hard examples
# More principled than post-hoc threshold hacking
def focal_loss(y_true, y_pred, gamma=2.0):
    # Implementation for LightGBM custom objective
    ...
```

With `num_leaves=63`, `learning_rate=0.05`, 1000-1500 rounds and early stopping, you should reach 0.35-0.40.

### 5b. Threshold calibration

If using standard logloss, the model's predicted probabilities will be miscalibrated toward common classes. You can correct this post-training, but understand *why* it's needed: logloss optimizes for log-likelihood, not F1. The optimal decision boundary for macro F1 is different from argmax on probabilities, especially when classes are imbalanced.

Multiply the predicted probability (or logit) for each class by a tunable scalar, then argmax:

```python
# Grid search multipliers on OOF predictions using the actual weighted macro F1
# Use OOF from full CV to get stable estimates
multipliers = [1.0, 1.0, 1.0, 1.0, 1.0]  # one per class
# Search: class 3 multiplier in [1.0, 3.0], class 4 in [1.5, 5.0]
adjusted = oof_probs * multipliers
predictions = adjusted.argmax(axis=1)
```

Tune on OOF predictions from your full CV (not a single fold — class 4 estimates from one fold are too noisy with only ~900 samples per fold). This can add 3-5 points.

If you use focal loss or properly calibrated cost-sensitive training from 5a, you'll need less threshold adjustment — the model already learned to take rare classes seriously.

### 5c. Domain-informed feature engineering

Add the physics-proxy features from section 3. The ones most likely to help, in rough order:
1. Motorcycle flag + motorcycle × speed interaction
2. Kinetic energy proxy (speed² × collision-type multiplier)
3. Unrestrained × speed interaction
4. Missingness indicators (especially `VSPD_EST` and `RESTRAINT_USE`)
5. Target-encoded `VEH_MAKE`
6. Night + rural + high speed combination
7. Driver age bins with biomechanical cutoffs

### 5d. Stacking (optional)

If pushing for maximum score: collect OOF probability predictions from LightGBM, XGBoost (different tree structure can capture different interactions), and maybe a logistic regression (linear decision boundaries can complement trees). Feed all probability vectors into a regularized meta-learner (logistic regression with L2 or a shallow LightGBM). Keep heavy regularization to avoid overfitting to the meta-features.

**Why this approach over alternatives?** Gradient boosting works well here because the features are a mix of ordinal-ish codes and true categoricals, with complex interactions that matter (speed × vehicle type × collision geometry). Neural networks could work but need more careful preprocessing (embedding layers for the categoricals) and don't give you native missing value handling. The marginal gain from deep learning on tabular data with 200k rows and 28 features is minimal at best.

---

## 6. Error analysis

Once you have a model scoring 0.35+, look at where and why it fails.

- **Confusion matrix by true class.** You'll see heavy 0/1/2 confusion — that's partly irreducible. The KABCO distinction between "no apparent injury" and "possible injury" depends on the officer's judgment at the scene, not objective clinical criteria. Don't chase perfect separation between classes 0-2; the score gains are in classes 3 and 4.
- **Class 4 recall.** If recall for class 4 is near zero, your macro F1 has a hard ceiling. Fatal crashes have distinctive signatures that the model should learn: high speed + motorcycle + no helmet, or high speed + unrestrained + head-on collision with a fixed object. If the model isn't predicting any class 4, either the sentinel codes aren't handled right, the class weighting is too weak, or the energy-proxy features are missing.
- **High-RATWGT errors.** Sort OOF misclassifications by RATWGT. A single wrong prediction on a minor crash with RATWGT=700 costs more than 50 low-weight errors. These high-weight cases are typically class 0 or 1 low-severity crashes that represent many real-world crashes. If the model is misclassifying these upward (predicting class 2-3 for actual class 0), your class reweighting may be too aggressive.
- **Motorcycle crashes.** Body types 78-89 have a bimodal severity distribution — riders are either fine (low-speed tip-over, class 0) or badly hurt (any real impact, class 3-4), with less middle ground than car crashes. Check if the model is handling this bimodality or just predicting the average.
- **Missing speed ≠ missing information.** If the model systematically misclassifies cases where `VSPD_EST` was "Not Applicable," it might be because it's treating the missing speed as genuinely unknown rather than as a signal. Speed not being recorded often means the crash was minor enough that the officer didn't bother estimating it.

**Why this matters here:** The weighted metric means a handful of high-weight misclassifications can dominate your score. Understanding *why* errors happen — whether it's KABCO subjectivity, missing physics, or data quality variation — tells you which errors are fixable and which aren't.

---

## 7. Leaderboard safety

The train/test split is stratified random across all 5 years. No temporal component, no group structure. CV should track test score closely.

- Don't overfit to `RATWGT` patterns. If your model implicitly learns weight structure, it's learning survey design, not crash characteristics.
- `CRASH_YEAR` 2020 has fewer crashes (COVID) and slightly different patterns. Since both train and test contain all years this isn't a major risk, but be aware.
- Threshold multipliers should be tuned on OOF predictions from full CV, not a single fold. With only ~4,500 class 4 samples total (~900 per fold), single-fold F1 estimates for fatal crashes are noisy.
- The main risk is overfitting to the specific class 4 cases in training. There are only about 4,500 fatals. Robust CV and stable thresholds matter more than squeezing another 0.5 points.

---

## 8. Submission checks

1. Exactly 51,810 rows plus header.
2. Columns are exactly `id` and `INJ_SEV`.
3. Every `id` in test.csv appears exactly once.
4. Every `INJ_SEV` is an integer in {0, 1, 2, 3, 4}. No floats, NaN, or -1.
5. Run: `python3 score_submission.py --submission-path submission.csv --solution-path solution.csv`
6. Spot-check: a single-vehicle motorcycle crash with no helmet at 70 mph → class 3 or 4. A low-speed rear-end in daylight → class 0 or 1. A high-speed head-on → class 3+. If predictions don't match basic crash-safety intuition, something's wrong.
7. Check class distribution. If predicting < 1% class 4, thresholds need adjustment (or the model isn't learning fatality patterns at all).
