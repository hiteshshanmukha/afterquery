# Approaching this problem

## Understanding the sensor data

This isn't a typical tabular dataset where each column is an independent measurement. The 128 features come from 8 physical sensors, each producing 16 readings. That structure matters.

Each sensor responds differently to each gas. Sensor 1 might spike for ethanol and barely twitch for ammonia, while sensor 5 does the opposite. The "fingerprint" across all 8 sensors is what identifies the gas. So the inter-sensor relationships are more informative than any single sensor read in isolation.

Before modeling, it's worth looking at:
- Per-sensor variance. Some sensors have much wider dynamic range than others. Sensors with flat responses across gases aren't contributing much and might just add noise.
- Feature correlation within a sensor. The 16 features per sensor come from the same conductance curve, so many are heavily correlated (0.9+ pairwise within a sensor is common). There's also cross-sensor correlation since all 8 sensors see the same gas sample. Realistically you've got maybe 20-30 independent dimensions, not 128. You can do per-sensor PCA (keep 2-3 components each, giving 16-24 features), global PCA, or just lean on tree model regularization (max_depth, min_samples_leaf). If you're using linear models or SVMs, you need to reduce dimensionality or regularize hard, or the multicollinearity will mess up your coefficients.
- Class separability. A quick PCA on the full 128 features and color by gas type will show you that some gases (probably ethanol and toluene) separate easily, while others (ethylene vs acetone) cluster together and will be the main source of confusion.

## Cross-validation strategy

Stratified k-fold, 5 splits. The classes are moderately balanced so there's no risk of empty folds, but stratification is still good practice given the macro F1 metric.

One subtlety: the data was collected across 36 months and sensor behavior drifts over time. Since the split is random (not temporal), drift is mixed into both train and test. So this is basically standard classification. The model sees all drift conditions during training and doesn't need to extrapolate. Your CV estimate should track test performance well. If this were a temporal split (train on early batches, test on late ones), you'd need drift compensation and the problem would be way harder. It's not. The "drift" in the dataset name is a property of the data, not the core challenge here.

Print per-class F1 every time. Macro F1 can look reasonable while one class has near-zero recall.

## Preprocessing

**Scaling matters here.** The raw sensor values span very different ranges. Some features are in the tens, others in thousands. Tree models aren't affected, but if you try logistic regression, SVM, or neural nets, you need to standardize. Fit StandardScaler on the training fold only.

**Dimensionality reduction.** 128 features for ~11k samples is fine for tree models, but for anything using distance metrics (KNN, SVM), the curse of dimensionality starts to bite. Two options:
- PCA to 20-40 components (usually captures 95%+ variance in sensor data like this)
- Per-sensor aggregation: compute mean, max, and std of each sensor's 16 features, collapsing to 24 features. Loses the transient dynamics but is much more compact.

**Feature engineering.** Sensor ratios can be powerful. The ratio of sensor 2's response to sensor 5's response, for example, can be diagnostic for a specific gas pair. You can generate all pairwise sensor ratios (using each sensor's mean feature, say) for 28 extra features. Most will be noise, but a tree model will ignore the useless ones.

## Baseline models

A random forest with 200 trees and default parameters should give macro F1 around 0.90-0.95. This is a dataset where the classes are fairly separable from the sensor fingerprints alone, so even a basic model does well.

`DummyClassifier(strategy='most_frequent')` predicts all acetone (class 5) and scores roughly 0.06 macro F1.

Check the confusion matrix after the baseline. The confusion pattern will tell you where to focus. My expectation: ethanol (1) and toluene (6) are well-separated, while the main confusion axis runs between ethylene (2) and acetone (5).

## Pushing the score higher

The gap between 0.93 and 0.98 macro F1 is where this gets interesting.

**Gradient boosting.** LightGBM or XGBoost. For this dataset size and feature count, default configs already do well. Tune `num_leaves`, `learning_rate`, and `n_estimators` on your CV folds. You can probably push to 0.96+ with a tuned gradient booster.

**SVM with RBF kernel.** Chemical sensor classification is one of the domains where SVMs still compete well. The reason: the decision boundaries between gases in sensor space tend to be smooth, non-linear manifolds. Scale the features first (mandatory for SVM). Use grid search over C and gamma. This can match or beat gradient boosting here.

**KNN as a complement.** Gas sensor data tends to form tight clusters. A well-tuned KNN (k=5-15, distance weighting, with PCA preprocessing) can serve as a solid second model for an ensemble.

**Stacking.** If you want to squeeze out the last fraction: train 2-3 diverse models (tree-based + SVM + KNN), collect their out-of-fold probability predictions, and feed them into a logistic regression meta-learner. Keep the meta-learner regularized (C=0.1 to 1.0).

**Threshold tuning.** Less impactful here than on severely imbalanced datasets, but still worth trying if you're close to your target. Offset the predicted probabilities for the classes where recall is lowest.

## Where models fail

The main failure modes on this data:

1. **Ethylene/acetone confusion.** These gases produce similar sensor responses at certain concentration levels. If your per-class F1 for class 2 or class 5 is lagging, this is probably why. Feature engineering targeting the sensors that discriminate between these two (look at which sensors separate them in univariate plots) can help.

2. **Concentration confounds.** A sample of ethanol at low concentration might look similar to a sample of acetaldehyde at high concentration. If you had concentration as a feature, this would be easy to resolve. You don't, but the transient features (the dynamic part of the sensor response) encode concentration implicitly.

3. **Sensor drift mixing.** Even though the split is random (so drift is mixed evenly), individual samples from late in the collection period may have different baselines than early samples. If you notice a cluster of misclassifications, check whether they come from a particular temporal batch.

## Submission checklist

1. 2,782 rows plus header
2. Columns: `id` and `gas_type` only
3. All IDs match `test.csv`
4. `gas_type` values are integers in {1, 2, 3, 4, 5, 6}
5. Run the scorer, confirm it prints a number
6. Check that you're predicting all 6 classes. If `nunique() < 6` something went wrong.
