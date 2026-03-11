# How I'd work through this

## 1. Understand the spatial structure

This is fundamentally a spatial pricing problem. The single most important thing about a Delhi rental listing is *where* it is. Before touching models, map the data.

What to look at:
- Scatter plot of latitude vs. longitude, colored by monthly_rent. You'll see clear spatial clusters: Delhi South (high rents), Delhi Central (high), Delhi East (low-medium), outer suburbs (low). This spatial pattern is the dominant signal.
- Plot rent vs. size_sqft colored by suburb. The slope of the rent-size relationship varies by location. A sqft in Delhi South is worth 3-4x a sqft in outer East Delhi. If your model has a single global size coefficient, it'll underpredict premium areas and overpredict cheap ones.
- Boxplot of rent by suburb. "Delhi South" and "Delhi Central" will have the highest medians and widest spreads. "North Delhi" and outer suburbs will be lower. This suburb-level effect is strong and should be your first feature to nail down.
- Check `metro_dist_km` vs. rent. Properties within 0.5 km of a metro station command a premium, especially in Delhi East and North Delhi where metro connectivity is a differentiator. The effect is weaker in Delhi South where people drive anyway.

**Rationale:** Delhi's rental market is wildly heterogeneous. Understanding the spatial price surface before modeling tells you what features matter and what interactions to engineer. Without this, you might build a model that treats all sqft equally, which is wrong.

## 2. The locality / high-cardinality problem

`locality` has ~760 unique values, and `locality_grouped` still has ~190. This is the biggest modeling challenge. Locality is the most predictive feature, but naive one-hot encoding creates a ~190-column sparse matrix that doesn't generalize well to rare localities.

**How I'd handle it:**
- **Target encoding** (mean rent per locality from training data) is the most effective approach. It compresses ~190 categories into a single numeric feature. But it leaks the target, so you MUST do it within cross-validation folds — compute the mean from the training fold only and apply to the validation fold. Use smoothing (blend with the global mean when a locality has few observations).
- **Frequency encoding** (count of listings per locality) captures "popularity" but not price level. Include it as a secondary feature.
- **Suburb-level features.** Suburb has only 12 values, so one-hot is fine. Or compute suburb-level statistics: median rent, median size, count. These are coarser but more robust for rare localities.
- **Coordinate-based proxy.** For listings in "Other" locality (rare), the model only has lat/lon to infer neighborhood character. Clustering the coordinates into ~50 spatial bins (K-means on lat/lon) gives a location ID that works even for unknown localities.

**What's specific to this dataset:** The "Other" category in `locality_grouped` contains ~8% of listings (~1,400 rows). For these, the model has no locality signal and must rely on coordinates + distances. While most listings retain their named locality, ensuring robust predictions for the "Other" group is still important for overall model quality.

## 3. Validation strategy

The split is random and stratified on price quartile, so standard CV should work well.

- **5-fold CV** stratified on price quartile. This ensures each fold has a similar rent distribution.
- **Additionally**, track per-suburb RMSE. If the model nails Delhi South but fails in North Delhi, that's useful to know.
- **Group-based CV by suburb** (optional but informative). Hold out all listings in one suburb and predict them using only other suburbs. This tests whether the model generalizes to unseen areas — relevant if the "Other" suburb (outer NCR) is underrepresented.

**Rationale:** Random CV will give optimistic estimates because the model sees other listings from the same locality in training. The per-suburb RMSE check catches cases where the model memorizes locality patterns but doesn't generalize.

## 4. Feature engineering

Beyond what's already in the dataset:

**Spatial features:**
- `lat_x_lon = latitude * longitude` — simple interaction that can separate quadrants.
- `lat_binned`, `lon_binned` — discretize coordinates into ~20 bins each. Quick spatial bucketing.
- `dist_ratio = metro_dist_km / (center_dist_km + 0.1)` — are you near a metro station despite being far from center? That's well-connected suburbs.
- `metro_close = (metro_dist_km < 0.5).astype(int)` — binary: walking distance to metro.

**Property features:**
- `is_luxury = (property_type.isin(['Villa'])).astype(int)` — luxury segment behaves differently.
- `size_bucket = pd.qcut(size_sqft, q=10, labels=False)` — discretized size for modeling non-linear effects.
- `large_property = (size_sqft > 2000).astype(int)` — the price-size relationship changes slope above ~2000 sqft.
- `small_property = (size_sqft < 500).astype(int)` — very small properties may follow different pricing dynamics.

**Target encoding (careful):**
- `locality_median_rent` — median rent in the locality from training data. Use regularized/smoothed encoding.
- `suburb_median_rent` — same at suburb level. More stable, less leaky.
- `suburb_x_bedrooms_rent` — median rent for same suburb + bedroom count combination. Captures that a 3BHK in Delhi South prices very differently from a 3BHK in outer Delhi.

**Log transform the target:**
- Rent is heavily right-skewed. Training on `log(monthly_rent)` and exponentiating predictions often reduces RMSE because it prevents the model from overfitting to the extreme high end.

## 5. Baselines

- **Training median (₹22,000):** RMSE ~31,000. Floor.
- **Suburb median:** Predict using median rent for each suburb. RMSE ~25,000.
- **Linear on log_size + suburb one-hot:** RMSE ~20,000. Captures size + coarse location.
- **Locality target encoding + size + bedrooms:** Linear regression. RMSE ~16,000-18,000.

## 6. Models

**What to try, in order:**

1. **LightGBM** with target-encoded locality + suburb + all numeric features. Start: max_depth 7, n_estimators 1000, learning_rate 0.03, min_child_samples 20. Use early stopping on CV. Expect RMSE ~12,000-15,000.

2. **CatBoost.** Different from LightGBM in that it handles categorical features natively without explicit target encoding. Feed `locality`, `suburb`, `property_type` directly as categoricals. CatBoost uses ordered target encoding internally, which avoids leakage. Often competitive and simpler to set up.

3. **XGBoost with manual target encoding.** Similar to LightGBM but sometimes finds different optima. Worth running as a second model for ensembling.

4. **Stacking.** OOF predictions from LightGBM + CatBoost + Ridge. Ridge meta-learner. Usually worth 500-1000 RMSE improvement.

5. **Train on log(rent) then exponentiate.** This changes the loss landscape — the model focuses more on getting the ratio right rather than the absolute error. Can reduce RMSE significantly because it prevents the model from being dominated by high-rent outliers.

**What to skip:**
- KNN — spatial nearest-neighbors sounds intuitive but doesn't handle mixed feature types well, and the coordinate space alone isn't the right metric (two points 1km apart can have very different rents if there's a highway between them).
- Deep learning — not enough data and no image/text features to justify the complexity.

## 7. Error analysis

**Where will the model struggle?**

- **Luxury properties (rent > ₹100,000).** These are rare and price-volatile. A 4BHK villa in Chattarpur might be ₹100K or ₹200K depending on factors not in the data (furnishing quality, building age, floor). Expect high absolute errors here.
- **"Other" locality listings.** The model can't use locality-level knowledge for these ~8% of listings. It must rely on coordinates and distances alone. RMSE for "Other" listings will be worse than for named localities.
- **Outer NCR properties (the "Other" suburb category).** These areas from the broader NCR have different pricing dynamics than Delhi proper. If they're underrepresented in training, the model may not capture their specific patterns.
- **Small properties vs. 1BHK apartments.** Very small independent floors (< 500 sqft) may follow different pricing dynamics than larger properties. The model should capture non-linear size effects across property types.

Plot residuals on a map (lat/lon colored by error). If you see clusters of large errors, the model is missing a spatial pattern — maybe a new metro line that changed prices in certain areas, or a commercial development that made nearby residential areas more expensive.

## 8. Submission checklist

1. 3,549 data rows + header.
2. Two columns: `id` and `monthly_rent`.
3. IDs match test.csv.
4. All predictions are positive. No NaN, no strings.
5. Predictions should mostly fall in ₹5,000-₹200,000 range. If you see values below ₹1,000 or above ₹500,000, debug.
6. Run the scorer: `python3 score_submission.py --submission-path submission.csv --solution-path solution.csv`
7. Spot-check: a 3BHK apartment near a metro station in Delhi South should predict ₹35,000-₹60,000. A 1BHK in outer East Delhi should be ₹8,000-₹15,000.
