# Goa Solar Irradiation Forecasting — Golden Workflow

## 1  EDA & data understanding

```python
import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

print(train.shape, test.shape)
print(train.dtypes)
print(train.describe())
print(train["solar_irradiation"].describe())

# Check the temporal range
print("Train dates:", train["datetime"].min(), "to", train["datetime"].max())
print("Test dates:",  test["datetime"].min(),  "to", test["datetime"].max())

# Distribution of target
print("Fraction zero:", (train["solar_irradiation"] == 0).mean())
# ~49-51% zeros (nighttime)
```

Key observations:
- ~50% of rows have solar_irradiation == 0 (nighttime).
- Time series: 15-min intervals from Jul 2016 - Jul 2019.
- Temporal split: train ends Dec-2018, test starts Jan-2019.
- Test period includes dry season (Jan-May) which may have higher average irradiation than the monsoon-heavy training period.

## 2  Feature engineering

The dataset already includes several engineered features. Additional engineering:

```python
# Parse datetime for time-based features
train["datetime"] = pd.to_datetime(train["datetime"])
test["datetime"]  = pd.to_datetime(test["datetime"])

# Fractional hour (continuous)
for df in [train, test]:
    df["frac_hour"] = df["hour"] + df["minute"] / 60.0

# Solar position proxy: approximate solar elevation
# Goa latitude ~15.5°N
# Declination angle approximation
for df in [train, test]:
    dec = 23.45 * np.sin(np.radians(360 / 365 * (df["day_of_year"] - 81)))
    hour_angle = 15 * (df["frac_hour"] - 12)  # degrees from solar noon
    lat = 15.5
    sin_elev = (np.sin(np.radians(lat)) * np.sin(np.radians(dec)) +
                np.cos(np.radians(lat)) * np.cos(np.radians(dec)) *
                np.cos(np.radians(hour_angle)))
    df["solar_elevation"] = np.degrees(np.arcsin(np.clip(sin_elev, -1, 1)))
    df["solar_elevation_pos"] = df["solar_elevation"].clip(lower=0)

# Theoretical clear-sky irradiation (simplified)
# Solar constant ~1361 W/m², atmosphere attenuation factor ~0.7
for df in [train, test]:
    df["clearsky_proxy"] = 1361 * 0.7 * np.clip(
        np.sin(np.radians(df["solar_elevation"])), 0, None
    ) * 0.25  # 15-min to Wh conversion factor

# Dew point approximation (Magnus formula)
for df in [train, test]:
    a, b = 17.27, 237.7
    alpha = (a * df["temperature_c"]) / (b + df["temperature_c"]) + np.log(df["humidity"] / 100.0 + 1e-6)
    df["dew_point"] = (b * alpha) / (a - alpha)
    df["temp_dew_spread"] = df["temperature_c"] - df["dew_point"]
    # Larger spread → clearer skies → more solar
```

## 3  Model selection strategy

Given the nature of the task:
- **Gradient boosted trees** (LightGBM or XGBoost) are the go-to for tabular regression with mixed feature types.
- The zero-inflated distribution suggests two possible approaches:
  1. Single model trained on all data, relying on tree splits to handle zero/non-zero.
  2. Two-stage: first classify night vs day, then regress on daytime-only subset. The `solar_elevation` feature mostly resolves this.

LightGBM is recommended for speed and performance.

## 4  Training

```python
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

feature_cols = [c for c in train.columns
                if c not in ["id", "datetime", "solar_irradiation"]]

X_train = train[feature_cols]
y_train = train["solar_irradiation"]
X_test  = test[feature_cols]

# Time-series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
}

# Train with early stopping on last fold
for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_va, label=y_va)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    val_rmse = np.sqrt(((model.predict(X_va) - y_va) ** 2).mean())
    print(f"Fold {fold}: RMSE = {val_rmse:.4f}")
```

Expected CV RMSE: 15-25 depending on feature engineering depth.

## 5  Prediction & post-processing

```python
preds = model.predict(X_test)
preds = np.clip(preds, 0, None)  # Solar irradiation is non-negative

submission = pd.DataFrame({
    "id": test["id"],
    "solar_irradiation": preds,
})
submission.to_csv("submission.csv", index=False)
```

## 6  Key modelling insights

1. **Solar elevation** is the single most important feature — it determines the theoretical maximum irradiation and naturally zeros out nighttime predictions.
2. **Humidity and rainfall** are strong cloud-cover proxies that reduce irradiation.
3. **Temporal context** (rolling averages, pressure changes) captures weather persistence: a cloudy hour usually follows another cloudy hour.
4. **Monsoon flag** helps the model learn the dramatically different irradiation regime during Jun-Sep.
5. **Distribution shift**: the test set has proportionally more dry-season months, so models trained on monsoon-dominated data may underpredict. Cross-validation with temporal splits helps detect this.

## 7  Expected scores

| Approach | RMSE |
|----------|------|
| Predict training mean everywhere | ~76 |
| Zero at night + mean during day | ~45 |
| LightGBM with basic features | ~20-25 |
| LightGBM with solar elevation + rolling | ~15-20 |
| Tuned ensemble (LGB + XGB) | ~12-16 |
