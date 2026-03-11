# How I'd work through this

## 1. Understand the time structure

This isn't an i.i.d. tabular problem. It's 10-minute interval power data with strong daily and weekly cycles, a seasonal trend, and a temporal train/test split. The first thing to do is plot the time series.

What I'd look at:
- Plot zone_1_power across the full training period. You'll see clear daily oscillations (low at night, peaks during the day) and a seasonal envelope (higher in summer from AC, lower in spring/fall, slight uptick in winter from heating).
- Plot a few representative weeks side by side -- one from January, one from July, one from October. The shape of the daily curve changes with the season. Summer has a broader daytime peak, winter has a sharper one.
- Overlay zone_1_power, zone_2_power, and zone_3_power. They move together but with different amplitudes. Zone 2 and Zone 3 are going to be the strongest predictors just because they're measuring the same kind of thing at the same time.
- Check for outliers or gaps: any 10-minute intervals where zone_1_power drops to near zero or spikes unreasonably? The dataset claims no missing values but partial outages or meter glitches could still show up as valid-looking numbers.
- Plot temperature vs. zone_1_power. Expect a U-shaped relationship: low temps drive heating, high temps drive AC, moderate temps are the trough. This U-shape is important because a linear model won't capture it without help.

## 2. The distribution shift problem

Nov-Dec 2017 looks different from the training average. Zone 1 mean drops from 33K to 29K. Temperatures are lower, days are shorter. A model fit to Jan-Oct will be biased toward summer patterns because summer months dominate the training set (more months, higher consumption, bigger variance).

How I'd handle this:
- Weight recent training data more heavily. October is more informative about November than January is. Consider giving exponentially decaying weights, or just training on the last 3-4 months only and comparing to using the whole year.
- Include `month` and `day_of_year` as features so the model can learn the seasonal trend. Cyclical encoding (sin/cos) for monthly and hourly periodicity.
- Check whether the Oct-to-Nov transition in zone_2_power and zone_3_power is smooth. If those zones also drop in Nov-Dec, the model can piggyback on that signal.

## 3. Validation strategy

Can't use random K-fold on time series. Two options:

1. **Rolling origin validation.** Train on Jan-Aug, validate on Sep. Then train on Jan-Sep, validate on Oct. Average the two SMAPE scores. This simulates the test scenario (predicting the next unseen month) but only gives 2 folds.

2. **Sliding window.** Train on months 1-7, validate on 8. Then 2-8, validate on 9. Then 3-9, validate on 10. Three folds, each predicting one month ahead. Less data in the earliest fold but gives a better feel for stability.

I'd go with option 2. Three folds is better than two for estimating variance. Track SMAPE on each fold separately. If the model does well on month 8 but badly on month 10, that's a sign it can't handle the fall transition.

For quick iteration, just hold out October as a single validation set. It's the last training month and most similar to the test period.

## 4. Feature engineering

The pre-extracted time features (hour, minute, day_of_week, month, day_of_year) are a start, but there's more signal to pull out.

**Cyclical encoding:** Hour and day_of_week have circular structure (hour 23 is close to hour 0). Encode them as sin/cos pairs:
- `hour_sin = sin(2*pi*hour/24)`, `hour_cos = cos(2*pi*hour/24)`
- `dow_sin = sin(2*pi*day_of_week/7)`, `dow_cos = cos(2*pi*day_of_week/7)`
- `month_sin = sin(2*pi*month/12)`, `month_cos = cos(2*pi*month/12)`

**Temperature interactions:**
- `temp_squared`: captures the U-shaped relationship between temperature and power.
- `temp_x_hour`: the effect of temperature on consumption depends on time of day. A hot afternoon drives AC way more than a hot night.
- `temp_x_humidity`: humidity amplifies the "feels like" temperature, which drives AC usage.

**Solar features:**
- `is_daytime`: 1 if general_diffuse_flows > 0, else 0. Simplifies the day/night split.
- `solar_ratio = diffuse_flows / (general_diffuse_flows + 0.01)`: fraction of diffuse vs. total solar. Higher ratio means cloudier.

**Cross-zone features:**
- `zone_23_avg = (zone_2_power + zone_3_power) / 2`: aggregate other-zone signal.
- `zone_23_ratio = zone_2_power / (zone_3_power + 1)`: relative loading between the two other zones.

**Lag features (careful here):**
For the training set you can compute lags: zone_1_power 24 hours ago, 1 week ago. But for the test set, you don't have zone_1_power at all. So lags of the target are out. Lags of zone_2_power and zone_3_power are available though, and they're a good proxy for yesterday's demand pattern in the city.

**Weekend/holiday flag:**
- `is_weekend = day_of_week >= 5`. Power patterns differ noticeably on weekends.
- Morocco's weekends are Saturday-Sunday. Friday is a normal workday (unlike some other MENA countries).
- National holidays in Nov-Dec 2017: Independence Day (Nov 18), Green March (Nov 6). These would look like weekend patterns.

## 5. Baselines

- **Training mean everywhere (33K):** SMAPE around 18-19. That's the floor.
- **Month-specific mean:** use the October monthly average for Nov-Dec predictions. Should be more accurate since October is closer in season.
- **Hour-of-day mean from training set:** compute the average zone_1_power for each hour across training, then just map test timestamps to the corresponding hour mean. This captures the daily cycle. Probably SMAPE around 12-14.
- **Zone 2/3 linear regression:** fit `zone_1_power = a * zone_2_power + b * zone_3_power + c`. Since the zones are highly correlated, this alone should get SMAPE well under 10.

## 6. Models

**What I'd try, roughly in order:**

1. **Linear regression with engineered features.** All the features from section 4 plus the raw weather columns. Ridge regularization. Fast to run, good interpretability, decent baseline. Expect SMAPE around 5-8.

2. **LightGBM.** Handles non-linear relationships (the temp U-shape, time-of-day interactions) without explicit feature engineering. Start with max_depth 6, learning rate 0.05, 500 trees, subsampling 0.8. Use the time-based validation for early stopping. Should push SMAPE below 5.

3. **XGBoost with careful hyperparameter search.** Similar to LightGBM but sometimes handles small-to-medium data a bit differently. Try both and pick whichever validates better.

4. **Stacking.** Out-of-fold predictions from Ridge + LightGBM + maybe SVR, combined with a simple meta-learner. Worth 0.5-1.0 SMAPE improvement usually.

**What I'd skip:**
- Deep learning / LSTM / Transformers. For 10-minute univariate power data with weather covariates, gradient boosting with good features will match or beat LSTM. The dataset is also small enough that LSTMs would overfit without heavy regularization.
- ARIMA/SARIMA as the primary model. 52K 10-minute intervals means the seasonal period is 144 (one day). SARIMA with period=144 is computationally painful and usually underperforms ML approaches that can use covariates directly.
- Prophet. Fine for daily/weekly data but 10-minute resolution with covariates isn't really its sweet spot.

## 7. Error analysis

Where will the model struggle?

- **Transition days.** The first few days of November, right after the training period ends. The model hasn't seen November before, so it's extrapolating. Check whether errors are worse in early November vs. late December.
- **Unusual weather.** If Nov-Dec has a cold snap or warm spell that's outside the range seen in Oct training data, the model's temperature response might be off.
- **Holiday patterns.** Nov 6 (Green March) and Nov 18 (Independence Day) will have weekend-like consumption patterns. If the model doesn't know about holidays, it'll over-predict those days.
- **Night vs. day errors.** SMAPE can blow up at night when both actual and predicted values are relatively small. A 1000 kW error at 2 AM (actual: 20,000 kW) gives a higher SMAPE contribution than a 1000 kW error at 2 PM (actual: 35,000 kW).
- **Weekday vs. weekend.** Different load profiles. Check if errors are worse on one vs. the other.

Plot residuals over time. If there's a systematic drift (errors getting worse as you go deeper into the test period), the model is missing a trend.

## 8. Submission checklist

1. 8,640 data rows + header.
2. Two columns: `id` and `zone_1_power`.
3. IDs match test.csv.
4. All predictions are positive floats. No NaN, no strings, no negatives.
5. Predictions should mostly fall in the 15,000-45,000 range for Nov-Dec. If you see values below 10,000 or above 55,000, something's off.
6. Run the scorer: `python3 score_submission.py --submission-path submission.csv --solution-path solution.csv`
7. Spot-check: nighttime predictions (~2 AM) should be around 18,000-22,000. Midday peaks should be around 30,000-35,000. If the model predicts 40,000 at 3 AM, debug it.
