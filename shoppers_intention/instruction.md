# Online Shoppers Purchase Intention Prediction

## Objective

Predict whether a visitor's browsing session on an e-commerce site will end in a purchase. Given page-view counts, time spent, bounce/exit rates, Google Analytics metrics, visitor metadata, and session timing, classify each session as `revenue = 1` (purchase made) or `revenue = 0` (no purchase).

Only about 15.5% of sessions result in a purchase. A model that always predicts "no purchase" gets 84.5% accuracy but is completely useless. The metric (AUC-ROC) specifically rewards the ability to *rank* sessions by purchase likelihood, not just classify.

## Inputs

| File | Contents |
|------|----------|
| `train.csv` | 9,864 browsing sessions with all features + `revenue` label |
| `test.csv`  | 2,466 sessions, same features but no `revenue` column |

### Features

**Page engagement metrics:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Row ID |
| `admin_pages` | int | Number of administrative pages visited (account, addresses, etc.) |
| `admin_duration` | float | Total seconds spent on admin pages |
| `info_pages` | int | Number of informational pages visited (about, FAQ, etc.) |
| `info_duration` | float | Total seconds spent on informational pages |
| `product_pages` | int | Number of product-related pages visited |
| `product_duration` | float | Total seconds spent on product pages |

**Google Analytics metrics:**

| Column | Type | Description |
|--------|------|-------------|
| `bounce_rate` | float | Average bounce rate of pages visited. A "bounce" = visitor entered and left from the same page without any interaction. |
| `exit_rate` | float | Average exit rate of pages visited. The percentage of pageviews that were the last in the session. |
| `page_values` | float | Average page value of pages visited. Google Analytics metric: contribution of a page to a transaction (0 = page never preceded a transaction). **This is the single strongest feature.** |

**Session context:**

| Column | Type | Description |
|--------|------|-------------|
| `special_day` | float | Closeness to a special day (Valentine's, Mother's Day, etc.). 0-1 scale where 1 = on the day. |
| `month` | string | Month of the session (Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec). Note: Jan and Apr are missing from the data. |
| `operating_system` | int | OS ID (1-8) |
| `browser` | int | Browser ID (1-13) |
| `region` | int | Geographic region ID (1-9) |
| `traffic_type` | int | Traffic source type (1-20, e.g. direct, referral, search) |
| `visitor_type` | string | "Returning_Visitor", "New_Visitor", or "Other" |
| `weekend` | int | 1 if session was on a weekend, 0 otherwise |

**Derived features:**

| Column | Type | Description |
|--------|------|-------------|
| `total_pages` | int | admin_pages + info_pages + product_pages |
| `total_duration` | float | admin_duration + info_duration + product_duration |
| `product_focus` | float | product_pages / (total_pages + 1) — how product-focused the session was |
| `avg_time_per_page` | float | total_duration / (total_pages + 1) |
| `bounce_exit_gap` | float | exit_rate - bounce_rate — how much more likely to exit than bounce |

## Output

Produce a `submission.csv` with 2,466 rows (plus header).

```
id,revenue
3,0
7,1
15,0
...
```

| Column | Type | Details |
|--------|------|--------|
| `id` | int | Must match IDs in `test.csv` |
| `revenue` | int | 0 (no purchase) or 1 (purchase) |

## Metric

AUC-ROC (Area Under the Receiver Operating Characteristic Curve). Higher is better (0.5 is random, 1.0 is perfect).

```
roc_auc_score(y_true, y_pred)
```

Note: AUC-ROC works on hard labels here (0/1), not probabilities. A model that perfectly separates purchasers from non-purchasers scores 1.0. Random guessing scores ~0.5.

Predicting all 0s (the sample submission) gives AUC = 0.5. A decent model should exceed 0.85. Getting above 0.90 requires good feature engineering and handling of the class imbalance.

Score locally:

```bash
python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
```

## Constraints

- Don't use `id` as a feature.
- All other columns are fair game.
- Predictions must be exactly 0 or 1 (integers).
