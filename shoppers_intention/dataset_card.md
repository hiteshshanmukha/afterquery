# Online Shoppers Purchasing Intention Dataset

## Overview

Browsing session data from a real e-commerce website, capturing page navigation patterns, Google Analytics metrics, and visitor metadata. Each row is one session. The target variable (`revenue`) indicates whether the session ended with a transaction. Collected over a one-year period, roughly 12,330 sessions in total.

The dataset captures the full funnel: browsing behavior (page counts, durations), engagement quality (bounce rates, exit rates), monetization signals (page values from Google Analytics), and context (time of year, visitor type, device). The class imbalance is realistic — about 84.5% of sessions don't convert, which matches typical e-commerce conversion rates.

The most interesting aspect is the `page_values` feature from Google Analytics. It's calculated as the total revenue from transactions divided by the number of unique pageviews for each page — essentially, how "valuable" the pages visited in a session tend to be. Sessions where `page_values > 0` have a very high conversion rate. This feature is legitimately available at session time but it's so predictive that it can dominate the model and mask other useful patterns.

Domain: e-commerce / web analytics / conversion optimization.

## Source

Sakar, C.O., Polat, S.O., Katircioglu, M. et al. (2019). "Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks." Neural Computing and Applications, 31(10), 6893–6908.

Original data: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

DOI: 10.24432/C5F88Q

Created by C. Okan Sakar, Yildiz Technical University, and Yomi Kastro, Inveon Information Technologies Consultancy and Trade, Istanbul.

## License

CC-BY-4.0

Full license text: https://creativecommons.org/licenses/by/4.0/

Original dataset by C. Okan Sakar and Yomi Kastro. Published at UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset), DOI: 10.24432/C5F88Q.

Modifications from the original, per CC-BY-4.0 Section 3(b):

- Cleaned column names: lowercased, replaced CamelCase with snake_case.
- Converted boolean columns (`Revenue`, `Weekend`) to integer (0/1).
- Added derived features: `total_pages`, `total_duration`, `product_focus`, `avg_time_per_page`, `bounce_exit_gap`.
- Added a sequential `id` column.
- Shuffled and did a stratified 80/20 split on `revenue` to get train and test sets.
- Removed the `revenue` target column from the test set.

## Features

| Column | Type | Notes |
|--------|------|-------|
| `id` | int | Sequential row ID |
| `admin_pages` | int | Admin page views (0 to ~27). Most sessions = 0. |
| `admin_duration` | float | Seconds on admin pages. Heavy right skew — most are 0, some are 3000+. |
| `info_pages` | int | Info page views (0 to ~24). Most sessions = 0. |
| `info_duration` | float | Seconds on info pages. Same heavily skewed pattern. |
| `product_pages` | int | Product page views (0 to ~705). This is the main browsing activity. |
| `product_duration` | float | Seconds on product pages. Ranges from 0 to ~64,000. |
| `bounce_rate` | float | 0-0.2 range. Average bounce rate of visited pages. |
| `exit_rate` | float | 0-0.2 range. Average exit rate. Always ≥ bounce_rate. |
| `page_values` | float | 0 to ~361. Google Analytics page value. **Strongest predictor by far.** 0 for ~78% of sessions. |
| `special_day` | float | 0 to 1. Proximity to special shopping days. Most sessions = 0. |
| `month` | string | Feb, Mar, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec. (Jan, Apr missing.) |
| `operating_system` | int | 1-8. ID code for the operating system. |
| `browser` | int | 1-13. ID code for the browser. |
| `region` | int | 1-9. Geographic region. |
| `traffic_type` | int | 1-20. Source of the visit (search, direct, referral, etc.). |
| `visitor_type` | string | "Returning_Visitor" (~85%), "New_Visitor" (~14%), "Other" (~1%). |
| `weekend` | int | 0 (weekday) or 1 (weekend). About 23% weekend. |
| `total_pages` | int | Sum of admin + info + product pages. |
| `total_duration` | float | Sum of all duration columns. |
| `product_focus` | float | product_pages / (total_pages + 1). How product-focused the session is. |
| `avg_time_per_page` | float | total_duration / (total_pages + 1). |
| `bounce_exit_gap` | float | exit_rate - bounce_rate. Always ≥ 0. |
| `revenue` | int | **Target.** 1 = purchase, 0 = no purchase. Only in train.csv. |

## Splitting & Leakage

Stratified random split: 80% train (9,864 rows), 20% test (2,466 rows), stratified on `revenue` to preserve the 15.5% positive rate in both sets.

Important considerations:

- **`page_values` is borderline leaky.** It's derived from Google Analytics data that reflects how often pages lead to transactions. A page that many buyers visit will have a high page value. When `page_values > 0`, conversion probability is very high (~50%+). When `page_values = 0`, it's around 5%. The feature is legitimately available during a session (Google Analytics computes it in real-time from historical data), but it's so predictive that it can make everything else irrelevant. A model that only uses this feature already gets AUC > 0.90. The challenge is: can you do better by also using the browsing behavior?
- **No temporal split.** The data spans a year but we split randomly, not by time. This means there's no distribution shift between train and test. In a real deployment you'd want a temporal holdout, but for this task the focus is on feature interactions and handling class imbalance.
- The derived features (`total_pages`, `total_duration`, `product_focus`, `avg_time_per_page`, `bounce_exit_gap`) are deterministic functions of the raw columns. No leakage.
- `month` doesn't have January or April — those months are simply missing from the original data collection. Models shouldn't assume all 12 months are present.
