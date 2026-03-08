# Dataset Card — Credit Card Default Prediction

## Overview

Credit card customer records with demographics, credit limits, 6 months of repayment statuses, bill amounts, and payment amounts. Target is whether the customer defaults next month. Around 22% default rate, so there's notable class imbalance.

The tricky parts: EDUCATION and MARRIAGE have undocumented category codes that show up in the data, the PAY_X columns have non-obvious semantics (-2 vs -1 mean different things), and the financial columns are highly correlated with each other.

## Source

Based on the UCI Default of Credit Card Clients dataset:

Yeh, I. C. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.

Original: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

The data here is synthetically generated to match the statistical properties (distributions, correlations, default rate) of the original — no rows are copied verbatim.

## License
CC-BY-4.0

License text: https://creativecommons.org/licenses/by/4.0/
Original source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Changes from the original UCI dataset:
- Synthetic data (25k rows) matching the original's distributional properties, not a direct copy
- Added an `id` column
- Split into train (80%) and test (20%) with stratified random split on the target
- Target column removed from test.csv
- Renamed `PAY_1` to `PAY_0` to match what it actually represents (September status)
- Replaced periods with underscores in column names for CSV compatibility

## Features

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Row identifier |
| `LIMIT_BAL` | int | Credit limit (NTD) |
| `SEX` | int | 1=male, 2=female |
| `EDUCATION` | int | 1=grad school, 2=university, 3=high school, 4=others; 0/5/6 are undocumented |
| `MARRIAGE` | int | 1=married, 2=single, 3=other; 0=undocumented |
| `AGE` | int | Age in years |
| `PAY_0`, `PAY_2`–`PAY_6` | int | Repayment status per month (note: PAY_1 doesn't exist in this dataset — the columns jump from PAY_0 to PAY_2); -2=no usage, -1=paid in full, 0=min payment, 1-8=months of delay |
| `BILL_AMT1`–`BILL_AMT6` | int | Bill amounts in NTD (can be negative) |
| `PAY_AMT1`–`PAY_AMT6` | int | Payment amounts in NTD |
| `default_payment_next_month` | int | 1=defaulted, 0=didn't (train.csv only) |

## Splitting & Leakage

Random stratified split on `default_payment_next_month`, 80/20 train/test. No temporal ordering. All features represent info available before the target event (October default), so no leakage concerns. PAY_0 (September status) is the strongest single predictor — it's legitimate but models can over-rely on it.
