# Credit Card Default Prediction

## Task

Given 6 months of a customer's repayment history, bill amounts, payment amounts, and some demographics, predict the probability they'll default next month.

Output probabilities, not labels. Scored with AUC-ROC.

The default rate is around 22%, so there's class imbalance — predicting all zeros gets you 0.5 AUC (useless).

## Data

`train.csv` — 20k rows, includes the target column `default_payment_next_month`
`test.csv` — 5k rows, same columns minus the target

Columns:

- `id` — row identifier, don't use as a feature
- `LIMIT_BAL` — credit limit in NTD
- `SEX` — 1=male, 2=female
- `EDUCATION` — 1=grad school, 2=university, 3=high school, 4=others. Values 0, 5, 6 show up too but aren't documented — they're not errors, just unlabeled categories
- `MARRIAGE` — 1=married, 2=single, 3=other. 0 also appears (undocumented)
- `AGE` — years
- `PAY_0`, `PAY_2` through `PAY_6` — repayment status (PAY_0 = September / most recent, PAY_6 = April). There's no PAY_1 column — it's just how the original dataset was set up. Codes: -2 means card wasn't used, -1 means paid in full, 0 means minimum payment, 1+ means months of delay
- `BILL_AMT1` through `BILL_AMT6` — bill amounts (NTD), Sept through April. Can be negative (overpayment credit)
- `PAY_AMT1` through `PAY_AMT6` — payment amounts (NTD), Sept through April
- `default_payment_next_month` — 1 if defaulted, 0 otherwise. Only in train.

## Submission

Produce `submission.csv` with 5000 data rows + header:

```
id,default_prob
5,0.12
13,0.78
15,0.04
...
```

`id` must match test.csv exactly. `default_prob` is a float between 0 and 1.

Submit actual probabilities, not hard 0/1 labels.

## Scoring

AUC-ROC via `roc_auc_score(y_true, y_score)`. Range 0–1, higher is better. 0.5 = random.

```bash
python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
```

## Things to watch out for

- Don't use `id` as a feature
- BILL_AMT columns can go negative — that's real data, don't clip to zero
- PAY_X = -2 and PAY_X = -1 are different things (-2 = no usage, -1 = paid in full). Don't merge them
- The undocumented category codes in EDUCATION and MARRIAGE are valid, not noise
- Stratified random split, no temporal component
