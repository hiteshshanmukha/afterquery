# Wine Quality Prediction

## Objective

You're given physicochemical lab measurements of Vinho Verde wines (red and white). Predict the expert quality rating for each wine in the test set. Ratings are integers from 3 (worst) to 8 (best).

Target column: `quality`. There are 6 classes (3, 4, 5, 6, 7, 8). Heads up — the distribution is quite skewed, with 5 and 6 making up the bulk of the data and 3/8 being rare.

---

## Inputs

| File | What's in it |
|------|-------------|
| `train.csv` | 5,197 wines with features and the `quality` label |
| `test.csv`  | 1,300 wines, features only — no `quality` column |

### Features

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Row ID |
| `fixed_acidity` | float | Tartaric acid, g/dm³ |
| `volatile_acidity` | float | Acetic acid, g/dm³ (high = vinegary) |
| `citric_acid` | float | Citric acid, g/dm³ |
| `residual_sugar` | float | Sugar remaining after fermentation, g/dm³ |
| `chlorides` | float | Salt, g/dm³ |
| `free_sulfur_dioxide` | int | Free SO₂, mg/dm³ |
| `total_sulfur_dioxide` | int | Total SO₂, mg/dm³ |
| `density` | float | g/cm³ |
| `pH` | float | Acidity (0–14) |
| `sulphates` | float | Potassium sulphate, g/dm³ |
| `alcohol` | float | ABV, % vol |
| `wine_type` | string | "red" or "white" |

---

## Output

Generate `submission.csv` — 1,300 rows plus the header.

Format:

```
id,quality
1,6
4,5
7,7
...
```

| Column | Type | Details |
|--------|------|--------|
| `id` | int | Must match the IDs in `test.csv` |
| `quality` | int | Your prediction — one of 3, 4, 5, 6, 7, 8 |

---

## Metric

Macro-averaged F1. Higher is better (0 to 1).

```
f1_score(y_true, y_pred, average='macro', labels=[3,4,5,6,7,8])
```

This weights every class equally regardless of how many samples it has. So even though classes 3 and 8 are rare, they count just as much as 5 and 6. If your model only ever predicts the common classes, the score will be bad.

To score locally:

```bash
python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
```

---

## Constraints

- Don't use `id` as a feature — it's just a row identifier.
- `wine_type` is fair game as a feature.
- No temporal ordering — train/test is a random stratified split (80/20).
- Predictions must be integers in {3, 4, 5, 6, 7, 8}. Anything else will fail validation.
