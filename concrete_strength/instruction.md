# Concrete Compressive Strength Prediction

## Objective

Predict the compressive strength (in megapascals, MPa) of concrete given its mix proportions and curing age. You're working with lab-measured data from real concrete samples tested according to standard procedures.

Compressive strength is the single most important property of structural concrete — it determines whether a building stands or collapses. It depends on a complex interaction of ingredients and time, not just how much cement you dump in.

## Inputs

| File | Contents |
|------|----------|
| `train.csv` | 818 concrete samples with all features + `compressive_strength` |
| `test.csv`  | 212 samples, same features but no `compressive_strength` |

### Features

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Row ID |
| `cement` | float | Cement content (kg/m³). Primary binder. Range ~100-540. |
| `blast_furnace_slag` | float | Blast furnace slag (kg/m³). Supplementary cite binder that strengthens slowly. 0 means none used. |
| `fly_ash` | float | Fly ash (kg/m³). Pozzolanic additive. 0 if not used. |
| `water` | float | Water content (kg/m³). More water = weaker concrete. |
| `superplasticizer` | float | Superplasticizer (kg/m³). Chemical admixture that improves workability without extra water. |
| `coarse_aggregate` | float | Coarse aggregate (kg/m³). Gravel or crushed stone. |
| `fine_aggregate` | float | Fine aggregate (kg/m³). Sand. |
| `age` | int | Age of sample when tested (days). Ranges from 1 to 365. |
| `water_cement_ratio` | float | water / cement — the most important predictor in concrete science. |
| `total_binder` | float | cement + blast_furnace_slag + fly_ash — total cite binder content. |
| `coarse_fine_ratio` | float | coarse_aggregate / fine_aggregate. |
| `log_age` | float | ln(1 + age). Strength develops roughly logarithmically with time. |

## Output

Produce a `submission.csv` with 212 rows (plus header).

```
id,compressive_strength
5,34.56
12,52.10
...
```

| Column | Type | Details |
|--------|------|--------|
| `id` | int | Must match IDs in `test.csv` |
| `compressive_strength` | float | Predicted strength in MPa (must be positive) |

## Metric

RMSE (root mean squared error). Lower is better. 0 is perfect.

```
RMSE = sqrt(mean((y_true - y_pred)²))
```

Predicting the training mean (~35.7 MPa) for every row gives RMSE around 16.8. A model that captures the main effects of cement, water, and age should get below 10. A well-tuned model should push below 6. Anything under 5 is excellent.

Score locally:

```bash
python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
```

## Constraints

- Don't use `id` as a feature.
- All other columns are fair game.
- Predictions must be positive (compressive strength can't be negative).
- Predictions above 120 MPa are implausible for this data.
