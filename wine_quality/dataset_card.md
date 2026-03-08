# Wine Quality Dataset

## Overview

This is the well-known Vinho Verde wine dataset — lab measurements from Portuguese red and white wines, plus quality ratings from expert panels. At least 3 tasters scored each wine and the median was taken. You're trying to predict that quality score (integer, 3 through 8) from the chemistry alone.

Most wines end up rated 5 or 6. The tails (3 and 8) are pretty sparse, under 4% each. I merged red and white into one file with a `wine_type` flag since the two behave quite differently chemically.

## Source

From the paper by Cortez et al.:

> Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4), 547–553.

Upstream source on UCI:
https://archive.ics.uci.edu/dataset/186/wine+quality  
DOI: https://doi.org/10.24432/C56S3T

## License
CC-BY-4.0

https://creativecommons.org/licenses/by/4.0/

Original data lives at: https://archive.ics.uci.edu/dataset/186/wine+quality

**Changes I made to the UCI version:**
- Added an `id` column (1-indexed integers) since the original doesn't have one.
- The originals ship as two separate CSVs (`winequality-red.csv` and `winequality-white.csv`). I combined them and tacked on a `wine_type` column so you know which is which.
- Shuffled and did a stratified 80/20 split on `quality` to get train and test sets. The UCI files aren't pre-split.
- Removed the `quality` column from test.csv (that's what you're predicting).

## Features

| Column | Type | Notes |
|--------|------|-------|
| `id` | int | Row identifier, added for this task |
| `fixed_acidity` | float | Tartaric acid, g/dm³ |
| `volatile_acidity` | float | Acetic acid, g/dm³ — too high and it tastes like vinegar |
| `citric_acid` | float | g/dm³, adds a fresh flavour |
| `residual_sugar` | float | Sugar left over after fermentation, g/dm³ |
| `chlorides` | float | Sodium chloride, g/dm³ |
| `free_sulfur_dioxide` | int | Free SO₂ (mg/dm³) — the part that actually fights microbes |
| `total_sulfur_dioxide` | int | Total SO₂, mg/dm³ — hard to taste below ~50 |
| `density` | float | g/cm³, driven mainly by alcohol and sugar |
| `pH` | float | 0–14 acidity scale |
| `sulphates` | float | Potassium sulphate, g/dm³ — feeds into SO₂ levels |
| `alcohol` | float | ABV, % vol |
| `wine_type` | string | "red" (1,599 rows) or "white" (4,898 rows) |
| `quality` | int | Expert rating: 3, 4, 5, 6, 7, or 8 — **the target** |

## Splitting & Leakage

Stratified random split (80% train / 20% test), stratified on `quality`. There's no time component in this data so no temporal leakage to worry about. All features are chemical measurements taken independently of the tasting, so no obvious leakage paths.

One thing to keep in mind: `density` is physically related to `alcohol` and `residual_sugar` (basic physics), so those three are correlated. Not leakage, but it'll show up as multicollinearity if you look.
