# Dry Bean Classification

## Objective

You're given morphological measurements of dry beans extracted from grain images. Predict the variety (type) of each bean in the test set.

Target column: `Class`. There are 7 classes: **BARBUNYA**, **BOMBAY**, **CALI**, **DERMASON**, **HOROZ**, **SEKER**, and **SIRA**. The distribution is imbalanced — DERMASON makes up ~26% of the data while BOMBAY is under 4%.

---

## Inputs

| File | What's in it |
|------|-------------|
| `train.csv` | 10,888 beans with features and the `Class` label |
| `test.csv`  | 2,723 beans, features only — no `Class` column |

### Features

All 16 features are numeric, derived from computer-vision analysis of high-resolution bean grain images.

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Row ID |
| `Area` | int | Area of the bean zone (pixels) |
| `Perimeter` | float | Perimeter of the bean (pixels) |
| `MajorAxisLength` | float | Length of the longest line through the bean |
| `MinorAxisLength` | float | Length of the shortest line through the bean |
| `AspectRatio` | float | MajorAxisLength / MinorAxisLength |
| `Eccentricity` | float | How elongated the bean is (0 = circle, 1 = line) |
| `ConvexArea` | int | Area of the smallest convex shape enclosing the bean |
| `EquivDiameter` | float | Diameter of a circle with the same area as the bean |
| `Extent` | float | Ratio of bean area to bounding box area |
| `Solidity` | float | Area / ConvexArea — how "filled in" the bean is |
| `Roundness` | float | 4π × Area / Perimeter² — how circular |
| `Compactness` | float | EquivDiameter / MajorAxisLength |
| `ShapeFactor1` | float | MajorAxisLength / Area |
| `ShapeFactor2` | float | MinorAxisLength / Area |
| `ShapeFactor3` | float | Area / (MajorAxisLength / 2)² × π |
| `ShapeFactor4` | float | Area / (MinorAxisLength / 2)² × π |

---

## Output

Generate `submission.csv` — 2,723 rows plus the header.

Format:

```
id,Class
3,SEKER
7,DERMASON
12,BOMBAY
...
```

| Column | Type | Details |
|--------|------|--------|
| `id` | int | Must match the IDs in `test.csv` |
| `Class` | string | Your prediction — one of BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA |

---

## Metric

**Macro-averaged F1**. Higher is better (0 to 1).

```
f1_score(y_true, y_pred, average='macro', labels=['BARBUNYA','BOMBAY','CALI','DERMASON','HOROZ','SEKER','SIRA'])
```

This weights every class equally regardless of sample count. Since BOMBAY only has ~4% of rows, your model still needs to classify it well — macro F1 penalizes heavily if you ignore minority classes.

To score locally:

```bash
python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
```

---

## Constraints

- Don't use `id` as a feature — it's just a row identifier.
- All features are fair game. They are all derived from grain image analysis; none leak the label.
- Class names are case-sensitive: `SEKER`, not `seker` or `Seker`.
