# Dry Bean Dataset

## Overview

This dataset contains morphological measurements of 13,611 dry bean grains belonging to seven different varieties, acquired via a computer vision system. A high-resolution camera captured images of the grains, and 16 shape-based features (area, perimeter, axis lengths, shape factors, etc.) were extracted from each grain image after segmentation. The goal is to classify each grain into its correct variety based on these geometric features alone.

The domain is agricultural science / food engineering. The seven bean varieties (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) are commercially important varieties in Turkey, and automated classification has practical value for sorting and quality control.

## Source

From the paper by Koklu & Ozkan:

> Koklu, M. and Ozkan, I.A., (2020). "Multiclass classification of dry beans using computer vision and machine learning techniques." *Computers and Electronics in Agriculture*, 174, 105507. DOI: [10.1016/j.compag.2020.105507](https://doi.org/10.1016/j.compag.2020.105507)

Upstream source on UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

## License

CC-BY-4.0

https://creativecommons.org/licenses/by/4.0/

Original data: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

**Changes made to the UCI version:**
- Added an `id` column (1-indexed integers) since the original doesn't have one.
- Shuffled the data and performed a stratified 80/20 split on `Class` to create train and test sets. The UCI file is not pre-split.
- Removed the `Class` column from test.csv (that's what you're predicting).

## Features

| Column | Type | Notes |
|--------|------|-------|
| `id` | int | Row identifier, added for this task |
| `Area` | int | Area of the bean zone in pixels — ranges from ~20,000 to ~250,000 depending on variety |
| `Perimeter` | float | Perimeter in pixels. BOMBAY beans tend to have the largest perimeters |
| `MajorAxisLength` | float | Length of the longest diameter line. Strongly correlated with Area |
| `MinorAxisLength` | float | Length of the shortest diameter line |
| `AspectRatio` | float | MajorAxisLength / MinorAxisLength. HOROZ beans are the most elongated (~2.0), BOMBAY the roundest (~1.1) |
| `Eccentricity` | float | Eccentricity of the ellipse fit. 0 = perfect circle, 1 = line |
| `ConvexArea` | int | Smallest convex polygon enclosing the bean. Very close to Area for most beans (Solidity > 0.98) |
| `EquivDiameter` | float | Diameter of a circle with the same area |
| `Extent` | float | Ratio of bean area to bounding rectangle area. Typically 0.65–0.85 |
| `Solidity` | float | Area / ConvexArea. How "filled in" the shape is |
| `Roundness` | float | 4π × Area / Perimeter². 1.0 = perfect circle |
| `Compactness` | float | EquivDiameter / MajorAxisLength |
| `ShapeFactor1` | float | MajorAxisLength / Area |
| `ShapeFactor2` | float | MinorAxisLength / Area |
| `ShapeFactor3` | float | Relates area to major axis — proportional to compactness squared |
| `ShapeFactor4` | float | Relates area to minor axis |
| `Class` | string | Bean variety: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA — **the target** |

## Splitting & Leakage

Stratified random split (80% train / 20% test), stratified on `Class`. There's no temporal component in this data, so no temporal leakage.

All features are geometric measurements extracted from grain images — none directly encode the variety label. However, there are strong correlations among the features because many are derived from the same underlying geometry:
- `Area`, `ConvexArea`, and `EquivDiameter` are near-redundant (all measure size).
- `AspectRatio`, `Eccentricity`, and `Compactness` all capture elongation.
- `ShapeFactor1`–`ShapeFactor4` are engineered ratios of the base measurements.

This multicollinearity is a known property of the dataset and may affect linear models but is not leakage. BOMBAY beans are easily separable by size alone (much larger than the rest), while SEKER/SIRA/DERMASON have significant overlap in feature space and are harder to distinguish.
