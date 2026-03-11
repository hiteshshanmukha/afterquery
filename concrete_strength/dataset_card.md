# Concrete Compressive Strength Dataset

## Overview

Lab measurements of concrete compressive strength alongside the mix design variables. Each row represents a single concrete sample — a specific recipe of cement, water, aggregates, and admixtures — tested at a particular curing age. The compressive strength was measured by physically crushing cylindrical specimens in a universal testing machine according to standard procedures.

This is the Yeh (1998) dataset that's widely used in civil engineering ML research. It's small (1,030 samples) but the underlying relationships are genuinely non-linear and interactive. The water-cement ratio is the dominant factor (Abrams' law, dating back to 1918), but supplementary cementitious materials like slag and fly ash create complex time-dependent interactions that simple models struggle with.

The strength values range from about 2 MPa (barely holds together) to about 83 MPa (high-performance concrete). Most structural concrete falls in the 20-50 MPa range. The distribution has a slight right skew — there are more ordinary mixes than high-performance ones.

Domain: civil engineering / materials science.

## Source

Yeh, I-C. (1998). "Modeling of strength of high-performance concrete using artificial neural networks." Cement and Concrete Research, 28(12), 1797-1808.

Original data: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength

DOI: 10.24432/C5PK67

Created by I-Cheng Yeh, Department of Information Management, Chung-Hua University, Hsin Chu, Taiwan.

## License

CC-BY-4.0

Full license text: https://creativecommons.org/licenses/by/4.0/

Original dataset by I-Cheng Yeh. Published at UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), DOI: 10.24432/C5PK67.

Modifications from the original, per CC-BY-4.0 Section 3(b):

- Cleaned column names (removed spaces, standardized to lowercase with underscores).
- Added derived features: `water_cement_ratio`, `total_binder`, `coarse_fine_ratio`, `log_age`.
- Added a sequential `id` column.
- Shuffled and split randomly: 80% train, 20% test.
- Removed the `compressive_strength` target column from the test set.

## Features

| Column | Type | Notes |
|--------|------|-------|
| `id` | int | Sequential row ID |
| `cement` | float | Portland cement, kg/m³. Range ~100-540. More cement generally = stronger, but diminishing returns. |
| `blast_furnace_slag` | float | Ground granulated blast furnace slag, kg/m³. 0 if not used. Develops strength slowly — improves 28-day+ strength. |
| `fly_ash` | float | Fly ash, kg/m³. 0 if not used. Pozzolanic — reacts with calcium hydroxide from cement hydration. Slow strength gain. |
| `water` | float | Water, kg/m³. Higher water = higher porosity = lower strength. The water/cement ratio is the key. |
| `superplasticizer` | float | Chemical admixture, kg/m³. Allows reduced water while maintaining workability. Small amounts matter a lot. |
| `coarse_aggregate` | float | Gravel/crushed stone, kg/m³. Bulk filler. Less direct effect on strength unless poorly graded. |
| `fine_aggregate` | float | Sand, kg/m³. Fills voids between coarse aggregate. |
| `age` | int | Days since casting when the sample was tested. 1 to 365. Strength continues developing for months/years. |
| `water_cement_ratio` | float | water / cement. **The** fundamental predictor. Abrams' law: strength ∝ 1 / (water/cement ratio). |
| `total_binder` | float | cement + slag + fly_ash. Total cementitious material. |
| `coarse_fine_ratio` | float | coarse_aggregate / fine_aggregate. Affects workability and packing density. |
| `log_age` | float | ln(1 + age). Linearizes the age-strength relationship somewhat. |
| `compressive_strength` | float | **Target.** Measured compressive strength in MPa. Only in train.csv. |

## Splitting & Leakage

Random split (80% train / 20% test) with fixed seed. No temporal component in this data — samples come from multiple studies over many years, and the age column refers to curing age of each individual sample, not calendar time.

Important nuances:

- The dataset contains **duplicate mix designs tested at different ages**. The same recipe of cement, water, etc. might appear at age 3, 7, 14, 28, 56, 90, and 365 days. With a random split, the same recipe can appear in both train and test at different ages. This isn't leakage — it's realistic (you'd know your recipe and test at different ages in practice) — but a model that memorizes recipe-level patterns could fool your CV estimate.
- There are also true duplicate rows (same recipe, same age, different measured strength). This is real experimental variation — repeating the exact same test gives slightly different results. About 25 rows have exact duplicates on features.
- `water_cement_ratio`, `total_binder`, `coarse_fine_ratio`, and `log_age` are deterministically derived from the raw columns. No leakage, just convenience features. A model could reconstruct them from the raw inputs.
