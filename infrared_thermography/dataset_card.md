# Infrared Thermography Temperature Dataset

## Overview

Paired measurements from infrared thermal cameras and oral thermometers for 1,018 people. Each person sat in front of a thermal imaging camera which took an infrared picture of their face. Software then pulled temperature readings from several facial regions (inner canthi near the tear ducts, cheeks, forehead, orbital area). At the same time, their oral temperature was taken with a regular clinical thermometer.

The whole point of the dataset is fever screening -- can you figure out someone's actual body temperature just from what the thermal camera sees? This became a big deal during COVID when thermal scanners got installed in airports and hospitals everywhere. Turns out it's harder than it looks. Skin surface temperature depends on room conditions, distance from the camera, and even skin tone, so a raw thermal reading isn't a great proxy for core temp without some correction.

Domain: biomedical engineering / clinical thermometry. Practical use case is calibrating non-contact thermal screening systems.

## Source

Moran, M.P., Bui, A.T., Baran, N.M., Zhang, S.Y., et al. (2020). "Infrared Thermography Temperature Dataset." UCI Machine Learning Repository.

Original data: https://archive.ics.uci.edu/dataset/925/infrared+thermography+temperature+dataset

This came from a study at the FDA's Center for Devices and Radiological Health (CDRH), evaluating how well infrared systems measure body temperature.

## License

Public-Domain-US-Gov

Made by U.S. government employees as part of their official duties, so it's public domain in the US. No copyright restrictions.

**What I changed from the UCI version:**
- Dropped `aveOralF` (kept only `aveOralM`, renamed to `oral_temp` for clarity)
- Renamed the cryptic column codes (like `T_RC_Dry1`, `T_FHCC1`) to readable names (`right_cheek_dry`, `forehead_center`, etc.)
- Dropped 2 rows that had missing values
- Added an `id` column
- Did a stratified 80/20 train/test split on binned oral temperature
- Removed `oral_temp` from the test set

## Features

| Column | Type | Notes |
|--------|------|-------|
| `id` | int | Row identifier |
| `Gender` | string | "Male" or "Female" (roughly 55/45 split) |
| `Age` | string | Brackets: "21-30", "31-40", "41-50", "51-60", "61+". Skews young, most subjects are 21-40 |
| `Ethnicity` | string | Self-reported. Majority White, smaller groups for other categories |
| `ambient_temp` | float | Room temp in C, ranges 21-26. Shifts all the thermal readings |
| `Humidity` | float | Room humidity (%). Affects evaporation from skin |
| `Distance` | float | Camera distance in meters. Changes spatial resolution |
| `temp_offset` | float | Camera calibration offset. Important for absolute accuracy |
| `max_right_inner_canthus` | float | Max temp from right inner canthus ROI. Literature says this is the best spot for estimating core temp |
| `max_left_inner_canthus` | float | Same thing, left side |
| `avg_right_inner_canthus` | float | Mean temp, right canthus region |
| `avg_left_inner_canthus` | float | Mean temp, left canthus region |
| `right_cheek_temp` | float | Right cheek average. Can be thrown off by facial hair or makeup |
| `right_cheek_dry` | float | Right cheek dry-skin zone |
| `right_cheek_wet` | float | Right cheek near moisture. Usually reads a bit cooler from evaporation |
| `right_cheek_max` | float | Right cheek max |
| `left_cheek_temp` | float | Left cheek average |
| `left_cheek_dry` | float | Left cheek dry-skin zone |
| `left_cheek_wet` | float | Left cheek moisture zone |
| `left_cheek_max` | float | Left cheek max |
| `right_canthus_corrected` | float | Right canthus after environmental correction |
| `left_canthus_corrected` | float | Left canthus after environmental correction |
| `canthi_max` | float | Higher of the two canthus readings |
| `canthi_4_max` | float | Max across 4 canthus measurements |
| `forehead_center` | float | Forehead center. Clinically useful but hair and sweat mess with it |
| `forehead_right` | float | Forehead right quadrant |
| `forehead_left` | float | Forehead left quadrant |
| `forehead_bottom` | float | Forehead bottom |
| `forehead_top` | float | Forehead top, near the hairline. Tends to read cooler |
| `forehead_max` | float | Max forehead reading |
| `forehead_center_max` | float | Max of the center forehead ROI |
| `face_max_temp` | float | Highest temperature anywhere on the face. Strongest single predictor of oral temp |
| `orbital_right` | float | Right orbital region |
| `orbital_right_max` | float | Right orbital max |
| `oral_temp` | float | **Target.** Oral temperature in C, measured by clinical thermometer |

## Splitting & Leakage

80/20 stratified random split using quantile-binned oral temperature, so the temperature distributions in train and test are similar.

Things to be aware of on the leakage front:

Some features are essentially direct measurements of the thing you're predicting. `face_max_temp` and `canthi_max` are skin-surface readings of body heat, and `oral_temp` is a direct measurement of body heat, so they're measuring closely related quantities. That's by design (this is a sensor calibration problem), but it means a handful of features will dominate and the rest are mostly redundant.

`temp_offset` is a calibration value that was set during the measurement session. It's worth checking whether it was tuned against the oral reading, in which case it might carry target information indirectly.

Environmental features (`ambient_temp`, `Humidity`, `Distance`) shift all the thermal readings. If your model ignores these, it probably won't transfer well to a different room or camera setup.

On multicollinearity: the 30 thermal features are highly correlated with each other because they're all measuring the same face at the same moment. Pairwise correlations above 0.9 are normal here. Not leakage, but linear models will need heavy regularization, and tree-based feature importances will jump around between folds.
