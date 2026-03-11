# Tetouan City Power Consumption Dataset

## Overview

10-minute interval power consumption readings from three distribution zones in Tetouan, Morocco, paired with local weather data. Tetouan is a city of about 400,000 people in the Rif region along the Mediterranean coast. The climate is Mediterranean with hot dry summers and mild wet winters, which drives very seasonal electricity usage patterns -- AC load spikes in summer, heating bumps in winter.

The three zones are separate distribution networks feeding different parts of the city. Zone 1 is the largest (avg ~33,000 kW), Zone 2 is medium (~21,000 kW), and Zone 3 is the smallest (~18,000 kW). They're all correlated because they share the same weather and daily rhythm, but each zone has its own load profile based on the mix of residential, commercial, and industrial customers.

Weather features come from the local station and include temperature, humidity, wind speed, and two solar radiation metrics ("general diffuse flows" and "diffuse flows"). The solar columns are proxies for cloud cover and irradiance. Higher diffuse flow means more scattered sunlight, which correlates with both solar panel output and AC demand.

The dataset covers the full year 2017 at 10-minute resolution. That gives 52,416 samples total.

Domain: power systems / energy forecasting.

## Source

Salam, A. & El Hibaoui, A. (2018). "Comparison of Machine Learning Algorithms for the Power Consumption Prediction: Case Study of Tetouan city." 6th International Renewable and Sustainable Energy Conference (IRSEC). IEEE.

Original data: https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city

DOI: 10.24432/C5B034

Created at Abdelmalek Essaadi University, Tetouan, Morocco.

## License

CC-BY-4.0

Full license text: https://creativecommons.org/licenses/by/4.0/

Original dataset by Abdulwahed Salam and Abdelaaziz El Hibaoui, Abdelmalek Essaadi University. Published at UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city), DOI: 10.24432/C5B034.

Modifications from the original, per CC-BY-4.0 Section 3(b):

- Cleaned column names (removed spaces, standardized to lowercase with underscores).
- Extracted time features from the datetime column (hour, minute, day_of_week, month, day_of_year).
- Added a sequential `id` column.
- Split temporally: Jan-Oct 2017 for training, Nov-Dec 2017 for testing.
- Removed the `zone_1_power` target column from the test set.

## Features

| Column | Type | Notes |
|--------|------|-------|
| `id` | int | Sequential row ID, ordered by time |
| `datetime` | string | "YYYY-MM-DD HH:MM:SS", 10-min intervals |
| `temperature` | float | Air temperature, Celsius. Ranges from ~3 to ~39 across the year |
| `humidity` | float | Relative humidity (%). Higher in winter, lower in summer |
| `wind_speed` | float | Local wind speed. Generally low, occasional spikes |
| `general_diffuse_flows` | float | General diffuse solar radiation. Zero at night, peaks midday. Proxy for available solar energy |
| `diffuse_flows` | float | Diffuse component of solar radiation. Correlated with general_diffuse_flows but not identical |
| `zone_2_power` | float | Zone 2 power consumption (kW). Available at prediction time since zones are metered independently |
| `zone_3_power` | float | Zone 3 power consumption (kW). Same as above |
| `hour` | int | 0-23 |
| `minute` | int | 0, 10, 20, 30, 40, or 50 |
| `day_of_week` | int | 0=Monday, 6=Sunday |
| `month` | int | 1-12 |
| `day_of_year` | int | 1-365 |
| `zone_1_power` | float | **Target.** Zone 1 power consumption in kW. Only in train.csv |

## Splitting & Leakage

Temporal split: January through October 2017 for training (43,776 rows), November through December 2017 for testing (8,640 rows).

This is important. The test period has different characteristics from most of the training period:

- Nov-Dec has shorter days, so solar radiation patterns shift. The diffuse flow features have lower peaks and narrower non-zero windows.
- Temperature drops noticeably. Summer training data has temps up to 39C, but Nov-Dec mostly stays in the 10-20C range. If a model learned "high temp = high consumption" from summer AC load, it needs to also handle heating load in winter.
- Zone 1 average drops from ~33,000 kW (train) to ~29,000 kW (test). The distribution shift is real and a model that ignores the temporal trend will over-predict.
- Ramadan in 2017 was roughly May 26 - June 24. This affects daily consumption patterns (more nighttime cooking, shifted meal times). This won't directly affect the test period but the model might overweight Ramadan-period patterns if it doesn't account for it.

On leakage: `zone_2_power` and `zone_3_power` are from the same timestamp as the target. They're strongly correlated (same weather, same city, same time of day) but they're legitimately available at prediction time because the zones are metered independently and simultaneously. They're the strongest features and using them is fair.

The pre-extracted time features (`hour`, `minute`, `day_of_week`, `month`, `day_of_year`) are derived from `datetime`. No leakage there, just convenience.
