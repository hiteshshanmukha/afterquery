# New Delhi Residential Rental Listings Dataset

## Overview

Residential rental property listings from New Delhi and the broader National Capital Region (NCR), covering apartments, independent floors, independent houses, and villas. Each row represents a single property listing with its size, location coordinates, neighborhood, and distances to major city landmarks.

New Delhi's rental market spans an enormous range — from ₹5,000/month single-bedroom flats in outer suburbs to ₹300,000+ luxury villas in premium Delhi South neighborhoods. The market is highly spatial: proximity to metro stations, institutional hubs (AIIMS, JNU, government offices), and established commercial centers drives most of the price variation. Within a single suburb, prices can vary 3-5x depending on the specific locality and micro-neighborhood characteristics.

The dataset includes properties primarily from established Delhi neighborhoods and satellite areas. The geographic coordinates span the NCR, though the suburb labels are Delhi-centric (with outer NCR areas grouped under "Other").

Domain: real estate / urban planning / spatial economics.

## Source

OpenML Dataset ID 43837: New-Delhi-Rental-Listings.
Original data collected from online real estate listing platforms for the New Delhi metropolitan area.

https://www.openml.org/search?type=data&sort=runs&id=43837

## License

CC-BY-4.0

The upstream dataset (OpenML ID 43837) was contributed under the CC BY 4.0 license. The original data was scraped from publicly available online rental listing platforms for the New Delhi metropolitan area. Our derived version adds engineered features (distance computations, grouped locality, log-transforms) and a train/test split; the derivation is fully reproducible via `build_dataset.py`. Any use should cite the OpenML source.

## Features

| Column | Type | Notes |
|--------|------|-------|
| `id` | int | Sequential row ID |
| `size_sqft` | int | Property area in sqft. Median ~900 sqft. Heavy right tail with luxury properties at 5000+ sqft. |
| `property_type` | string | "Independent Floor" is most common (~63%), followed by "Apartment" (~32%), then "Independent House" (~5%) and "Villa" (<1%). |
| `bedrooms` | int | 1-10. Median is 2. Properties with 5+ bedrooms are almost always independent houses/villas. |
| `latitude` | float | ~28.0–29.0, centered around 28.61 (central Delhi). |
| `longitude` | float | ~76.5–77.6, centered around 77.17. |
| `locality` | string | Specific neighborhood. ~760 unique values. "Patel Nagar", "Paschim Vihar", "Chattarpur", "Malviya Nagar" etc. |
| `suburb` | string | Broader area. "Delhi South", "Delhi Central", "Dwarka", "West Delhi", "Delhi East", "North Delhi" etc. 12 unique values (including "Other" for outer NCR). |
| `metro_dist_km` | float | Distance to nearest Delhi Metro station. Median ~0.7 km. Metro proximity is a major price driver in Delhi. |
| `airport_dist_km` | float | Distance to IGI Airport. Ranges 2-50 km. |
| `aiims_dist_km` | float | Distance to AIIMS (All India Institute of Medical Sciences). Proxy for centrality. |
| `railway_station_dist_km` | float | Distance to New Delhi Railway Station. Another centrality proxy. |
| `suburb_locality_count` | int | Number of distinct localities in the same suburb. Higher = more developed/diverse area. |
| `center_dist_km` | float | Euclidean distance to Connaught Place (28.6315°N, 77.2167°E). The geographical and commercial heart of Delhi. |
| `log_size` | float | ln(size_sqft). Linearizes the size-price relationship. |
| `bedrooms_per_100sqft` | float | Bedrooms per 100 sqft. Higher values = smaller rooms. |
| `avg_landmark_dist` | float | Mean of metro/airport/AIIMS/railway station distances. Overall accessibility score. |
| `property_type_freq` | int | How common this property type is in the dataset. |
| `locality_grouped` | string | Locality with rare categories (< 10 listings) grouped as "Other". |
| `monthly_rent` | int | **Target.** Monthly rent in INR. Only in train.csv. |

## Splitting & Leakage

Random split: 80% train (14,193 rows), 20% test (3,549 rows), stratified on price quartile to ensure similar price distributions in both sets. The split is fully reproducible via `build_dataset.py` using `numpy.random.seed(42)`. Price quartile boundaries were computed on the full dataset before splitting, then 80% of each quartile was randomly assigned to train.

Key considerations:

- **Spatial autocorrelation.** Properties in the same locality have similar rents. With a random split, the model sees other listings from the same locality in training. This is realistic (you'd know neighborhood prices before predicting a new listing), but a model that memorizes locality-level averages could have inflated CV scores. Consider group-based CV by suburb for a harder but more honest evaluation.
- **`locality` and `suburb` are the most informative features** — they encode all the spatial/socioeconomic factors that drive rent. A model that ignores them and only uses numeric features will perform significantly worse.
- **`locality_grouped`** groups rare localities (< 10 listings) into "Other". About 8% of listings fall into "Other" (~1,400 rows), so while most listings retain their named locality, the model still needs coordinate-based fallbacks for these.
- **Derived features** (`center_dist_km`, `log_size`, `bedrooms_per_100sqft`, `avg_landmark_dist`, `suburb_locality_count`, `property_type_freq`) are deterministic functions of raw columns. No leakage.
- **No temporal information.** The dataset doesn't include listing dates, so there's no temporal leakage risk.
