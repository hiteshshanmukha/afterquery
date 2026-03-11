# New Delhi Rental Price Prediction

## Objective

Predict the monthly rental price (in Indian Rupees, INR) for residential properties in New Delhi. You're given property characteristics (size, type, bedrooms), geographic coordinates, distances to key city landmarks (metro stations, airport, AIIMS hospital, New Delhi Railway Station), and neighborhood information.

New Delhi's rental market is extremely heterogeneous — a 2BHK flat in Delhi South can cost 5x what a similar flat costs in outer suburbs. Location, connectivity (metro access), and proximity to institutional hubs drive most of the price variation. The challenge is capturing these spatial effects alongside the property-level features.

## Inputs

| File | Contents |
|------|----------|
| `train.csv` | 14,193 rental listings with all features + `monthly_rent` |
| `test.csv`  | 3,549 listings, same features but no `monthly_rent` |

### Features

**Property characteristics:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Row ID |
| `size_sqft` | int | Property area in square feet. Range 100-10,000. |
| `property_type` | string | "Apartment", "Independent Floor", "Independent House", or "Villa" |
| `bedrooms` | int | Number of bedrooms (1-10) |

**Location:**

| Column | Type | Description |
|--------|------|-------------|
| `latitude` | float | Latitude (~28.0-29.0 range, within NCR) |
| `longitude` | float | Longitude (~76.5-77.6 range) |
| `locality` | string | Specific neighborhood (e.g., "Patel Nagar", "Paschim Vihar", "Malviya Nagar") |
| `suburb` | string | Broader area (e.g., "Delhi South", "Delhi Central", "Dwarka") |

**Distances to landmarks (km):**

| Column | Type | Description |
|--------|------|-------------|
| `metro_dist_km` | float | Distance to nearest Delhi Metro station |
| `airport_dist_km` | float | Distance to Indira Gandhi International Airport |
| `aiims_dist_km` | float | Distance to AIIMS (premier hospital in Central Delhi) |
| `railway_station_dist_km` | float | Distance to New Delhi Railway Station |

**Derived features:**

| Column | Type | Description |
|--------|------|-------------|
| `suburb_locality_count` | int | Number of distinct localities in the same suburb (proxy for area density) |
| `center_dist_km` | float | Euclidean distance to Connaught Place (city center) |
| `log_size` | float | ln(size_sqft) — linearizes the size-price relationship |
| `bedrooms_per_100sqft` | float | bedrooms / (size_sqft / 100) — space efficiency |
| `avg_landmark_dist` | float | Mean distance to all four landmarks |
| `property_type_freq` | int | Frequency count of the property type in the dataset |
| `locality_grouped` | string | Locality name, with rare localities (< 10 listings) grouped as "Other" (~8% of data) |

## Output

Produce a `submission.csv` with 3,549 rows (plus header).

```
id,monthly_rent
3,25000
7,42000
15,15500
...
```

| Column | Type | Details |
|--------|------|--------|
| `id` | int | Must match IDs in `test.csv` |
| `monthly_rent` | int/float | Predicted monthly rent in INR (must be positive) |

## Metric

RMSE (Root Mean Squared Error). Lower is better.

```
RMSE = sqrt(mean((y_true - y_pred)²))
```

Predicting the training median (~₹22,000) for every row gives RMSE around 31,000. A model that captures location and size effects should get below 20,000. A well-tuned model should push below 15,000. Anything under 12,000 is excellent.

Score locally:

```bash
python3 score_submission.py --submission-path submission.csv --solution-path solution.csv
```

## Constraints

- Don't use `id` as a feature.
- All other columns are fair game.
- Predictions must be positive.
- Predictions above ₹500,000/month are implausible for this data.
