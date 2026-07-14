# Wisconsin Census Population Predictor

## Overview

This project predicts U.S. Census population counts for Wisconsin using regression models. Data is extracted from four independent sources — a GeoJSON, a shapefile, a SQLite database, and a satellite land-cover raster — transformed into a single tabular DataFrame, and used to train and evaluate `scikit-learn` regression models at both the county and census-tract level. Geospatial data is visualized with `geopandas` and `rasterio`.

---

## Data Sources

| File | Description |
|---|---|
| `counties.geojson` | Population (`POP100`) and boundaries of each of Wisconsin's 72 counties |
| `tracts.shp` (+ `.dbf` / `.shx` / `.prj` / `.cpg`) | Boundaries of each census tract (counties are subdivided into tracts) |
| `counties_tracts.db` | SQLite database with `AREALAND` (land area) per county and `HU100` (housing units) per tract |
| `land.zip` | Zipped NLCD 2019 land-cover raster (`wi.tif`) — 20 land-use classes (water, forest, developed, crops, wetlands, etc.) |

---

## Approach & Results

Each stage adds a richer feature set and asks whether the added complexity is actually earning its keep.

### 1. Population from land area (county level)

Added `AREALAND` from the SQLite database to the counties GeoDataFrame, split 75/25 (`random_state=320`), and fit a `LinearRegression`.

- **R² ≈ 0.02** — land area on its own explains almost none of the variation in population. Large rural counties and small dense ones don't follow a consistent area-to-population relationship.

### 2. Population from housing units (county level)

Joined the `counties` and `tracts` tables in SQL, summing `HU100` per county, and evaluated with 5-fold `cross_val_score` instead of a single split.

- **R² ≈ 0.97 average (5-fold CV)**, **R² ≈ 0.996 on the held-out test set**
- Fitted relationship: `POP100 ≈ 2.39 * HU100 − 7910.46`, roughly 2.4 residents per housing unit
- Housing units are a near-tautological predictor of population — a strong forecasting signal, but one that mostly restates the target rather than explaining it

### 3. Land use as a feature (county level)

Used `rasterio` to mask the NLCD raster to each county's polygon (after reprojecting to a shared CRS) and count pixels per land-use class.

- Dane County is **~47% cropland**
- Explored the relationship between individual land-cover types (e.g., evergreen forest) and county population

### 4. Multi-feature regression at the census-tract level (individual part)

Switched to the finer-grained `tracts.shp`, engineered 20 feature columns (one per NLCD land-use class) via raster masking, and split 80/20.

- Compared a plain `LinearRegression` against a `Pipeline([StandardScaler, LinearRegression])` using cross-validation mean and standard deviation
- Both scored effectively identically (~0.515 avg R², since OLS is invariant to feature scaling) — the scaled pipeline was recommended for more interpretable, comparable coefficients
- Final model achieved **~0.39 explained variance** on the held-out test set, clearing the 0.35 target

---

## Questions Answered

### Part 1 — Population from Area

| # | Question | Answer |
|---|---|---|
| Q1 | How many counties are in Wisconsin? | 72 |
| Q2 | What is the population of each county in WI? | Choropleth map colored by `POP100` (`img/population_WI.png`) |
| Q3 | What are the counties in the test dataset? | List of 18 counties (25% test split, `random_state=320`) |
| Q4 | How much variance in `POP100` can a `LinearRegression` model explain based only on `AREALAND`? | R² ≈ 0.022 |
| Q5 | What is the predicted population of a county with 300 square miles of area? | ≈ 89,016 |

### Part 2 — Population from Housing Units

| # | Question | Answer |
|---|---|---|
| Q6 | What are the counties in the test dataset? | Same 18 counties as Q3 |
| Q7 | What are the `HU100` values for the counties in the test dataset? | Dict of housing-unit counts per county |
| Q8 | How much variance in `POP100` can a `LinearRegression` model explain based only on `HU100`? | R² ≈ 0.965 (average of 5-fold CV) |
| Q9 | What is the standard deviation of the cross-validation scores from Q8? | ≈ 0.037 |
| Q10 | What is the formula relating `POP100` and `HU100`? | `POP100 = 2.39*HU100 + -7910.46` |
| Q11 | What is the r-squared score of the fitted model (from Q10) on the test set? | R² ≈ 0.996 |
| Q12 | What is the relationship between `HU100` and `POP100`, visually? | Scatter plot with fit line, Waukesha County annotated (`img/bestfitline_POP100vsH100.png`) |

### Part 3 — Land Use Features

| # | Question | Answer |
|---|---|---|
| Q13 | How many numbers in matrix `A` are between 1 and 4 (inclusive)? | 13 |
| Q14 | How does Dane County look? | Land-cover raster plot with custom NLCD colormap (`img/DaneCounty_colormap.png`) |
| Q15 | What portion of Dane County is "Crops"? | ≈ 46.7% |
| Q16 | What is the relationship between `POP100` and a chosen land type? | Scatter plot: population vs. evergreen-forest cell count (`img/scatterplot.png`) |

### Individual Part — Tract-Level Multi-Feature Model

| # | Question | Answer |
|---|---|---|
| Q17 | What features does the model rely on most? | Bar plot of feature coefficients (`img/barplot_featureimportance.png`) |
| Q18 | What is the mean and standard deviation of the cross-validation scores from both models? | `{'model1-avg': 0.5155, 'model1-std': 0.0240, 'model2-avg': 0.5155, 'model2-std': 0.0240}` |
| Q19 | How does the recommended model score against the test dataset? | Explained variance ≈ 0.389 |

---

## Key Libraries

- `geopandas`, `rasterio` — geospatial vector and raster data handling
- `scikit-learn` — `LinearRegression`, `train_test_split`, `cross_val_score`, `Pipeline`, `StandardScaler`, `metrics`
- `numpy`, `pandas` — data manipulation
- `matplotlib` — plotting and visualization
- `sqlite3` — relational database access

---

## Repository Structure

```
main.ipynb              # Full analysis notebook
main.csv                # Reference output values
counties.geojson        # County boundaries + population
tracts.shp (+ sidecars) # Census tract boundaries + population
counties_tracts.db       # SQLite DB: area & housing unit data
land.zip                 # NLCD land-cover raster
img/                     # Saved figures (choropleth map, land-cover map,
                          # fit line, feature-importance bar chart)
```

---

## Setup & Usage

1. Install dependencies: `geopandas`, `rasterio`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`
2. Ensure `counties.geojson`, `tracts.shp` (with its sidecar files), `counties_tracts.db`, and `land.zip` are present in the project root
3. Open and run `main.ipynb` top to bottom

---

## Methodology Summary

| Feature(s) | Granularity | Evaluation | Result |
|---|---|---|---|
| `AREALAND` | County | Train/test split | R² ≈ 0.02 |
| `HU100` (housing units) | County | 5-fold CV / test | R² ≈ 0.97 (CV) / 0.996 (test) |
| Individual land-use types | County | Scatter/exploratory | Weak individually |
| All 20 land-use types | Census tract | 5-fold CV / test | R² ≈ 0.52 (CV) / 0.39 explained variance (test) |