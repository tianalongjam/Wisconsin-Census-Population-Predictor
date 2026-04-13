# Wisconsin Emerging Scholars

## Overview

This project uses regression models to predict census data for Wisconsin. Data is extracted from multiple sources, transformed into DataFrames, and used to train and evaluate machine learning models. Geographical data is visualized using `geopandas`.

---

## Data Sources

| File | Description |
|---|---|
| `counties.geojson` | Population and boundaries of each Wisconsin county |
| `tracts.shp` | Boundaries of each census tract (counties are subdivided into tracts) |
| `counties_tracts.db` | Details about housing units per tract (SQLite database) |
| `land.zip` | Land use data (farm, forest, developed, etc.) |

---

## Learning Objectives

- Extract, transform, and load data from multiple sources into a single DataFrame for ML training
- Analyze model performance using several metrics and feature impact on model score
- Plot geographical data using `geopandas`

---

## Key Libraries

- `geopandas`, `rasterio` — Geospatial data handling
- `sklearn` — `LinearRegression`, `train_test_split`, `cross_val_score`, `Pipeline`, `metrics`
- `numpy`, `pandas` — Data manipulation
- `matplotlib` — Plotting and visualization
- `sqlite3` — Database access

---

## Main File

- `main.ipynb` — Notebook containing all question answers

---

## Project Structure

### Part 1 — Predicting Population Using Area (Group)

- Load `counties.geojson` into a GeoDataFrame
- Add an `AREALAND` feature from `counties_tracts.db`
- Split data using `train_test_split(random_state=320, test_size=0.25)`
- Train a `LinearRegression` model on `AREALAND` to predict `POP100`

**Questions:** Q1–Q5

---

### Part 2 — Predicting Population Using Housing Units (Group)

- Add an `HU100` (housing units) feature by joining the `counties` and `tracts` tables in the SQLite DB
- Perform 5-fold cross validation using `cross_val_score`
- Extract the model's formula relating `POP100` and `HU100`
- Visualize the relationship with a scatter plot and fit line, annotating Waukesha County

**Questions:** Q6–Q12

---

### Part 3 — Land Use Features (Group)

- Work with raster data from `land.zip` using `rasterio`
- Visualize Dane County's land use map with a custom colormap
- Calculate what portion of Dane County consists of a specific land type (e.g., crops)
- Explore the relationship between land use cell counts and population

**Questions:** Q13–Q16

**Land use codes** reference: [NLCD 2019 Legend](https://www.mrlc.gov/data/legends/national-land-cover-database-2019-nlcd2019-legend)

---

### Part 1 and 2


- Load `tracts.shp` into a GeoDataFrame
- Add a feature column for every land use type (cell count from raster data)
- Split data using `random_state=320, test_size=0.20`
- Build and compare at least 2 regression models (varying features and/or sklearn `Pipeline` transformers)
- Evaluate using cross validation (mean and variance of scores)
- Final model must achieve **R² > 0.35** on the test set

**Questions:** Q17–Q19
