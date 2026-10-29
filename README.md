

# GIS-Based Micro-Watershed Prioritization and Water Conservation Potential Index Mapping Using WCHI in the Lower Vellar Sub-Basin, Tamil Nadu, India

## Project Title
GIS-Based Micro-Watershed Prioritization and Water Conservation Potential Index Mapping Using WCHI in the Lower Vellar Sub-Basin, Tamil Nadu, India.

## Problem Statement
Micro-watersheds in semi-arid and data-scarce landscapes require a transparent, evidence-based approach for prioritization and conservation planning. This project develops an academic machine-learning and GIS decision-support workflow to estimate watershed health, prioritize restoration actions, and recommend suitable water conservation structures using the Lower Vellar WCHI synthetic dataset.

## Objectives
- Load and validate the watershed dataset dynamically.
- Preprocess environmental and hydro-geomorphic variables without data leakage.
- Train and compare machine learning models for score prediction and priority classification.
- Compute a Water Conservation Potential Index (WCHI) using normalized component scores.
- Classify micro-watersheds into High, Medium, and Low priority.
- Recommend practical conservation structures.
- Visualize results in a web dashboard and GIS-style map.
- Support new micro-watershed prediction based on actual saved models.

## Dataset Parameters
Core variables include spatial, hydrological, meteorological, soil, vegetation, land use, drainage, and governance-related indicators such as:
- Micro_Watershed_ID
- Date, Year, Month
- Latitude, Longitude
- Area_km2, Perimeter_km, Basin_Length_km
- Elevation and terrain variables
- Rainfall, temperature, humidity, PET, AET
- Soil moisture, runoff, streamflow, groundwater depth
- Soil texture and infiltration variables
- NDVI, NDWI
- Agriculture, forest, built-up, waterbody, barren land
- Lineament density, geological permeability
- Existing ponds and check dams
- Rainfall_Score, Recharge_Score, Terrain_Score, Drainage_Score, Vegetation_Score, Groundwater_Score, Runoff_Score
- WCHI_Score, WCHI_Priority, Recommended_Conservation_Structure
- Dataset_Split

## System Architecture
The system follows a modular academic pipeline:
- Data input and validation
- Preprocessing and feature selection
- Train/Validation/Test split using Dataset_Split
- Model training and comparison
- Prediction outputs and feature importance
- WCHI calculation and priority classification
- Suitability and recommendation analysis
- Flask API
- React dashboard and GIS map

## ML Methodology
The project uses scikit-learn pipelines for robust preprocessing and model training. For each target, candidate models are trained on the dataset split provided by the original CSV. Validation performance is used to select the best model before test evaluation.

## WCHI Methodology
WCHI is calculated from normalized component indicators using configurable weights stored in backend/config.py. Weighted values are combined and normalized to a 0–1 scale. Priority classification uses the fixed thresholds: High at WCHI >= 0.70, Medium at 0.40 <= WCHI < 0.70, and Low below 0.40.

## Feature Selection
Feature selection was performed using domain-informed hydrological, climatic, geologic, and land-use variables. Leakage-prone target-derived variables were explicitly excluded from model input features to prevent data leakage.

## Data Leakage Prevention
The following columns were not used as standard inputs when predicting their own targets:
- Rainfall_Score
- Recharge_Score
- Terrain_Score
- Drainage_Score
- Vegetation_Score
- Groundwater_Score
- Runoff_Score
- WCHI_Score
- WCHI_Priority
- Recommended_Conservation_Structure

This prevents target leakage and preserves scientific validity.

## Train/Validation/Test Strategy
The CSV contains an existing Dataset_Split column. The project uses:
- Train: model training
- Validation: model comparison and selection
- Test: final evaluation
The original split is respected and no random reshuffling is performed when the provided dataset split exists.

## Model Comparison
Model comparison includes:
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor
- Random Forest Classifier
- Gradient Boosting Classifier

Validation performance is used to select the best algorithm, and final model performance is reported on the test split.

## Verified Model Inputs and Evaluation
The primary dataset is `data/Lower_Vellar_WCHI_Synthetic_Dataset(2).csv`: 480 monthly observations for 20 unique watershed IDs from January 2025 through December 2026. Its existing split is preserved exactly: 336 Train, 72 Validation, and 72 Test rows. Site-level dashboard, map, and ranking views aggregate the 24 observations for each ID; this does not create or alter training rows.

All 38 new-watershed form inputs are environmental, spatial, or time fields present in the dataset. The model targets and fitted input features are:

- `Rainfall_Score`: Month, Year, Latitude, Longitude, Tmax_C, Tmin_C, Mean_Temperature_C, Relative_Humidity_pct, PET_mm, AET_mm, Soil_Moisture_m3_m3, NDVI, NDWI, Mean_Slope_deg, Drainage_Density_km_per_km2, Runoff_mm, Groundwater_Depth_m, Available_Water_Capacity, Infiltration_Rate_mm_hr.
- `Recharge_Score`: Rainfall_mm, Soil_Moisture_m3_m3, Runoff_mm, Streamflow_m3_s, Groundwater_Depth_m, Sand_pct, Clay_pct, Silt_pct, Available_Water_Capacity, Infiltration_Rate_mm_hr, Lineament_Density_km_per_km2, Geological_Permeability_Index, NDVI, NDWI, Mean_Elevation_m, Mean_Slope_deg, Drainage_Density_km_per_km2, Stream_Frequency_per_km2, Soil_Organic_Carbon_pct, Agriculture_pct, Forest_pct.
- `WCHI_Score` and `WCHI_Priority`: the same 33 raw environmental inputs: Rainfall_mm, Tmax_C, Tmin_C, Mean_Temperature_C, Relative_Humidity_pct, PET_mm, AET_mm, Soil_Moisture_m3_m3, Runoff_mm, Streamflow_m3_s, Groundwater_Depth_m, Sand_pct, Clay_pct, Silt_pct, Soil_Organic_Carbon_pct, Soil_pH, Available_Water_Capacity, Infiltration_Rate_mm_hr, NDVI, NDWI, Agriculture_pct, Forest_pct, Builtup_pct, Waterbody_pct, Barren_Land_pct, Lineament_Density_km_per_km2, Geological_Permeability_Index, Population_Density_per_km2, Mean_Slope_deg, Max_Slope_deg, Drainage_Density_km_per_km2, Stream_Frequency_per_km2, Stream_Order.

WCHI, priority, recommendation, and derived component scores are not predictors for the refit WCHI or priority models. The refit reused the fixed split. Measured metrics are:

| Target | Algorithm | Validation | Test |
| --- | --- | --- | --- |
| WCHI_Score | Gradient Boosting Regressor | MAE 0.0115, RMSE 0.0140, R2 0.9743 | MAE 0.0241, RMSE 0.0266, R2 0.8857 |
| WCHI_Priority | Gradient Boosting Classifier | Accuracy 0.8889, F1 0.9056 | Accuracy 0.9306, F1 0.9308 |

Priority classifier confidence is an uncalibrated `predict_proba` score and is labeled as such. Structure suitability is rule-based, not a trained structure classifier or construction probability. Global explainability uses tree impurity feature importance; SHAP is not implemented.

## Decision-Support Features
The existing React/Vite and Flask application includes site rankings and filters, point-based thematic map layers, backend suitability with supporting/limiting factors, per-site WCHI explanations, global model importance, a WCHI what-if simulator, strict single/batch input validation, site CSV exports, and a browser-printable watershed report. APIs include `/api/rankings`, `/api/scenario`, `/api/model-explainability`, `/api/explanations/<id>`, `/api/future-projection`, `/api/analytics/monthly`, `/api/prediction-defaults`, and `/api/batch-predict` in addition to the existing endpoints.

## Data Limitations
- No district field or watershed boundary polygons/GeoJSON are included. GIS output is point-based and cannot show authoritative watershed outlines or district search.
- The dataset contains only 24 monthly periods across two years. The application deliberately reports future projections as unavailable and generates no future observations.
- Training metrics were not saved in the prior outputs. Candidate validation scores are generated by `python backend/train_models.py --evaluate-candidates`; no missing values are invented, and the Test split remains reserved for the active selected models.
- PDF output uses the browser's Print / Save as PDF workflow. SHAP and Excel export are not installed or claimed.

## Recommendation Methodology
The recommendation engine independently evaluates runoff, recharge, terrain, drainage, slope, infiltration, permeability, and land-cover conditions to estimate the suitability of conservation structures such as Check Dam, Farm Pond, Percolation Pond, Recharge Structure, Contour Bund, and Vegetative/Soil Conservation measures. It does not simply copy the dataset’s Recommended_Conservation_Structure column.

## GIS Methodology
The map uses the dataset's Latitude and Longitude for point-based visualization and aggregates the monthly records by watershed ID. Selectable point-value layers include mean elevation, mean slope, drainage score/density, WCHI score/priority, and conservation suitability. The terrain basemap is contextual only; its rendered tiles are not an analysis-ready DEM. The repository does not include a DEM raster, drainage-network geometry, or micro-watershed boundary polygons/GeoJSON, so the map does not claim to derive or display raster surfaces or official watershed outlines. Dataset-derived terrain attributes are from the supplied synthetic dataset and are not independently verified GIS measurements.

## Technology Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Flask
- React
- Vite
- Tailwind CSS
- React-Leaflet
- Recharts
- Matplotlib

## Folder Structure
- data/
- models/
- outputs/
- backend/
- frontend/
- requirements.txt
- README.md

## Installation
Use Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Training
```powershell
python backend/train_models.py
```

To refit only the WCHI and priority targets with leakage-safe raw environmental inputs while retaining the existing split:

```powershell
python backend/train_models.py --refit-leakage-safe
```

To compare candidate algorithms on the supplied training and validation split without replacing active model files:

```powershell
python backend/train_models.py --evaluate-candidates
```

## Backend Startup
```powershell
python backend/app.py
```
The API runs on http://localhost:5000.

## Frontend Startup
```powershell
cd frontend
npm install
npm run dev
```
The frontend runs via Vite, typically on http://localhost:5173.

## Testing
The project was validated by:
- Running the training pipeline with the actual CSV.
- Verifying the creation of all model files in models/.
- Verifying model metrics and output CSV files in outputs/.
- Starting the Flask backend and testing the health endpoint.
- Starting the Vite frontend and confirming the app loads.

## Expected Output
The final project generates:
- models/rainfall_model.pkl
- models/recharge_model.pkl
- models/wchi_model.pkl
- models/priority_model.pkl
- outputs/model_metrics.csv
- outputs/predictions.csv
- outputs/recommendations.csv
- outputs/micro_watershed_priority.csv

## Limitations
- This is a preliminary decision-support assessment rather than an engineering approval.
- The project depends on dataset quality and environmental assumptions.
- GIS boundary layers and field validation are not included unless provided by the user later.

## Future Enhancement
- Add official watershed boundary GeoJSON integration.
- Include geospatial interpolation and hotspot analysis.
- Expand to deeper hydrological or climate model coupling.
- Add uncertainty quantification and explainability modules.

## Academic Disclaimer
This system provides machine-learning-based decision support for watershed conservation planning. The recommended locations and structures are preliminary analytical recommendations and require field validation, hydrological investigation, land ownership verification, and engineering assessment before actual construction.

## Exact Windows PowerShell Commands
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python backend/train_models.py
python backend/app.py

cd frontend
npm install
npm run dev
```
