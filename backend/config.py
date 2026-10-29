from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'
DATASET_DIR = ROOT_DIR / 'dataset'
MODELS_DIR = ROOT_DIR / 'models'
OUTPUTS_DIR = ROOT_DIR / 'outputs'

DATASET_CANDIDATES = [
    DATA_DIR / 'Lower_Vellar_WCHI_Synthetic_Dataset(2).csv',
    DATASET_DIR / 'Lower_Vellar_WCHI_Synthetic_Dataset.csv',
    DATASET_DIR / 'Lower_Vellar_WCHI_Synthetic_Dataset(2).csv',
]

REQUIRED_COLUMNS = [
    'Micro_Watershed_ID', 'Latitude', 'Longitude', 'Dataset_Split',
    'Rainfall_mm', 'Mean_Temperature_C', 'Relative_Humidity_pct', 'PET_mm', 'AET_mm',
    'Soil_Moisture_m3_m3', 'Runoff_mm', 'Groundwater_Depth_m', 'Sand_pct', 'Clay_pct',
    'Silt_pct', 'Soil_Organic_Carbon_pct', 'Soil_pH', 'Available_Water_Capacity',
    'Infiltration_Rate_mm_hr', 'NDVI', 'NDWI', 'Agriculture_pct', 'Forest_pct',
    'Builtup_pct', 'Waterbody_pct', 'Barren_Land_pct', 'Lineament_Density_km_per_km2',
    'Geological_Permeability_Index', 'Mean_Slope_deg', 'Drainage_Density_km_per_km2',
    'Stream_Frequency_per_km2', 'Rainfall_Score', 'Recharge_Score', 'Terrain_Score',
    'Drainage_Score', 'Vegetation_Score', 'Groundwater_Score', 'Runoff_Score',
    'WCHI_Score', 'WCHI_Priority', 'Recommended_Conservation_Structure'
]

LEAKAGE_EXCLUDED = {
    'Rainfall_Score', 'Recharge_Score', 'Terrain_Score', 'Drainage_Score',
    'Vegetation_Score', 'Groundwater_Score', 'Runoff_Score', 'WCHI_Score',
    'WCHI_Priority', 'Recommended_Conservation_Structure'
}

RAIN_FALL_WEIGHT = 0.18
RECHARGE_WEIGHT = 0.18
TERRAIN_WEIGHT = 0.15
DRAINAGE_WEIGHT = 0.15
VEGETATION_WEIGHT = 0.12
GROUNDWATER_WEIGHT = 0.12
RUNOFF_WEIGHT = 0.10

WEIGHTS = {
    'Rainfall': RAIN_FALL_WEIGHT,
    'Recharge': RECHARGE_WEIGHT,
    'Terrain': TERRAIN_WEIGHT,
    'Drainage': DRAINAGE_WEIGHT,
    'Vegetation': VEGETATION_WEIGHT,
    'Groundwater': GROUNDWATER_WEIGHT,
    'Runoff': RUNOFF_WEIGHT,
}

PRIORITY_ORDER = ['Low', 'Medium', 'High']

RAW_ENVIRONMENT_FEATURES = [
    'Rainfall_mm', 'Tmax_C', 'Tmin_C', 'Mean_Temperature_C', 'Relative_Humidity_pct',
    'PET_mm', 'AET_mm', 'Soil_Moisture_m3_m3', 'Runoff_mm', 'Streamflow_m3_s',
    'Groundwater_Depth_m', 'Sand_pct', 'Clay_pct', 'Silt_pct', 'Soil_Organic_Carbon_pct',
    'Soil_pH', 'Available_Water_Capacity', 'Infiltration_Rate_mm_hr', 'NDVI', 'NDWI',
    'Agriculture_pct', 'Forest_pct', 'Builtup_pct', 'Waterbody_pct', 'Barren_Land_pct',
    'Lineament_Density_km_per_km2', 'Geological_Permeability_Index', 'Population_Density_per_km2',
    'Mean_Slope_deg', 'Max_Slope_deg', 'Drainage_Density_km_per_km2', 'Stream_Frequency_per_km2',
    'Stream_Order'
]

MODEL_FILES = {
    'rainfall': MODELS_DIR / 'rainfall_model.pkl',
    'recharge': MODELS_DIR / 'recharge_model.pkl',
    'wchi': MODELS_DIR / 'wchi_model.pkl',
    'priority': MODELS_DIR / 'priority_model.pkl',
}

FEATURE_SET = {
    'rainfall': [
        'Month', 'Year', 'Latitude', 'Longitude', 'Tmax_C', 'Tmin_C', 'Mean_Temperature_C',
        'Relative_Humidity_pct', 'PET_mm', 'AET_mm', 'Soil_Moisture_m3_m3', 'NDVI',
        'NDWI', 'Mean_Slope_deg', 'Drainage_Density_km_per_km2', 'Runoff_mm',
        'Groundwater_Depth_m', 'Available_Water_Capacity', 'Infiltration_Rate_mm_hr'
    ],
    'recharge': [
        'Rainfall_mm', 'Soil_Moisture_m3_m3', 'Runoff_mm', 'Streamflow_m3_s',
        'Groundwater_Depth_m', 'Sand_pct', 'Clay_pct', 'Silt_pct', 'Available_Water_Capacity',
        'Infiltration_Rate_mm_hr', 'Lineament_Density_km_per_km2', 'Geological_Permeability_Index',
        'NDVI', 'NDWI', 'Mean_Elevation_m', 'Mean_Slope_deg', 'Drainage_Density_km_per_km2',
        'Stream_Frequency_per_km2', 'Soil_Organic_Carbon_pct', 'Agriculture_pct', 'Forest_pct'
    ],
    'wchi': RAW_ENVIRONMENT_FEATURES,
}

PRIORITY_INPUT_FEATURES = RAW_ENVIRONMENT_FEATURES

NEW_WATERSHED_FEATURES = [
    'Month', 'Year', 'Latitude', 'Longitude', 'Mean_Elevation_m',
    *RAW_ENVIRONMENT_FEATURES,
]
