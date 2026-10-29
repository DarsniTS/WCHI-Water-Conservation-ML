import pandas as pd

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(
    "dataset/Lower_Vellar_WCHI_Synthetic_Dataset.csv"
)

print("Dataset loaded successfully")


# ==========================================
# FEATURES USED BY RECHARGE MODEL
# ==========================================

features = [
    "Rainfall_mm",
    "Mean_Temperature_C",
    "Relative_Humidity_pct",
    "PET_mm",
    "AET_mm",
    "Soil_Moisture_m3_m3",
    "Groundwater_Depth_m",
    "Sand_pct",
    "Clay_pct",
    "Silt_pct",
    "Soil_Organic_Carbon_pct",
    "Available_Water_Capacity",
    "Infiltration_Rate_mm_hr",
    "NDVI",
    "NDWI",
    "Lineament_Density_km_per_km2",
    "Geological_Permeability_Index",
    "Mean_Slope_deg",
    "Drainage_Density_km_per_km2",
    "Waterbody_pct"
]


# ==========================================
# FEATURE DISTRIBUTION
# ==========================================

print("\n========== FEATURE MEANS BY SPLIT ==========")

for feature in features:

    print(f"\n{feature}")

    for split in ["Train", "Validation", "Test"]:

        data = df[df["Dataset_Split"] == split]

        mean_value = data[feature].mean()

        print(f"  {split}: {mean_value:.4f}")


# ==========================================
# FEATURE MIN/MAX BY SPLIT
# ==========================================

print("\n\n========== FEATURE RANGES BY SPLIT ==========")

for feature in features:

    print(f"\n{feature}")

    for split in ["Train", "Validation", "Test"]:

        data = df[df["Dataset_Split"] == split]

        min_value = data[feature].min()
        max_value = data[feature].max()

        print(
            f"  {split}: "
            f"Min={min_value:.4f}, "
            f"Max={max_value:.4f}"
        )


print("\n========== CHECK COMPLETED ==========")