import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "dataset/Lower_Vellar_WCHI_Synthetic_Dataset.csv"
)

print("Dataset loaded successfully")
print("Rows:", len(df))


# ==========================================
# 2. SELECT INPUT FEATURES
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

target = "Recharge_Score"


# ==========================================
# 3. DATASET SPLIT
# ==========================================

train = df[df["Dataset_Split"] == "Train"]
validation = df[df["Dataset_Split"] == "Validation"]
test = df[df["Dataset_Split"] == "Test"]


X_train = train[features]
y_train = train[target]

X_val = validation[features]
y_val = validation[target]

X_test = test[features]
y_test = test[target]


print("\n========== DATA SPLIT ==========")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))


# ==========================================
# 4. CREATE MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\nTraining Recharge Model...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 6. VALIDATION
# ==========================================

val_predictions = model.predict(X_val)

val_mae = mean_absolute_error(y_val, val_predictions)
val_rmse = mean_squared_error(y_val, val_predictions) ** 0.5
val_r2 = r2_score(y_val, val_predictions)


print("\n========== VALIDATION RESULTS ==========")
print("MAE :", val_mae)
print("RMSE:", val_rmse)
print("R2  :", val_r2)


# ==========================================
# 7. TEST
# ==========================================

test_predictions = model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = mean_squared_error(y_test, test_predictions) ** 0.5
test_r2 = r2_score(y_test, test_predictions)


print("\n========== TEST RESULTS ==========")
print("MAE :", test_mae)
print("RMSE:", test_rmse)
print("R2  :", test_r2)


# ==========================================
# 8. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/recharge_model.pkl"
)

print("\nRecharge model saved successfully!")
print("File: models/recharge_model.pkl")