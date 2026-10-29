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
    "Year",
    "Month",
    "Latitude",
    "Longitude",
    "Tmax_C",
    "Tmin_C",
    "Mean_Temperature_C",
    "Relative_Humidity_pct",
    "PET_mm",
    "AET_mm"
]

target = "Rainfall_Score"


X_train = df[df["Dataset_Split"] == "Train"][features]
y_train = df[df["Dataset_Split"] == "Train"][target]

X_val = df[df["Dataset_Split"] == "Validation"][features]
y_val = df[df["Dataset_Split"] == "Validation"][target]

X_test = df[df["Dataset_Split"] == "Test"][features]
y_test = df[df["Dataset_Split"] == "Test"][target]


# ==========================================
# 3. CREATE MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# ==========================================
# 4. TRAIN
# ==========================================

print("\nTraining Rainfall Model...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 5. VALIDATION
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
# 6. TEST
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
# 7. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/rainfall_model.pkl"
)

print("\nRainfall model saved successfully!")
print("File: models/rainfall_model.pkl")