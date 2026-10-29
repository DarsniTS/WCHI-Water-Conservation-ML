import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Lower_Vellar_WCHI_Synthetic_Dataset.csv")

# Score columns
score_columns = [
    "Rainfall_Score",
    "Recharge_Score",
    "Terrain_Score",
    "Drainage_Score",
    "Vegetation_Score",
    "Groundwater_Score",
    "Runoff_Score",
    "WCHI_Score"
]

print("\n========== DATASET INFO ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n========== DATASET SPLIT ==========")
print(df["Dataset_Split"].value_counts())

print("\n========== SCORE INFORMATION ==========")

for col in score_columns:
    print(f"\n--- {col} ---")
    print("Minimum :", df[col].min())
    print("Maximum :", df[col].max())
    print("Mean    :", round(df[col].mean(), 2))
    print("Unique values:", df[col].nunique())

print("\n========== WCHI PRIORITY ==========")
print(df["WCHI_Priority"].value_counts())

print("\n========== SAMPLE SCORE DATA ==========")
print(df[
    [
        "Micro_Watershed_ID",
        "Rainfall_mm",
        "Rainfall_Score",
        "Recharge_Score",
        "Terrain_Score",
        "Drainage_Score",
        "Vegetation_Score",
        "Groundwater_Score",
        "Runoff_Score",
        "WCHI_Score",
        "WCHI_Priority"
    ]
].head(10))