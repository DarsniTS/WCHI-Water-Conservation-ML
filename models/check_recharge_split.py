import pandas as pd

# Load dataset
df = pd.read_csv(
    "dataset/Lower_Vellar_WCHI_Synthetic_Dataset.csv"
)

target = "Recharge_Score"

print("\n========== RECHARGE SCORE BY SPLIT ==========")

for split in ["Train", "Validation", "Test"]:

    data = df[df["Dataset_Split"] == split][target]

    print(f"\n--- {split} ---")
    print("Rows   :", len(data))
    print("Min    :", data.min())
    print("Max    :", data.max())
    print("Mean   :", data.mean())
    print("Std    :", data.std())


print("\n========== MICRO-WATERSHEDS BY SPLIT ==========")

for split in ["Train", "Validation", "Test"]:

    data = df[df["Dataset_Split"] == split]

    print(f"\n--- {split} ---")
    print("Rows:", len(data))
    print("Unique watersheds:", data["Micro_Watershed_ID"].nunique())
    print("Watersheds:", data["Micro_Watershed_ID"].unique())


print("\n========== RECHARGE SCORE SAMPLE ==========")

print(
    df[
        [
            "Micro_Watershed_ID",
            "Date",
            "Recharge_Score",
            "Dataset_Split"
        ]
    ].head(20)
)