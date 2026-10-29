import pandas as pd

# Load the WCHI dataset
df = pd.read_csv("dataset/Lower_Vellar_WCHI_Synthetic_Dataset.csv")

print("\n========== ALL COLUMNS ==========")

for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())