import pandas as pd

# Load the WCHI dataset
df = pd.read_csv("dataset/Lower_Vellar_WCHI_Synthetic_Dataset.csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nWCHI Priority Distribution:")
print(df["WCHI_Priority"].value_counts())

print("\nFirst 5 Rows:")
print(df.head())