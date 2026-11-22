
import pandas as pd

# Load dataset
df = pd.read_csv("data/heart.csv")

print("Original Dataset Shape:", df.shape)

# 1. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 2. Check duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# 3. Check zero values in important columns
print("\nZero Values:")
print("RestingBP:", (df["RestingBP"] == 0).sum())
print("Cholesterol:", (df["Cholesterol"] == 0).sum())

# 4. Check categorical columns
print("\nCategorical Columns:")
print(df.select_dtypes(include="object").columns.tolist())

# 5. Check target distribution
print("\nTarget Distribution:")
print(df["HeartDisease"].value_counts())

print("\nFinal Dataset Shape:", df.shape)
