
import pandas as pd

# Load dataset
df = pd.read_csv("data/heart.csv")

# Display dataset shape
print("Dataset Shape:", df.shape)

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Display column information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check target distribution
print("\nHeart Disease Distribution:")
print(df["HeartDisease"].value_counts())
