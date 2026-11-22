
import pandas as pd

# Load dataset
df = pd.read_csv("data/heart.csv")

# Remove duplicate records
df = df.drop_duplicates()

# Replace invalid zero values with NaN
df["RestingBP"] = df["RestingBP"].replace(0, float("nan"))
df["Cholesterol"] = df["Cholesterol"].replace(0, float("nan"))

# Fill missing values using median
df["RestingBP"] = df["RestingBP"].fillna(df["RestingBP"].median())
df["Cholesterol"] = df["Cholesterol"].fillna(df["Cholesterol"].median())

# Identify categorical columns
categorical_columns = [
    "Sex", "ChestPainType", "RestingECG",
    "ExerciseAngina", "ST_Slope"
]

# Convert categorical columns into numerical columns
df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

# Separate features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

print("Cleaned Dataset Shape:", df.shape)
print("\nRemaining Missing Values:")
print(df.isnull().sum().sum())

print("\nEncoded Features:")
print(X.columns.tolist())

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# Save cleaned dataset
df.to_csv("data/cleaned_heart.csv", index=False)

print("\nCleaned dataset saved successfully!")
