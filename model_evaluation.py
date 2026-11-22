
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Load dataset
df = pd.read_csv("data/cleaned_heart.csv")

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train best performing model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Disease"]
)

disp.plot(cmap="Blues")
plt.title("Heart Disease Prediction - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

# Accuracy comparison graph
results = {
    "Logistic Regression": 89.13,
    "Decision Tree": 77.72,
    "Random Forest": 87.50
}

plt.figure(figsize=(8, 5))

sns.barplot(
    x=list(results.keys()),
    y=list(results.values())
)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.xlabel("Machine Learning Models")
plt.ylim(0, 100)
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

print("Evaluation completed!")
