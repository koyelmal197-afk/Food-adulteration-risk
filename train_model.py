import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Load dataset (FIXED Windows path)
# -------------------------------
data = pd.read_csv(r"C:\Users\USER\Documents\food_data.csv")

# -------------------------------
# Check target column
# -------------------------------
if "Risk_Level" not in data.columns:
    raise ValueError("Target column 'Risk_Level' not found in dataset")

# -------------------------------
# Split features and target
# -------------------------------
X = data.drop("Risk_Level", axis=1)
y = data["Risk_Level"]

# -------------------------------
# Handle categorical data
# -------------------------------
X = pd.get_dummies(X, drop_first=True)

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# -------------------------------
# Train model
# -------------------------------
model.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# Save model (IMPORTANT)
# -------------------------------
joblib.dump(model, "food_adulteration_model.pkl")

print("\nModel saved as food_adulteration_model.pkl")

#save model
joblib.dump("food_adulteration_model.pkl")
print("Model saved as food_adulteration_model.pkl")