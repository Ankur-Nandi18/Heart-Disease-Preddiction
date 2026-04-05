import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("heart.csv")

# Features and target
X = data.drop("target", axis=1)

# New Target Meaning:
# 0 = Low Risk
# 1 = High Risk
# Original dataset target:
# 0 = Disease Present
# 1 = No Disease
# So reverse it

y = 1 - data["target"]
# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)
# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "model/heart_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model and scaler saved successfully.")