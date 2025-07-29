import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
csv = os.path.join("..", "data", "pose_landmarks.csv")
df = pd.read_csv(csv)

# Separate features and label
X = df.drop('label', axis=1)
y = df['label']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save label encoder
os.makedirs("../models", exist_ok=True)
joblib.dump(le, "../models/label_encoder.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=21, stratify=y_encoded
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"[INFO] Model Accuracy: {accuracy * 100:.2f}%")

report = classification_report(y_test, y_pred, target_names=le.classes_)
print("[INFO] Classification Report:\n", report)

# Save model and predictions
joblib.dump(model, "../models/pose_classifier.pkl")
joblib.dump((y_test, y_pred), "../models/test_predictions.pkl")
joblib.dump((model, le), "../models/test_model.pkl")


