import pandas as pd
import joblib
from sklearn import metrics

print("Loading model and test data...")
model = joblib.load("artifacts/model.joblib")
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

print("Running inference...")
predictions = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predictions)

print(f"--- Inference Results ---")
print(f"Model Accuracy: {accuracy:.3f}")