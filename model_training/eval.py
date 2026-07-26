import os
import pandas as pd
import mlflow.sklearn
from sklearn import metrics

# 1. Connect to the DagsHub MLflow Server
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

print("Loading test data...")
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

print("Fetching model from MLflow Registry...")
# 2. Define the exact model and version to fetch
model_name = "iris_classifier"
model_version = "12"  # Update this if you want a different version or use "@champion" for an alias
model_uri = f"models:/{model_name}/{model_version}"

# 3. Download and load the model directly into memory
model = mlflow.sklearn.load_model(model_uri)

print("Running inference...")
predictions = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predictions)

print(f"--- Inference Results ---")
print(f"Model Accuracy: {accuracy:.3f}")