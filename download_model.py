import os
import mlflow.sklearn
import joblib

# Connect to DagsHub MLflow Server
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# CRITICAL FIX: You MUST include the exact version number (e.g., /1)
# Ensure "iris_classifier" exactly matches the name in your DagsHub MLflow UI
model_uri = "models:/iris_classifier/12" 

print(f"Fetching champion model from: {model_uri}...")
model = mlflow.sklearn.load_model(model_uri)

# Save it to the local directory so the API can use it
joblib.dump(model, "model.joblib")
print("Model successfully downloaded and saved as model.joblib!")