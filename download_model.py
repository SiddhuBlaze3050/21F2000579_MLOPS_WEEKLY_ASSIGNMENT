import os
import mlflow.sklearn
import joblib

# Connect to DagsHub MLflow Server
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

print("Fetching champion model from MLflow Registry...")
# Replace "1" with your specific version number or "@champion" if you set an alias
model_uri = "models:/iris_classifier/@champion" 
model = mlflow.sklearn.load_model(model_uri)

# Save it to the local directory so the API can use it
joblib.dump(model, "model.joblib")
print("Model successfully downloaded and saved as model.joblib!")