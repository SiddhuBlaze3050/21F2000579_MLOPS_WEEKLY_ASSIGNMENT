import os
import mlflow.sklearn
import pandas as pd

def test_model_accuracy():
    # 1. Load test data (still pulled via DVC)
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")
    
    # 2. Connect to MLflow and fetch the model
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    model_uri = "models:/iris_classifier/12" # Or whatever version/alias you used
    model = mlflow.sklearn.load_model(model_uri)
    
    # 3. Run predictions and assert accuracy
    predictions = model.predict(X_test)
    accuracy = (predictions == y_test.values.ravel()).mean()
    
    assert accuracy >= 0.85, f"Model accuracy {accuracy} is below the 85% threshold!"