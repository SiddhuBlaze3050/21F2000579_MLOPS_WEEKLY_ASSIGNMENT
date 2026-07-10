import pytest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import os

# Paths based on your dvc pull output
MODEL_PATH = "artifacts/model.joblib"
X_TEST_PATH = "data/X_test.csv"
Y_TEST_PATH = "data/y_test.csv"

@pytest.fixture
def test_data():
    """Load the evaluation dataset"""
    assert os.path.exists(X_TEST_PATH), f"Test features missing: {X_TEST_PATH}"
    assert os.path.exists(Y_TEST_PATH), f"Test labels missing: {Y_TEST_PATH}"
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)
    return X_test, y_test

@pytest.fixture
def model():
    """Load the trained model artifact"""
    assert os.path.exists(MODEL_PATH), f"Model missing: {MODEL_PATH}"
    return joblib.load(MODEL_PATH)

def test_model_accuracy(model, test_data):
    """TASK 2: Assert model accuracy meets a minimum threshold"""
    X_test, y_test = test_data
    
    # Run inference on the evaluation set
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Assert that accuracy is strictly greater than 85%
    # If the model degrades below this, the CI pipeline will fail the pull request!
    assert accuracy >= 0.95, f"Model accuracy degraded! Current accuracy: {accuracy * 100:.2f}%"

def test_model_prediction_shape(model, test_data):
    """TASK 2: Sanity check to ensure model outputs correct number of predictions"""
    X_test, _ = test_data
    predictions = model.predict(X_test)
    
    # Number of predictions must match number of input rows
    assert len(predictions) == len(X_test), "Mismatch between input rows and output predictions!"