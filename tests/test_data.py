import pytest
import pandas as pd
import os

# 🛑 UPDATE THIS PATH to point to your actual training data CSV
DATA_PATH = "data/iris.csv" 

@pytest.fixture
def data():
    """
    This is a pytest fixture. It loads the data once and provides it to all the tests below.
    This saves memory and time.
    """
    assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    return df

def test_expected_schema(data):
    """TASK 1: Check for expected schema"""
    expected_columns = [
        "sepal_length", 
        "sepal_width", 
        "petal_length", 
        "petal_width", 
        "species" # or "species" depending on your dataset
    ]
    
    for col in expected_columns:
        assert col in data.columns, f"Missing expected column: {col}"

def test_missing_values(data):
    """TASK 1: Check for missing/null values"""
    missing_counts = data.isnull().sum().sum()
    assert missing_counts == 0, f"Found {missing_counts} missing values in the dataset!"

def test_feature_types(data):
    """TASK 1: Check for correct feature types"""
    # The features should be numeric (float64)
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    
    for col in features:
        # Check if the column is a float or int type
        assert pd.api.types.is_numeric_dtype(data[col]), f"Column {col} should be numeric, but is {data[col].dtype}"

def test_reasonable_value_ranges(data):
    """TASK 1: Check for reasonable value ranges"""
    # In the real world, a flower cannot have negative petal lengths!
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    
    for col in features:
        min_value = data[col].min()
        assert min_value > 0, f"Column {col} contains invalid negative or zero values: {min_value}"