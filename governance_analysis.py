import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Set random seed for reproducibility
np.random.seed(42)

def prepare_data_and_train():
    print("1. Loading Training and Test datasets...")
    
    # Load the complete training data
    df_train = pd.read_csv("data/iris.csv")
    X_train = df_train.drop(columns=["species"])
    y_train = df_train["species"]

    # Load the separated test data
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")

    # TASK 1: Introduce the Sensitive Location Attribute
    print("2. Generating random 'location' attribute (0 or 1) for the test set...")
    
    # We generate the sensitive attribute array exclusively for the test set.
    # This ensures the location is used ONLY as a group identifier for the 
    # upcoming Fairlearn fairness analysis, and is completely isolated from the model.
    A_test = pd.Series(np.random.choice([0, 1], size=len(X_test)), name="location")

    print("3. Training the classifier exclusively on original features...")
    model = DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42)
    
    # Train the model (it remains completely blind to 'location')
    model.fit(X_train, y_train)
    
    print("✅ Task 1 Complete: Model trained successfully without the location feature!")
    
    return model, X_test, y_test, A_test

if __name__ == "__main__":
    model, X_test, y_test, A_test = prepare_data_and_train()