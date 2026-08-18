import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame

# Set random seed for reproducibility
np.random.seed(42)

def prepare_data_and_train():
    print("--- TASK 1: Data Prep & Training ---")
    df_train = pd.read_csv("data/iris.csv")
    X_train = df_train.drop(columns=["species"])
    y_train = df_train["species"]

    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")

    # Generate sensitive attribute array for the test set
    A_test = pd.Series(np.random.choice([0, 1], size=len(X_test)), name="location")

    # Train the model 
    model = DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42)
    model.fit(X_train, y_train)
    
    print("✅ Model trained successfully without the location feature!\n")
    return model, X_test, y_test, A_test

def assess_fairness(model, X_test, y_test, A_test):
    print("--- TASK 2: Fairlearn Fairness Audit ---")
    
    # 1. Get model predictions
    y_pred = model.predict(X_test)
    y_true = y_test.values.ravel() # Flatten to 1D array for metrics
    
    # 2. Define the metrics we want to audit
    # Using macro average because Iris is a multi-class dataset
    metrics = {
        "accuracy": accuracy_score,
        "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    # 3. Create the MetricFrame
    # This evaluates the metrics grouped by the sensitive attribute (location)
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=A_test
    )
    
    # 4. Print the disaggregated results
    print("Metrics evaluated by Location group (0 vs 1):")
    print(metric_frame.by_group)
    
    print("\n✅ Fairness audit complete! (Expect near-equal results since location is random)")

if __name__ == "__main__":
    # Run Task 1
    model, X_test, y_test, A_test = prepare_data_and_train()
    
    # Run Task 2 (passing the in-memory variables)
    assess_fairness(model, X_test, y_test, A_test)