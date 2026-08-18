import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
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

    A_test = pd.Series(np.random.choice([0, 1], size=len(X_test)), name="location")

    model = DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42)
    model.fit(X_train, y_train)
    
    print("✅ Model trained successfully without the location feature!\n")
    return model, X_test, y_test, A_test

def assess_fairness(model, X_test, y_test, A_test):
    print("--- TASK 2: Fairlearn Fairness Audit ---")
    y_pred = model.predict(X_test)
    y_true = y_test.values.ravel() 
    
    metrics = {
        "accuracy": accuracy_score,
        "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=A_test
    )
    
    print("Metrics evaluated by Location group (0 vs 1):")
    print(metric_frame.by_group)
    print("✅ Fairness audit complete!\n")

def explain_with_shap(model, X_test):
    print("--- TASK 3: SHAP Explainability ---")
    
    # 1. Initialize the SHAP TreeExplainer with our trained model
    explainer = shap.TreeExplainer(model)
    
    # 2. Calculate SHAP values for the test dataset
    shap_values = explainer.shap_values(X_test)
    
    # 3. Generate and save the Summary Plot for Virginica (Class Index 2)
    print("Generating SHAP Summary Plot for Virginica (Class 2)...")
    
    # Scikit-learn Decision Trees return a list of arrays for multi-class classification
    # Index 0: Setosa, Index 1: Versicolor, Index 2: Virginica
    virginica_shap_values = shap_values[2] if isinstance(shap_values, list) else shap_values[:, :, 2]
    
    # Create the plot
    plt.figure()
    shap.summary_plot(virginica_shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot: Virginica")
    plt.tight_layout()
    
    # Save the plot to an image file so you can open it in your Workbench
    plt.savefig("shap_virginica_summary.png")
    print("✅ SHAP plot saved as 'shap_virginica_summary.png'.")

if __name__ == "__main__":
    model, X_test, y_test, A_test = prepare_data_and_train()
    assess_fairness(model, X_test, y_test, A_test)
    
    # Run Task 3
    explain_with_shap(model, X_test)