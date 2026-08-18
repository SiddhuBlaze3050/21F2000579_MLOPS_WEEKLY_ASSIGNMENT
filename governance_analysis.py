import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame

# Corrected Evidently Imports based on your environment
from evidently import Report
from evidently.presets import DataDriftPreset

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
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("Generating SHAP Summary Plot for Virginica (Class 2)...")
    virginica_shap_values = shap_values[2] if isinstance(shap_values, list) else shap_values[:, :, 2]
    
    plt.figure()
    shap.summary_plot(virginica_shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot: Virginica")
    plt.tight_layout()
    plt.savefig("shap_virginica_summary.png")
    print("✅ SHAP plot saved as 'shap_virginica_summary.png'.\n")

def detect_data_drift():
    print("--- TASK 4: Data Drift Detection with Evidently ---")
    
    # 1. Load the clean original training data (Our Reference Baseline)
    reference_data = pd.read_csv("data/iris.csv")
    
    # 2. Simulate Production Data (Our Current Data)
    current_data = reference_data.copy()
    current_data['petal_length'] = current_data['petal_length'] + 2.5
    
    # 3. Initialize the Evidently Report
    print("Running statistical drift tests...")
    drift_report = Report(metrics=[DataDriftPreset()])
    
    # CRITICAL FIX for Evidently 0.7+: Capture the returned snapshot object
    report_snapshot = drift_report.run(reference_data=reference_data, current_data=current_data)
    
    # 4. Save the report to an interactive HTML file using the snapshot
    report_path = "drift_report.html"
    report_snapshot.save_html(report_path)
    
    print(f"✅ Data Drift report saved successfully as '{report_path}'.")

if __name__ == "__main__":
    model, X_test, y_test, A_test = prepare_data_and_train()
    assess_fairness(model, X_test, y_test, A_test)
    explain_with_shap(model, X_test)
    
    # Run Task 4
    detect_data_drift()