import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Set Remote MLflow Tracking Server (DagsHub)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("iris_mlsecops_poisoning")

# 2. Load Clean Test Data (Used across all runs to ensure fair evaluation)
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

# 3. Define the datasets and their corresponding poisoning levels
datasets = {
    "data/iris.csv": 0.0,
    "data/iris_poison_5.csv": 0.05,
    "data/iris_poison_10.csv": 0.10,
    "data/iris_poison_50.csv": 0.50
}

print("🚀 Starting MLSecOps Poisoning Experiments...")

# 4. Loop Through Dataset Configurations
for file_path, poison_level in datasets.items():
    
    # Name the run based on the poisoning level
    run_name = f"poisoning_{int(poison_level * 100)}_percent"
    
    with mlflow.start_run(run_name=run_name):
        
        # A. Read the specific training dataset variant
        df_train = pd.read_csv(file_path)
        X_train = df_train.drop(columns=["species"]) 
        y_train = df_train["species"]
        
        # B. Initialize and train model (Fixed hyperparameters for fair comparison)
        model = DecisionTreeClassifier(
            max_depth=5, 
            min_samples_split=2, 
            random_state=42
        )
        model.fit(X_train, y_train.values.ravel())
        
        # C. Evaluate model against the CLEAN test set
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average="weighted", zero_division=0)
        rec = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
        
        # D. Log Parameters (Specifically tracking poisoning level)
        mlflow.log_params({
            "poisoning_level": poison_level,
            "dataset_file": file_path,
            "max_depth": 5,
            "min_samples_split": 2,
            "random_state": 42
        })
        
        # E. Log Evaluation Metrics
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })
        
        # F. Log Model Artifact
        mlflow.sklearn.log_model(model, name="model")
        
        print(f"Logged Run [{run_name}] -> Accuracy: {acc:.4f} | F1: {f1:.4f}")

print("✅ All poisoning experiments completed and logged to MLflow!")