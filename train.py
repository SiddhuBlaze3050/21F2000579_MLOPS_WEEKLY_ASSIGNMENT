import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# 1. Set Remote MLflow Tracking Server (DagsHub)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("iris_hyperparameter_tuning")


# 2. Load Data and Split
# Read the complete dataset
df = pd.read_csv("data/iris.csv")

# Separate the features (X) from the target label (y)
X_train = df.drop(columns=["species"]) 
y_train = df["species"]

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

# 3. Define Hyperparameter Search Space
max_depth_options = [3, 5, 10, None]
min_samples_split_options = [2, 5, 10]

print("🚀 Starting MLflow Hyperparameter Experiments...")

# 4. Loop Through Configurations
for depth in max_depth_options:
    for min_samples in min_samples_split_options:
        
        # Name the run based on parameters for easy identification
        run_name = f"depth_{depth}_split_{min_samples}"
        
        with mlflow.start_run(run_name=run_name):
            # A. Initialize and train model
            model = DecisionTreeClassifier(
                max_depth=depth, 
                min_samples_split=min_samples, 
                random_state=42
            )
            model.fit(X_train, y_train.values.ravel())
            
            # B. Evaluate model
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            prec = precision_score(y_test, predictions, average="weighted")
            rec = recall_score(y_test, predictions, average="weighted")
            f1 = f1_score(y_test, predictions, average="weighted")
            
            # C. Log Hyperparameters
            mlflow.log_params({
                "max_depth": str(depth),
                "min_samples_split": min_samples,
                "random_state": 42
            })
            
            # D. Log Evaluation Metrics
            mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
            })
            
            # E. Log Model Artifact directly to MLflow
            mlflow.sklearn.log_model(model, artifact_path="model")
            
            print(f"Logged Run [{run_name}] -> Accuracy: {acc:.4f}")

print("✅ All experiments completed and logged to MLflow!")