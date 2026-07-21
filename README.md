# 🚀 Week 5: MLflow Integration & Experiment Tracking

## 📖 Overview
In Week 4, we established a robust CI/CD pipeline using DVC and GitHub Actions. For Week 5, we have upgraded our MLOps architecture by integrating **MLflow**. We decoupled our model storage from DVC and transitioned to a centralized MLflow Model Registry using **DagsHub** as our remote tracking server. 

This branch (`week_5`) demonstrates the transition from static model training to dynamic hyperparameter experimentation, comprehensive metric tracking, and registry-based model serving.

---

## 🏗️ Architectural Shift
*   **Previously (Week 1-4):** DVC tracked both the heavy CSV datasets and the `model.joblib` binary file. 
*   **Currently (Week 5):** **DVC** strictly handles data versioning (datasets). **MLflow** exclusively handles model versioning, lifecycle stages, experiment tracking, and artifact storage.

---

## ✨ Key Features & Tasks Completed

### 1. Hyperparameter Tuning (Task 1)
Refactored the static `train.py` script to include a nested loop that systematically iterates through a defined hyperparameter search space (`max_depth` and `min_samples_split`). This generates multiple distinct model configurations in a single run.

### 2. Experiment Tracking & DagsHub Integration (Task 2 & 3)
*   Instrumented the training loop with `mlflow.start_run()`.
*   Logged hyperparameters (`mlflow.log_params`), evaluation metrics (`mlflow.log_metrics`), and model binaries (`mlflow.sklearn.log_model`) directly to a remote tracking server.
*   Configured **DagsHub** as a zero-configuration remote backend, allowing us to visualize runs side-by-side using Parallel Coordinates and Scatter plots in the MLflow UI.

### 3. DVC Model Decoupling (Task 4)
*   Safely removed the `model.joblib` file from DVC tracking using `dvc remove`.
*   Updated `.gitignore` and `dvc.yaml` to reflect that DVC is now strictly responsible for data files (`X_test.csv`, `y_test.csv`, etc.).

### 4. Dynamic Model Evaluation (Task 5)
Refactored `eval.py` to eliminate hardcoded local file paths. The evaluation script now dynamically fetches the latest (or best-performing) model directly from the MLflow Model Registry via the DagsHub URI using `mlflow.sklearn.load_model()`.

### 5. Automated CI/CD with MLflow (Task 6)
Updated the `.github/workflows/ci.yaml` file and Pytest scripts (`tests/test_model.py`) to authenticate with DagsHub securely. The GitHub Actions runner now evaluates incoming Pull Requests by pulling the champion model directly from the MLflow Registry, ensuring production readiness.

---

## 🛠️ Setup & Execution

### Prerequisites
To run this pipeline locally, you must configure your environment to communicate with the remote DagsHub MLflow server. Export the following variables in your terminal:

```bash
export MLFLOW_TRACKING_USERNAME="<Your_DagsHub_Username>"
export MLFLOW_TRACKING_PASSWORD="<Your_DagsHub_Token>"
export MLFLOW_TRACKING_URI="[https://dagshub.com/](https://dagshub.com/)<Your_DagsHub_Username>/21F2000579_MLOPS_WEEKLY_ASSIGNMENT.mlflow"