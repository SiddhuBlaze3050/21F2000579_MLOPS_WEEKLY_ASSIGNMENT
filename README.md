# MLOps Week 4: Continuous Integration & Automated Testing

This repository contains the Week 4 assignment for the MLOps curriculum. The objective of this week's milestone was to build a robust, enterprise-grade Continuous Integration (CI) pipeline using GitHub Actions, ensuring that no degraded models or corrupted data make their way into production.

## 🚀 Project Overview

We transitioned from local, manual model training (Week 2) to a fully automated CI/CD environment. The pipeline automatically triggers on code pushes and Pull Requests, provisioning a cloud runner to authenticate with Google Cloud, retrieve large datasets and model artifacts via DVC, and execute a comprehensive suite of Python tests. Finally, it uses CML to deliver an automated evaluation report directly to the PR for code review.

## 🏗️ Architecture & Key Components

### 1. Automated Testing (`pytest`)
We implemented a strict quality gate using `pytest` located in the `tests/` directory:
* **Data Validation (`test_data.py`):** Ensures the `iris.csv` training data maintains the correct schema, contains no missing values, utilizes appropriate numeric datatypes, and contains no invalid/negative measurements.
* **Model Evaluation (`test_model.py`):** Loads the trained `model.joblib` artifact and evaluates it against the `X_test.csv` and `y_test.csv` holdout sets. It enforces a strict accuracy threshold (`>= 85%`). If the model performance degrades below this, the CI pipeline fails and blocks the merge.

### 2. Secure Cloud Authentication (Workload Identity Federation)
Instead of relying on long-lived, risky JSON Service Account keys, this repository establishes a trust relationship with Google Cloud using **Workload Identity Federation (WIF)**. 
* GitHub Actions generates a temporary, cryptographically signed OIDC token.
* GCP verifies the repository and issues a short-lived access token, granting read access to the DVC remote storage bucket.
* Credentials (`WIF_PROVIDER` and `WIF_SERVICE_ACCOUNT`) are securely stored in GitHub Secrets.

### 3. Data Version Control (DVC)
The pipeline utilizes `dvc-gs` to execute `dvc pull` during the CI run. This seamlessly bridges the Git repository with the GCP bucket, pulling down the exact version of the datasets and `artifacts/model.joblib` required for the tests to run.

### 4. Continuous Integration (GitHub Actions)
The orchestration is handled by `.github/workflows/ci.yaml`. On every push or PR to `main`, the workflow:
1. Provisions an `ubuntu-latest` runner.
2. Sets up Python 3.10 and installs requirements (`pytest`, `pandas`, `scikit-learn`, `dvc-gs`).
3. Authenticates securely to GCP via WIF.
4. Pulls versioned artifacts via DVC.
5. Executes the test suite.

### 5. Automated Reporting (CML)
Using **Continuous Machine Learning (CML)**, the pipeline captures the standard output of the `pytest` execution. If a Pull Request is open, the CML bot automatically formats the results into a Markdown report and posts it as a comment on the PR thread, providing instant visibility to reviewers.

## 💻 Running the Tests Locally

If you wish to clone this repository and run the validation pipeline on your local machine:

1. Ensure your local `gcloud` CLI is authenticated with the correct GCP project.
2. Install the required dependencies:
   ```bash
   pip install pytest pandas scikit-learn joblib dvc dvc-gs