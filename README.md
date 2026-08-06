# Week 8: MLSecOps - Data Poisoning and Mitigation

## Overview

This branch (`week_8`) focuses on integrating MLSecOps principles into the `21F2000579_MLOPS_WEEKLY_ASSIGNMENT` pipeline. It explores machine learning security threat vectors and simulates a **data poisoning attack** on the Iris dataset. By injecting noise at varying severity levels (5%, 10%, and 50%), this project measures the resulting degradation in model performance (Accuracy, Precision, Recall, and F1 Score) using MLflow, and highlights the importance of data validation and quality gates in production.

---

## 📂 Repository Structure

```text
├── .dvc/                   # DVC configuration and cache
├── .github/
│   └── workflows/
│       └── ci.yaml         # GitHub Actions CI pipeline (Data validation & testing)
├── data/                   # Tracked via DVC
│   ├── iris.csv            # Clean baseline dataset (0% corruption)
│   ├── iris_poison_5.csv   # Dataset with 5% poisoned samples
│   ├── iris_poison_10.csv  # Dataset with 10% poisoned samples
│   ├── iris_poison_50.csv  # Dataset with 50% poisoned samples
│   ├── X_test.csv          # Clean test features
│   └── y_test.csv          # Clean test labels
├── model_training/
│   ├── eval.py             # Model evaluation utilities
│   └── train.py            # Model training script with MLflow tracking for poisoning levels
├── tests/
│   ├── test_data.py        # Pytest data validation (Schema, missing values, ranges)
│   └── test_model.py       # Pytest model validation (Skipped for Week 8 experiment focus)
├── .dvcignore              # DVC ignore rules
├── .gitignore              # Git ignore rules
├── Dockerfile              # Container blueprint
├── LICENSE                 # Repository license
├── README.md               # Project documentation (You are here)
├── poison_data.py          # Python script to generate the corrupted dataset variants
└── requirements.txt        # Python dependencies

```

---

## 🚀 Execution Guide

### 1. Environment Setup

Activate your environment and install the required dependencies:

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
pip install dagshub

```

### 2. Pulling the Data

Retrieve the base data using DVC:

```bash
dvc pull

```

### 3. Simulating Data Poisoning

Run the poisoning script to generate the corrupted variants (5%, 10%, and 50%):

```bash
python poison_data.py

```

*Note: The script replaces targeted features with random floats bounded by min/max dataset values and assigns a random target label.*

### 4. Running MLflow Experiments

Train the models across all dataset variants and track the metric degradation in DagsHub MLflow:

```bash
python model_training/train.py

```

### 5. Automated Data Validation (Quality Gates)

Run the test suite to demonstrate how production quality gates catch data schema and range anomalies:

```bash
pytest tests/ -v

```

---

## 🛡️ Key MLSecOps Concepts Explored

* **Data Poisoning:** Corrupting training data (Data Ingestion/Training phase) to degrade model accuracy.
* **Adversarial Examples:** Crafting perturbed inputs (Inference phase) to force a model misclassification.
* **Model Extraction:** Querying a deployed model repeatedly (API phase) to reconstruct or steal its parameters.
* **Prompt Injection:** Embedding malicious instructions in unstructured text (Inference phase) to override system logic.
* **Data Quality vs. Quantity:** When data is poisoned, scaling dataset size scales the noise. Security requires establishing strict data provenance and high clean-data ratios over raw data volume.