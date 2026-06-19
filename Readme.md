# IRIS ML Pipeline on Vertex AI 🌸

**Course:** MLOps - Assignment 1 (MAY 2026)

**Student ID:** 21F2000579

## 📖 Overview

This repository contains the implementation of a foundational end-to-end machine learning pipeline for an IRIS classifier utilizing Google Cloud Platform (GCP). The objective is to establish robust MLOps practices—specifically focusing on data plumbing, compute isolation, and artifact traceability—without relying on heavy orchestration frameworks.

## 🏗️ Architecture & Cloud Infrastructure

* **Compute:** Vertex AI Workbench (JupyterLab environment)
* **Storage:** Google Cloud Storage (GCS) buckets for tracking versioned datasets and dynamically timestamped model artifacts.
* **Frameworks:** Scikit-Learn, Pandas, Google Cloud AI Platform SDK.

## 📂 Directory Structure

Based on the parameterized execution lifecycle, the local Workbench environment dynamically manages the following structure:

```text
.
├── 21F2000579_Assignment_1_MAY_2026_MLOps.ipynb  # Main Execution Notebook
├── artifacts/                                    # Local staging for trained models
│   └── model.joblib
├── data/                                         # Versioned data directories
│   ├── v0/
│   │   ├── iris.csv                              # Raw baseline data
│   │   ├── train.csv                             # Processed 60% train split
│   │   └── test.csv                              # Processed 40% test split
│   ├── v1/                                       # Data Version 1
│   └── v2/                                       # Data Version 2
├── train_downloaded_v*.csv                       # Fetched from GCS during training pipeline
├── test_downloaded_v*.csv                        # Fetched from GCS during inference pipeline
└── model_downloaded.joblib                       # Fetched from GCS during inference pipeline

```

## 🚀 Pipeline Tasks Fulfilled

1. **GCP Setup (Task 1):** Initialized Vertex AI and authenticated with a dedicated GCS bucket (`gs://mlops-course-project-...`).
2. **Parameterized Data Ingestion (Task 2):** Programmatically stratifies and splits raw `iris.csv` into `train.csv` and `test.csv` based on a global `DATA_VERSION` parameter, organizing them locally and syncing them to GCS.
3. **Traceable Training Pipeline (Task 3):** Fetches versioned training data from GCS, trains a `DecisionTreeClassifier`, and dynamically logs the output to a timestamp-and-version-tagged GCS folder (e.g., `artifacts/2026-06-19T13-36-59_v2/`).
4. **Isolated Inference Pipeline (Task 4):** Strictly decouples testing from training by retrieving the exact timestamped model artifact and its corresponding test dataset from GCS to evaluate accuracy.
5. **Multi-Version Execution & Comparison (Tasks 5 & 6):** The pipeline is parameterized to seamlessly toggle between data versions (`v0`, `v1`, `v2`). This fulfills the requirement to run the pipeline multiple times and simulates handling data drift, creating discrete, traceable artifacts for every data iteration.

## ⚙️ Usage / How to Run

1. Ensure your GCP trial account is active and the Vertex AI API is enabled.
2. Open `21F2000579_Assignment_1_MAY_2026_MLOps.ipynb`.
3. Locate the Pipeline Parameter block in the cells:
```python
# --- PIPELINE PARAMETER ---
DATA_VERSION = "v0" # Toggle to "v1" or "v2"
# --------------------------

```


4. Execute the cells sequentially to trigger data ingestion, model training, and isolated inference for the selected data version.