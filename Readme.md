# MLOps Week 2: Integrating Data Version Control (DVC)

## Overview
This repository contains the Week 2 assignment for the MLOps pipeline, focusing on decoupling code versioning from data/artifact versioning. 

In Week 1, data and model artifacts were tracked manually via timestamped Google Cloud Storage (GCS) buckets using a Jupyter Notebook. For Week 2, the architecture has been upgraded to a production-grade setup using **Data Version Control (DVC)** backed by GCS, and the pipeline has been refactored into modular Python scripts.

## Key Architectural Changes
* **Modular Codebase:** Transitioned from a monolithic Jupyter Notebook to standalone `train.py` and `eval.py` scripts.
* **DVC Integration:** Datasets and model artifacts are now tracked via DVC. Git tracks the lightweight `.dvc` pointer files, while DVC handles the heavy binary files, pushing them to a centralized GCS remote bucket.
* **Experiment Reproducibility:** Training iterations are firmly locked to Git tags (e.g., `v0.1`, `v2.0`). Switching Git tags and running `dvc checkout` guarantees absolute reproducibility of historical experiments.
* **Automated Test Set Tracking:** To prevent "orphaned" evaluations during rollbacks, the generated `X_test.csv` and `y_test.csv` files are explicitly tracked alongside the core `iris.csv` and `model.joblib` files.

## Project Structure
```text
├── .dvc/                  # DVC internal configuration and cache
├── .github/               # (If applicable) CI/CD workflows
├── artifacts/
│   ├── model.joblib       # Serialized Decision Tree model (Tracked by DVC)
│   └── model.joblib.dvc   # DVC pointer for the model
├── data/
│   ├── iris.csv           # Active training dataset (Tracked by DVC)
│   ├── iris.csv.dvc       # DVC pointer for the dataset
│   ├── X_test.csv         # Generated test features (Tracked by DVC)
│   ├── X_test.csv.dvc
│   ├── y_test.csv         # Generated test targets (Tracked by DVC)
│   └── y_test.csv.dvc
├── eval.py                # Inference and evaluation script
├── train.py               # Model training and artifact generation script
└── .dvcignore             # DVC ignore rules