
# Week 9: Explainability, Fairness, and Drift in the IRIS Pipeline

## Overview
This branch (`week_9`) shifts the focus from building and scaling machine learning pipelines to ensuring **ML Governance and Trustworthy AI**. While previous weeks secured the pipeline against external threats, this week ensures the model is inherently fair, explainable, and resilient to real-world data changes. 

Using the Iris dataset, we audit the model for demographic bias using **Fairlearn**, explain its internal decision-making process using **SHAP**, monitor simulated production data for statistical degradation using **Evidently**, and formalize accountability via a **Model Card**.

---

## 📂 Repository Structure

```text
├── .dvc/                   # DVC configuration and cache
├── .github/
│   └── workflows/
│       └── ci.yaml         # GitHub Actions CI pipeline
├── data/                   # Tracked via DVC
│   ├── iris.csv            # Clean baseline dataset
│   ├── X_test.csv          # Clean test features
│   └── y_test.csv          # Clean test labels
├── model_training/
│   ├── eval.py             
│   └── train.py            
├── tests/
│   ├── test_data.py        # Pytest data validation (Quality Gates)
│   └── test_model.py       
├── governance_analysis.py  # MAIN SCRIPT: Runs Fairlearn, SHAP, and Evidently drift detection
├── MODEL_CARD.md           # Formal ML governance documentation and nutrition label
├── shap_virginica_summary.png # SHAP output explaining feature impact for the Virginica class
├── drift_report.html       # Interactive Evidently dashboard showing data drift
├── Dockerfile              # Container blueprint
├── LICENSE                 # Repository license
├── README.md               # Project documentation (You are here)
└── requirements.txt        # Python dependencies (includes fairlearn, shap, evidently)

```

---

## 🚀 Execution Guide

### 1. Environment Setup

Activate your environment and install the newly added governance dependencies (`fairlearn`, `shap`, `evidently`):

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt

```

### 2. Run the Governance Analysis

A single unified script handles Data Preparation, Fairness Auditing, Explainability, and Data Drift detection:

```bash
python governance_analysis.py

```

### 3. View the Outputs

Running the script will generate the following artifacts in your root directory:

* **Terminal Output:** Displays the Fairlearn `MetricFrame` showing disaggregated accuracy, precision, and recall for the simulated `location` sensitive attribute.
* **`shap_virginica_summary.png`:** A SHAP summary plot visualizing how feature values (like petal length/width) push the model toward or away from predicting the Virginica class.
* **`drift_report.html`:** An interactive Evidently dashboard comparing the baseline training data against a simulated production dataset (where `petal_length` was artificially offset to simulate sensor drift).

---

## 🛡️ Key ML Governance Concepts Explored

* **Explainability (SHAP):** Moving beyond simple accuracy metrics to understand *why* a model made a specific prediction by distributing credit among input features.
* **Fairness (Fairlearn):** Auditing model predictions against sensitive demographic attributes (e.g., location) to ensure equitable performance across all subgroups and prevent proxy discrimination.
* **Data Drift (Evidently):** Continuously monitoring the statistical distribution of production data against the training baseline to catch sensor degradation or real-world shifts before they silently degrade model accuracy.
* **Model Cards:** Standardized documentation providing transparency regarding a model's intended use, training data, biases, and known limitations.

