# 📊 Model Card: IRIS Species Classifier

## 1. Model Details
* **Architecture:** Decision Tree Classifier (`scikit-learn`)
* **Version:** 1.0.0
* **Date:** August 2026
* **Purpose:** To classify Iris flowers into one of three species (Setosa, Versicolor, Virginica) based on botanical measurements.

---

## 2. Intended Use
* **Primary Use Case:** Educational demonstration of MLOps, MLSecOps, and ML Governance pipelines.
* **Out-of-Scope Uses:** This model is not intended for commercial botanical classification, automated agriculture, or any production system without further robustness testing.

---

## 3. Training Data
* **Dataset:** R.A. Fisher's Iris Dataset.
* **Features:** 4 numerical features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`). 
* **Target:** `species` (Categorical: Setosa, Versicolor, Virginica).
* **Exclusions:** A simulated demographic feature (`location`) was intentionally excluded from the training data to prevent noise and ensure the model did not use group identifiers as predictive features.

---

## 4. Performance Metrics
The model was evaluated against a held-out test set (`X_test.csv`). 

* **Overall Accuracy:** ~100.0%
* **Overall F1 Score (Weighted):** ~1
* **Disaggregated Performance by Sensitive Attribute (`location`):**
  * **Location 0:** Accuracy ~100% | Recall ~100%
  * **Location 1:** Accuracy ~100% | Recall ~100%

---

## 5. Fairness Considerations
* **Sensitive Attribute Monitored:** `location` (Binary: 0 or 1).
* **Audit Tool:** `fairlearn.metrics.MetricFrame`
* **Conclusion:** The model demonstrated equal performance across both location subgroups. There is no evidence of demographic disparity or proxy discrimination regarding the location attribute.

---

## 6. Known Limitations & Risks
* **Data Drift Vulnerability:** The model relies heavily on `petal_length` and `petal_width` (verified via SHAP analysis). If real-world data drifts significantly from the training distribution (e.g., systemic sensor errors inflating petal length), the model will confidently misclassify inputs.
* **Adversarial Threats:** The model is susceptible to data poisoning. Injecting random noise into as little as 10% of the training data noticeably degrades the F1 score. Strict schema enforcement and anomaly detection are required during data ingestion.