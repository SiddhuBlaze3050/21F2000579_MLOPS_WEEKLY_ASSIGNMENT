# Feast Feature Store Implementation: MLOps Pipeline

**Author:** Siddhartha Devulapalli  
**Program:** B.S. in Data Science and Applications, IIT Madras  
**Module:** MLOps Weekly Assignment (Feature Stores)

## 📌 Project Overview
This repository demonstrates the integration of **Feast**, an open-source feature store, into a machine learning pipeline. The primary objective is to decouple machine learning model training and inference scripts from raw data engineering pipelines, thereby preventing data leakage (training-serving skew) and standardizing feature access.

This project implements two distinct architectures:
1. **Local Architecture:** A lightweight, self-contained pipeline using Parquet files (Offline Store) and SQLite (Online Store/Registry).
2. **Cloud Architecture (GCP):** An enterprise-grade migration utilizing Google BigQuery for distributed point-in-time historical joins.

---

## 🏗️ Architecture & Component Breakdown

### Task 1: Repository Initialization
- Initialized a self-contained Feast feature repository (`iris_feature_repo`).
- Configured the `feature_store.yaml` blueprint to establish a local SQLite database for both the central metadata registry (`registry.db`) and the low-latency serving database (`online.db`).

### Task 2: Schema Definition & Application
- Defined the core data structures in `iris_features.py`.
- Established an `Entity` representing the primary key (`iris_id`).
- Created a `FeatureView` mapping the structural schema (sepal/petal dimensions and target species) to the raw `.parquet` offline data source.
- Executed `feast apply` to validate the schemas and register them into the SQLite catalog.

### Task 3: Data Materialization (Online Store Sync)
- Executed `feast materialize` to load point-in-time feature vectors from the offline Parquet file into the SQLite online store.
- **Data Engineering Fix:** Handled a schema collision where raw target variables were strings (`'setosa'`, `'versicolor'`). Programmatically remapped the categorical strings to strictly typed integers (0, 1, 2) to comply with both the Feast schema and downstream model requirements.

### Task 4: Offline Feature Retrieval (Model Training)
- Built `train.py` to completely bypass direct file reading. 
- Utilized `store.get_historical_features()` to pass an Entity DataFrame (timestamps and IDs) to Feast, ensuring the engine reconstructs the features exactly as they existed at specific points in time.
- Trained a Scikit-Learn `LogisticRegression` model on the safely joined dataset and serialized the artifact to `iris_model.joblib`.

### Task 5: Online Feature Retrieval (Real-Time Inference)
- Created `inference.py` to simulate a low-latency production API request.
- Utilized `store.get_online_features()` to instantly fetch the most recent feature values for specific `iris_ids` directly from the SQLite database.
- **Consistency Verification:** Wrote a deterministic check to compare the model predictions generated from the online store against predictions generated from the raw Parquet file. Implemented time-based deduplication (`sort_values` and `drop_duplicates`) to successfully prove parity between the two datasets.

---

## ☁️ Task 6: GCP BigQuery Migration (Optional Challenge)

To demonstrate scalability, the local offline store was migrated to Google Cloud BigQuery in a parallel environment (`iris_feature_repo_bigquery`). 

### Migration Steps:
1. Replicated the feature repository to isolate the cloud infrastructure.
2. Modified `feature_store.yaml` to point the offline store to BigQuery.
3. Updated `iris_features.py` to replace `FileSource` with `BigQuerySource`.
4. Uploaded the Parquet data to a dynamically provisioned BigQuery dataset.

### Technical Challenges & Solutions:
- **IAM Permission Constraints:** Handled a locked-down sandbox environment by identifying the underlying Compute Engine Default Service Account and binding the `BigQuery Admin` IAM role via the CLI to enable table creation.
- **Typological Collisions (`INT64` vs `STRING`):** BigQuery’s auto-detect CLI feature incorrectly inferred the Unix epoch timestamps as `INT64`, causing the point-in-time SQL join to crash when Feast passed the entity DataFrame timestamps as strings.
  - **Resolution:** Bypassed the CLI uploader entirely. Leveraged the Google Cloud BigQuery Python SDK to explicitly enforce timezone-aware Pandas datetimes, forcing the BigQuery schema to initialize a native `TIMESTAMP` column.

### Architectural Trade-Offs Observed:
| Feature | Local Setup (Parquet + SQLite) | Cloud Setup (BigQuery + SQLite) |
| :--- | :--- | :--- |
| **Compute Engine** | Local RAM / Dask | Distributed SQL Engine |
| **Scalability** | Limited by local machine memory | Effortlessly handles Terabyte-scale joins |
| **Latency** | Instantaneous offline retrieval | Minor network/API latency for job execution |
| **Best For** | Rapid prototyping, CI/CD testing | Production model training, massive datasets |

---

## 🚀 How to Run

**1. Run the Local Pipeline:**
```bash
# Register infrastructure and materialize data
cd iris_feature_repo
feast apply
feast materialize 2000-01-01T00:00:00 2030-01-01T00:00:00
cd ..

# Train model via offline store
python train.py

# Run real-time inference via online store
python inference.py