# 🌸 Iris Classifier - Continuous Deployment (CD) Pipeline

## 📖 Overview

This repository contains the Continuous Deployment (CD) pipeline for an Iris species classification model. The project demonstrates a complete MLOps workflow, transitioning a machine learning model from a DagsHub MLflow registry into a fully containerized FastAPI web service, which is automatically deployed to Google Kubernetes Engine (GKE) using GitHub Actions.

## 🏗️ Architecture & Workflow

1. **Model Retrieval:** During the Docker build process, the champion model is securely downloaded from the DagsHub MLflow Model Registry and baked directly into the image.
2. **Containerization:** The inference code and model are packaged using Docker.
3. **CI/CD Pipeline:** GitHub Actions automatically builds the Docker image and pushes it to Google Artifact Registry upon every push to the `week_6` branch.
4. **Keyless Authentication:** The pipeline uses Workload Identity Federation (WIF) to securely interact with Google Cloud without relying on long-lived JSON service account keys.
5. **Kubernetes Deployment:** The pipeline automatically applies Kubernetes manifests to update the deployment and service on a GKE cluster, exposing the API via a public Load Balancer.

## 📂 Repository Structure

```text
.
├── .github/workflows/
│   └── cd.yaml              # GitHub Actions Continuous Deployment pipeline
├── k8s/
│   ├── deployment.yaml      # Kubernetes manifest for the API pods
│   └── service.yaml         # Kubernetes manifest for the Load Balancer
├── Dockerfile               # Blueprint for building the API container
├── requirements.txt         # Python dependencies
├── iris_fastapi.py          # FastAPI application serving the predictions
└── download_model.py        # Script to fetch the MLflow model at build time

```

## 🚀 Step-by-Step Implementation Summary

### 1. Containerization & MLflow Integration (Tasks 2 & 6)

* **FastAPI Service:** Created `iris_fastapi.py` to expose `/` and `/predict/` endpoints. The script loads a local `.joblib` model and returns string predictions (e.g., `"setosa"`).
* **Build-Time Model Fetching:** Created `download_model.py` to securely connect to DagsHub and download the best registered model (`models:/iris_classifier/1`).
* **Docker Image:** Configured a `Dockerfile` using `python:3.10-slim`. It accepts DagsHub credentials as `--build-arg` variables, executes the model download script during the build, and sets up `uvicorn` to serve the API on port `8200`. This ensures the live container never requires internet access to MLflow to serve predictions.

### 2. Google Cloud Infrastructure & Security (Task 3)

* **Workload Identity Federation (WIF):** Bypassed restrictive organizational policies on JSON key creation by reusing existing WIF infrastructure.
* **IAM Roles:** Granted the CI/CD service account the `roles/artifactregistry.writer` and `roles/container.developer` roles, allowing GitHub to safely push images and update the cluster.
* **Artifact Registry:** Provisioned a Docker repository named `my-repo` in `us-central1`.

### 3. Continuous Deployment via GitHub Actions (Task 4)

* **Automated Workflow:** Created `cd.yaml` to trigger on pushes to the `week_6` branch.
* **Docker Build & Push:** The runner securely builds the image, passing in GitHub Secrets (`DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`) so the model can be fetched. The completed image is tagged and pushed to Google Artifact Registry.

### 4. Google Kubernetes Engine Deployment (Task 5)

* **GKE Cluster:** Provisioned a single-node Kubernetes cluster (`test-iris-v1`) in GCP.
* **Auth Plugin Fix:** Integrated `google-github-actions/get-gke-credentials@v2` in the CI/CD pipeline to automatically install the required `gke-gcloud-auth-plugin` and authenticate `kubectl`.
* **Declarative Infrastructure:** Deployed the application using `deployment.yaml` (managing the pod replicas) and `service.yaml` (exposing the application via a public Load Balancer on port 80).

## 💻 API Usage

The model can be queried by sending a POST request with the four physical measurements of the Iris flower to the Load Balancer's external IP.

**Example Request:**

```bash
curl -X POST "http://<EXTERNAL-IP>/predict/" \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

```

**Example Response:**

```json
{
  "predicted_class": "setosa"
}

```