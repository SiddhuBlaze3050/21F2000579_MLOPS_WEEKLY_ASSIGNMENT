# Week 7: AI/ML Infrastructure – Autoscaling, Load Testing, and Observability on GKE

## 📌 Overview
This repository contains the Week 7 Graded Assignment for the MLOps pipeline. The primary objective of this week was to evolve a containerized machine learning model (Iris Classification) into a highly available, observable, and auto-scaling API deployed on Google Kubernetes Engine (GKE). 

This project demonstrates core AI/ML Systems and Infrastructure Engineering practices, including continuous deployment, high-concurrency stress testing, horizontal autoscaling, and distributed telemetry.

## 🛠️ Technology Stack
* **Application:** Python 3.11, FastAPI, Uvicorn, Scikit-learn
* **Infrastructure:** Google Kubernetes Engine (GKE), Docker, Kubernetes (Deployments, Services, HPA)
* **CI/CD & Testing:** GitHub Actions, `wrk` (HTTP benchmarking)
* **Observability:** OpenTelemetry, GCP Cloud Logging (Structured JSON), GCP Cloud Monitoring, GCP Cloud Trace
* **Model Registry:** MLflow (via DagsHub)

---

## 🚀 Tasks Completed

### Task 1: Continuous Deployment & Automated Stress Testing
Extended the GitHub Actions CI/CD workflow (`.github/workflows/cd.yaml`) to automate the deployment to GKE. Following a successful rollout using the `Recreate` deployment strategy, the pipeline dynamically fetches the LoadBalancer's External IP and executes an automated HTTP load test using the `wrk` benchmarking tool and a custom `post.lua` script.

### Task 2: High-Concurrency Traffic Simulation
Simulated extreme production traffic by bombarding the deployed API endpoint with over 1,000 concurrent connections.
* **Tool:** `wrk`
* **Configuration:** `wrk -t4 -c1000 -d120s -s post.lua http://<EXTERNAL_IP>/predict`
* **Result:** Captured baseline latency, throughput (Requests/sec), and socket error rates to establish a performance benchmark prior to enabling autoscaling.

### Task 3: Horizontal Pod Autoscaler (HPA) Configuration
Implemented dynamic scaling capabilities (`k8s/hpa.yaml`) to ensure API reliability under heavy load.
* **Rules:** `minReplicas: 1`, `maxReplicas: 3`
* **Trigger:** Configured to scale up when average CPU utilization across existing pods exceeds **60%**.
* **Result:** Successfully observed the cluster dynamically provision new pods to distribute the `wrk` load, preventing connection timeouts.

### Task 4: Observability (GCP Cloud Monitoring & Logging)
Instrumented the FastAPI application with OpenTelemetry and bound the Kubernetes deployment to a GCP Workload Identity service account (`telemetry-access`).
* **Cloud Logging:** Implemented structured JSON logging for all predictions, capturing custom fields like `trace_id`, `latency_ms`, and `predicted_class`. 
* **Cloud Monitoring:** Monitored CPU and Memory saturation across pod replicas in real-time during the load tests to visually verify load distribution.

### Task 5: Bottleneck Analysis Under Constrained Scaling
Conducted a destructive test to identify system failure modes.
* **Scenario:** Restricted the HPA to `maxReplicas: 1` and doubled the `wrk` connections to `2,000`.
* **Findings:** Verified that the primary bottleneck was the strict CPU limit (`500m`). As CPU throttled, requests queued at the network layer, causing the `latency_ms` in our custom logs to spike exponentially. This eventually resulted in widespread socket timeouts, proving the absolute necessity of the HPA implemented in Task 3.

---

## 📂 Repository Structure

```text
├── .github/workflows/
│   └── cd.yaml              # CI/CD Pipeline with GKE deployment and wrk load testing
├── k8s/
│   ├── deployment.yaml      # K8s Deployment with health probes, resources, and strategy
│   ├── service.yaml         # LoadBalancer Service configuration
│   └── hpa.yaml             # Horizontal Pod Autoscaler definition
├── Dockerfile               # Production container blueprint
├── download_model.py        # MLflow model retrieval script
├── iris_fastapi.py          # FastAPI application with OpenTelemetry instrumentation
├── post.lua                 # Lua script for wrk POST request payload
└── requirements.txt         # Python dependencies