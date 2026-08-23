# Week 10: From MLOps to LLMOps — Fine-Tuning Gemma on the IRIS Pipeline

## 🚀 Overview
This project marks the transition from traditional MLOps (managing tabular feature vectors and Scikit-Learn classifiers) to **LLMOps**. Instead of training a model from scratch, we adapt a pre-trained foundation model to classify the classic IRIS dataset. 

The core experiment of this week is an **A/B test on data representation**. We fine-tune two versions of an instruction-tuned model (`gemma-3-1b-it`) to determine if structuring data as natural language prose yields better performance than feeding the LLM raw key-value pairs.

---

## 📂 Repository Structure

```text
├── .github/
│   └── workflows/
│       └── ci.yaml             # LLMOps CI/CD pipeline with regression guards
├── data/                   
│   ├── iris.csv                # Baseline dataset
│   ├── v1_raw.jsonl            # SFT Data: Raw key-value representation
│   ├── v2_desc.jsonl           # SFT Data: Natural language description representation
│   ├── X_test.csv              
│   └── y_test.csv              
├── prepare_llm_data.py         # Script to convert tabular CSV data into JSONL formats
├── llm_eval.py                 # Custom LLM evaluation script measuring Format Compliance & Concept Accuracy
├── .gitignore
├── Dockerfile              
├── README.md               
└── requirements.txt            # Includes transformers, torch, accelerate, and dvc

```

---

## 🛠️ Tasks & Implementation

### Phase 1: LLM Data Engineering (Tasks 1 & 2)

Traditional ML models process 2D arrays, but LLMs require sequential text tokens. The `prepare_llm_data.py` script transforms the tabular dataset into two distinct JSONL representations:

* **V1 (Raw Features):** `{"input_text": "sepal_length: 5.1, sepal_width: 3.5...", "output_text": "setosa"}`
* **V2 (Natural Language):** `{"input_text": "A flower specimen has a sepal length of 5.1 cm... Identify the iris species.", "output_text": "This is Iris setosa."}`

### Phase 2: Supervised Fine-Tuning (Task 3)

Using Google Cloud Vertex AI infrastructure, two parallel fine-tuning jobs were executed on the `gemma-3-1b-it` model. Hyperparameters (epochs, learning rate) were strictly locked across both jobs to isolate data representation as the sole independent variable.

### Phase 3: Evaluation & LLM Failure Modes (Task 4)

Evaluating LLMs requires custom metrics. The `llm_eval.py` script pulls the fine-tuned model checkpoints from Google Cloud Storage and runs a CPU-optimized inference loop to calculate:

1. **Format Compliance Rate:** Does the model output the strict species name without hallucinations or conversational filler?
2. **Concept Accuracy:** Is the correct species identified anywhere within the generated text?

**🧪 Experimental Findings:**

* **Format Compliance (0%):** Because `gemma-3-1b-it` is an instruction-tuned model, training it on raw `input_text`/`output_text` pairs without its native chat template caused a severe prompt mismatch. The model's conversational pre-training overrode the formatting constraints, resulting in outputs like *"Based on the measurements provided, this flower is..."* This highlights a critical LLMOps vulnerability: generative models require strict prompt-template anchoring.
* **Concept Accuracy (V1: 28.33% vs. V2: 33.33%):** Despite failing strict formatting, the V2 natural language model successfully identified the underlying species 5% more often than the V1 raw feature model. **Conclusion:** Translating structured tabular data into an LLM's native language improves task comprehension.

### Phase 4: Automated CI/CD (Task 5)

The GitHub Actions workflow (`ci.yaml`) was upgraded for LLMOps. On every push, the CI runner:

1. Authenticates keylessly via Workload Identity Federation (WIF).
2. Downloads the multi-gigabyte Gemma checkpoints from GCS.
3. Executes `llm_eval.py` as a regression guard.
4. Fails the build if Concept Accuracy drops below the defined threshold (25%), preventing degraded LLM behavior from reaching production.

---

## 💻 Execution Guide

### 1. Environment Setup

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt

```

### 2. Generate JSONL Datasets

```bash
python prepare_llm_data.py

```

### 3. Run Local Evaluation

*Ensure you have authenticated with `gcloud auth application-default login` to access the GCS bucket.*

```bash
# Download the weights locally first
mkdir -p models/v1_raw models/v2_desc
gcloud storage cp -r "gs://<YOUR_BUCKET_NAME>/fine-tuning/.../final/*" models/v1_raw/
gcloud storage cp -r "gs://<YOUR_BUCKET_NAME>/fine-tuning/.../final/*" models/v2_desc/

# Execute the evaluation script
python llm_eval.py

```