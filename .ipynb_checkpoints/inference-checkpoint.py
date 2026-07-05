import pandas as pd
import joblib
from feast import FeatureStore

# 1. Initialize the Feature Store and Load the Model
print("Loading model and connecting to Feast Online Store...")
store = FeatureStore(repo_path="iris_feature_repo")
model = joblib.load("iris_model.joblib")

# Updated target IDs
request_ids = [{"iris_id": 1001}, {"iris_id": 1002}, {"iris_id": 1003}]

# 2. Fetch Real-Time Features from the Online Store
print("\nFetching real-time features from Feast...")
online_response = store.get_online_features(
    entity_rows=request_ids,
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width"
    ]
)

inference_df = pd.DataFrame(online_response.to_dict())
# FIX 1: Feast APIs don't guarantee return order. We must explicitly sort by iris_id.
inference_df = inference_df.sort_values('iris_id').reset_index(drop=True)

print("\n--- Features Retrieved from Online Store ---")
print(inference_df[["iris_id", "sepal_length", "sepal_width", "petal_length", "petal_width"]])

X_inference = inference_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
predictions = model.predict(X_inference)

print(f"\n--- Online Predictions ---")
print(f"Predicted Species mapped integers: {predictions}")

# =====================================================================
# 3. Consistency Check (Proving the architecture works)
# =====================================================================
print("\n--- Running Consistency Check against Raw Data ---")
raw_data = pd.read_parquet("iris_feature_repo/data/iris_data_adapted_for_feast.parquet")

# Filter for the newly requested IDs
raw_samples = raw_data[raw_data['iris_id'].isin([1001, 1002, 1003])]

# FIX 2: Mimic Feast's behavior by getting ONLY the newest record for each ID
raw_samples = raw_samples.sort_values('event_timestamp', ascending=False)
raw_samples = raw_samples.drop_duplicates(subset=['iris_id'], keep='first')

# FIX 3: Sort the raw samples so they line up perfectly with the inference_df
raw_samples = raw_samples.sort_values('iris_id').reset_index(drop=True)

X_raw = raw_samples[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
raw_predictions = model.predict(X_raw)

print(f"Raw Data Predictions: {raw_predictions}")

if list(predictions) == list(raw_predictions):
    print("\n✅ SUCCESS: Online store predictions perfectly match raw data predictions!")
else:
    print("\n❌ WARNING: Mismatch detected between online store and raw data.")