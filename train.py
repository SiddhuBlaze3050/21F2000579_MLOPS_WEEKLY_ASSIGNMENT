import pandas as pd
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Initialize the Feature Store
# We point this to our Feast directory so the SDK can read the registry.db catalog we just built.
store = FeatureStore(repo_path="iris_feature_repo")

# 2. Build the Entity DataFrame
# Feast needs to know *who* we are predicting for, and *when* the prediction is happening.
# In production, this might come from a user-click stream or a label database. 
# Here, we will extract the IDs and timestamps from our raw parquet file to simulate our training cohort.
raw_data = pd.read_parquet("iris_feature_repo/data/iris_data_adapted_for_feast.parquet")
entity_df = raw_data[['iris_id', 'event_timestamp']]

# 3. Fetch Historical Features (The Offline Store)
# We hand Feast our list of IDs and Timestamps, and ask for specific features. 
# Feast automatically joins the data together, ensuring no future data leaks in!
print("Fetching historical features from Feast...")
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
        "iris_features:species"
    ]
).to_df()

print("Feature retrieval successful. Here is a peek at the Feast-generated dataset:")
print(training_df.head())

# 4. Train the Model
# From this point on, it is standard machine learning using Scikit-Learn!
X = training_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = training_df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel trained successfully! Accuracy: {accuracy * 100:.2f}%")