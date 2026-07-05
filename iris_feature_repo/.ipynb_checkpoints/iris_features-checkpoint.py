from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from feast.value_type import ValueType  # <-- Add this import

# 1. Define the Data Source
iris_source = FileSource(
    path="data/iris_data_adapted_for_feast.parquet",
    timestamp_field="event_timestamp"
)

# 2. Define the Entity
iris_entity = Entity(
    name="iris_id",
    join_keys=["iris_id"],
    value_type=ValueType.INT64,  # <-- Change from Int64 to ValueType.INT64
    description="A unique identifier for each iris flower sample"
)

# 3. Define the Feature View
iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris_entity],
    ttl=timedelta(days=3650), 
    source=iris_source,
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="species", dtype=Int64), 
    ]
)