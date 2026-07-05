from datetime import timedelta
from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.bigquery_source import BigQuerySource
from feast.types import Float32, Int64
from feast.value_type import ValueType

# Swap FileSource for BigQuerySource
iris_source = BigQuerySource(
    table="project-d5c04e1d-be23-4954-84c.feast_iris_dataset.iris_offline_table",
    timestamp_field="event_timestamp"
)

# The Entity and FeatureView remain exactly the same!
iris_entity = Entity(name="iris_id", join_keys=["iris_id"], value_type=ValueType.INT64)

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