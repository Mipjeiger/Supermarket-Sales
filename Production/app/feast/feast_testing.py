from feast import FeatureStore

# Initialize store pointing to your feature_store.yaml directory
store = FeatureStore(repo_path="Production/app/feast")

# Query features for a sample order_id
entity_rows = [{"order_id": 9012}]  # Replace with a real order_id from your dataset

response = store.get_online_features(
    features=[
        "sales_features:quantity",
        "sales_features:profit_margin",
        "fraud_features:shipping_days",
    ],
    entity_rows=entity_rows,
)

print(response.to_df())
