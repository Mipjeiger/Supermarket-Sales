import os
from kfp import dsl, compiler

# Import all components from the components.py file
from components import (
    data_ingestion_and_split,
    train_fraud_xgb_model,
    train_fraud_rf_model,
    train_fraud_gbc_model,
    train_sales_xgb_model,
    train_sales_rf_model,
    train_sales_cbr_model,
    train_sales_dt_model
)

@dsl.pipeline(
    name="supermarket-intelligence-pipeline",
    description="End-to-end parallel MLOps architecture executing concurrent training for 3 Fraud Classifiers and 4 Sales Regressors."
)
def supermarket_intelligence_pipeline(
    dataset_path: str = "/app/Production/app/data/cleaned/combined_sql_supermarket.parquet"
):
    # 1. Data Ingestion and Target stream partitioning
    ingest_task = data_ingestion_and_split(dataset_path=dataset_path)
    ingest_task.set_cpu_limit('1').set_memory_limit('2G')

    # 2, Fraud detection pipeline task (Executed in parallel)
    fraud_xgb = train_fraud_xgb_model(dataset=ingest_task.outputs['fraud_dataset'])
    fraud_xgb.set_cpu_limit('2').set_memory_limit('4G')

    fraud_rf = train_fraud_rf_model(dataset=ingest_task.outputs['fraud_dataset'])
    fraud_rf.set_cpu_limit('2').set_memory_limit('4G')

    fraud_gbc = train_fraud_gbc_model(dataset=ingest_task.outputs['fraud_dataset'])
    fraud_gbc.set_cpu_limit('2').set_memory_limit('4G')

    # 3. Sales prediction pipeline task (Executed in parallel)
    sales_xgb = train_sales_xgb_model(dataset=ingest_task.outputs['sales_dataset'])
    sales_xgb.set_cpu_limit('2').set_memory_limit('4G')

    sales_rf = train_sales_rf_model(dataset=ingest_task.outputs['sales_dataset'])
    sales_rf.set_cpu_limit('2').set_memory_limit('4G')

    sales_cbr = train_sales_cbr_model(dataset=ingest_task.outputs['sales_dataset'])
    sales_cbr.set_cpu_limit('2').set_memory_limit('4G')

    sales_dt = train_sales_dt_model(dataset=ingest_task.outputs['sales_dataset'])
    sales_dt.set_cpu_limit('2').set_memory_limit('4G')

if __name__ == "__main__":
    # The output compile destination configuration pipeline
    output_filename = "supermarket_pipeline_specification.yaml"

    print(f"📦 Compiling multi-model workflow to: {output_filename}...")
    compiler.Compiler().compile(
        pipeline_func=supermarket_intelligence_pipeline,
        package_path=output_filename
    )
    print("🏁 Success! Upload this file to your Kubeflow Pipelines UI to trigger the execution matrix.")