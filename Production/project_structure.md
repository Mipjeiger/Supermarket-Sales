## Project Structure

supermarket-sales-prediction/
├── .github/
│   └── workflows/
│       ├── ci_cd.yml
│       └── model_validation.yml
├── EDA/
│   └── 01_EDA.ipynb
├── LLMs/
│   ├── LLM.ipynb
│   ├── openai_llm.ipynb
│   └── vector_db/
│       └── chroma.sqlite3
├── Models/
│   ├── model_LinearRegression.pkl
│   ├── model_RandomForestRegressor.pkl
│   ├── model_XGBRegressor.pkl
│   └── model_performance_comparison.csv
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints/
│   │   │   ├── __init__.py
│   │   │   ├── prediction.py
│   │   │   ├── llm.py
│   │   │   └── monitoring.py
│   │   └── dependencies.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   └── security.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── raw/
│   │   │   └── SuperStoreOrders - SuperStoreOrders.csv
│   │   ├── cleaned/
│   │   │   └── data_sales_cleaned.parquet
│   │   ├── ingestion.py
│   │   ├── preprocessing.py
│   │   └── validation.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── init.sql
│   │   ├── models.py
│   │   └── crud.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── traditional/
│   │   │   ├── __init__.py
│   │   │   ├── linear_regression.py
│   │   │   ├── xgboost.py
│   │   │   └── random_forest.py
│   │   └── llm/
│   │       ├── __init__.py
│   │       ├── fine_tuning.py
│   │       ├── inference.py
│   │       └── prompt_templates.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── forecasting/
│   │   │   ├── __init__.py
│   │   │   └── prediction_service.py
│   │   ├── llm_agent/
│   │   │   ├── __init__.py
│   │   │   └── llm_service.py
│   │   ├── model_registry.py
│   │   └── data_service.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── alerts.py
│   │   └── slack_notifier.py
│   ├── src/
│   │   ├── __init__.py
│   │   └── data_ingestion.py
│   └── test/
│       ├── __init__.py
│       ├── test_api.py
│       └── test_models.py
├── dashboard/
│   ├── __init__.py
│   ├── app_ui.py
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── forecast.py
│   │   ├── model_performance.py
│   │   └── llm_chat.py
│   └── components/
│       ├── __init__.py
│       └── charts.py
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.streamlit
│   ├── Dockerfile.llm
│   └── Dockerfile.mlflow
├── kubernetes/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   └── hpa.yaml
├── scripts/
│   ├── train_models.py
│   ├── fine_tune_llm.py
│   └── benchmark_models.py
├── mlflow/
│   ├── __init__.py
│   ├── tracking.py
│   └── registry.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── pyproject.toml
├── Makefile
├── prometheus.yml
└── README.md