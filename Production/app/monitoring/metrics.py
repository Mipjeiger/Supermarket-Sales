from prometheus_client import Counter, Histogram, Gauge, Summary
import time

# Define metrics
PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Total number of predictions made", ["model_name", "status"]
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model_name"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

MODEL_RMSE = Gauge("model_rmse", "Current RMSE for each model", ["model_name"])

MODEL_MAE = Gauge("model_mae", "Current MAE for each model", ["model_name"])

MODEL_R2 = Gauge("model_r2", "Current R² score for each model", ["model_name"])

LLM_LATENCY = Histogram(
    "llm_latency_seconds",
    "LLM response latency in seconds",
    ["operation"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

API_REQUESTS = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "method", "status_code"]
)

DATA_QUALITY = Gauge("data_quality_score", "Data quality score (0-100)", ["dataset"])

SYSTEM_MEMORY = Gauge(
    "system_memory_bytes", "System memory usage in bytes", ["component", "type"]
)

SYSTEM_CPU = Gauge("system_cpu_percent", "System CPU usage percentage", ["component"])


class MetricsCollector:
    """Collect and track metrics."""

    @staticmethod
    def track_prediction(latency: float, model_name: str, status: str = "success"):
        """Track prediction metrics."""
        PREDICTIONS_TOTAL.labels(model_name=model_name, status=status).inc()
        PREDICTION_LATENCY.labels(model_name=model_name).observe(latency)

    @staticmethod
    def track_llm_response(latency: float, operation: str = "analysis"):
        """Track LLM response metrics."""
        LLM_LATENCY.labels(operation=operation).observe(latency)

    @staticmethod
    def track_api_request(endpoint: str, method: str, status_code: int):
        """Track API request metrics."""
        API_REQUESTS.labels(
            endpoint=endpoint, method=method, status_code=str(status_code)
        ).inc()

    @staticmethod
    def update_model_metrics(model_name: str, rmse: float, mae: float, r2: float):
        """Update model performance metrics."""
        MODEL_RMSE.labels(model_name=model_name).set(rmse)
        MODEL_MAE.labels(model_name=model_name).set(mae)
        MODEL_R2.labels(model_name=model_name).set(r2)

    @staticmethod
    def update_data_quality(dataset: str, quality_score: float):
        """Update data quality metrics."""
        DATA_QUALITY.labels(dataset=dataset).set(quality_score)
