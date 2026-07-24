import os
import pickle
import joblib
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI(
    title="Supermarket Intelligence Multi-Model Gateway using kubeflow",
    description="Dynamically servers all available Fraud Classifiers and Sales Regressors",
    version="2.0.0",
)

# 📁 Target directories mapped inside your Kubeflow Docker image
FRAUD_MODEL_DIR = "/app/Production/Models/fraud_ml_models"
SALES_MODEL_DIR = "/app/Production/Models/sales_ml_models"

loaded_engines = {"fraud": {}, "sales": {}}


@app.on_event("startup")
def dynamically_load_all_models():
    """Scans model directories and loads all available models into memory for inference."""
    try:
        print("🔄 Dynamically loading all available models from directories...")

        # Discover and Load All Fraud Models (.pkl)
        if os.path.exists(FRAUD_MODEL_DIR):
            for file in os.listdir(FRAUD_MODEL_DIR):
                if file.endswith(".pkl"):
                    model_name = os.path.splitext(file)[0]
                    full_path = os.path.join(FRAUD_MODEL_DIR, file)

                    with open(full_path, "rb") as f:
                        loaded_engines["fraud"][model_name] = pickle.load(
                            f
                        )  # nosec B301
                    print(f"✅ Loaded Fraud Model: {model_name}")

        else:
            print(f"⚠️ Fraud model directory not found: {FRAUD_MODEL_DIR}")

        # Discover and Load All Sales Models (.joblib)
        if os.path.exists(SALES_MODEL_DIR):
            for file in os.listdir(SALES_MODEL_DIR):
                if file.endswith(".joblib"):
                    model_name = os.path.splitext(file)[0]
                    full_path = os.path.join(SALES_MODEL_DIR, file)

                    loaded_engines["sales"][model_name] = joblib.load(full_path)
                    print(f"✅ Loaded Sales Model: {model_name}")
        else:
            print(f"⚠️ Sales model directory not found: {SALES_MODEL_DIR}")

        # Ensure at least one engine online
        if not loaded_engines["fraud"] and not loaded_engines["sales"]:
            raise RuntimeError(
                "No models were loaded. Ensure model files exist in the specified directories."
            )

        print(
            f"🚀 Dynamically initialization complete. Running serving registry: { {k: list(v.keys()) for k, v in loaded_engines.items()} }"
        )

    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        raise RuntimeError(f"Failed to load models: {e}")


class InferenceRequest(BaseModel):
    category: str
    sub_category: str
    quantity: int
    unit_price: float
    profit_margin: float
    sales: float


@app.get("/v1/models")
def kserve_health_probe():
    return {
        "status": "healthy",
        "active_engines": {k: list(v.keys()) for k, v in loaded_engines.items()},
    }


@app.post("/v1/models/supermarket-intelligence:predict")
def run_multi_model_inference(payload: InferenceRequest):
    """Endpoint to run inference using the dynamically loaded models."""
    try:
        input_data = payload.dict()
        response_payload = {"fraud_assessments": {}, "sales_assessments": {}}

        # Evaluate all active fraud classifiers
        if loaded_engines["fraud"]:
            df_fraud = pd.DataFrame(
                [
                    {
                        "category": input_data["category"],
                        "sub_category": input_data["sub_category"],
                        "quantity": input_data["quantity"],
                        "unit_price": input_data["unit_price"],
                        "profit_margin": input_data["profit_margin"],
                        "sales": input_data["sales"],
                    }
                ]
            )

            for name, model in loaded_engines["fraud"].items():
                pred = int(model.predict(df_fraud)[0])
                proba = (
                    model.predict_proba(df_fraud)[0].tolist()
                    if hasattr(model, "predict_proba")
                    else None
                )
                response_payload["fraud_assessments"][name] = {
                    "is_fraud": pred,
                    "probability_distribution": proba,
                }

        # Evaluate all active sales regressors
        if loaded_engines["sales"]:
            df_sales = pd.DataFrame(
                [
                    {
                        "category": input_data["category"],
                        "sub_category": input_data["sub_category"],
                        "quantity": input_data["quantity"],
                        "unit_price": input_data["unit_price"],
                        "profit_margin": input_data["profit_margin"],
                    }
                ]
            )

            for name, model in loaded_engines["sales"].items():
                pred = float(model.predict(df_sales)[0])
                response_payload["sales_assessments"][name] = {
                    "predicted_sales": round(pred, 2),
                    "variance_delta": round(float(input_data["sales"] - pred), 2),
                }

        return {"predictions:": response_payload}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Multi-engine processing cluster execution error: {str(e)}",
        )


if __name__ == "__main__":
    uvicorn.run(
        "Production.app.services.inference_gateway:app",
        host="0.0.0.0",  # nosec B104 - DEBUG FIX: Explicitly allow binding all interfaces in container
        port=8085,
    )
