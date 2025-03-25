from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import numpy as np
import os
import uvicorn
import onnxruntime as ort
import time
import logging
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model ONNX
onnx_model_path = "model/random_forest_churn.onnx"
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name

# Inisialisasi FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="API untuk prediksi churn pelanggan dengan perbandingan performa",
    version="2.0"
)

# Data untuk menyimpan metrik performa
performance_metrics: Dict[str, Dict] = {
    "serverless": {"response_times": [], "instance_id": None},
    "non_serverless": {"response_times": [], "instance_id": None}
}

# Struktur request


class ChurnRequest(BaseModel):
    features: List[float] = Field(..., min_items=19, max_items=19)

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0, 0, 1, 0, 45, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 2, 89.85, 4034.45]
            }
        }

# Middleware untuk tracking performa


@app.middleware("http")
async def track_performance(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    mode = "serverless" if os.getenv(
        "RAILWAY_ENVIRONMENT") == "production" else "non_serverless"
    performance_metrics[mode]["response_times"].append(process_time)

    # Catat instance yang menangani request
    hostname = os.getenv("HOSTNAME", "unknown")
    performance_metrics[mode]["instance_id"] = hostname

    return response


@app.get("/")
def home():
    return {"message": "Churn Prediction API 🚀", "mode": "serverless" if os.getenv("RAILWAY_ENVIRONMENT") else "non-serverless"}


@app.post("/predict")
def predict_churn(request: ChurnRequest):
    try:
        input_data = np.array(
            request.features, dtype=np.float32).reshape(1, -1)
        output = session.run(None, {input_name: input_data})

        pred_label = int(output[0][0])
        probabilities = output[1]
        prob = probabilities[pred_label] if pred_label in probabilities else 0.0

        return {
            "prediction": pred_label,
            "probability": round(prob, 4),
            "status": "Churn" if pred_label == 1 else "Not Churn",
            "handled_by": performance_metrics["serverless" if os.getenv("RAILWAY_ENVIRONMENT") else "non_serverless"]["instance_id"]
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/performance")
def get_performance():
    """Endpoint untuk membandingkan performa"""
    def calculate_stats(times: List[float]):
        if not times:
            return {"avg": 0, "min": 0, "max": 0, "count": 0}
        return {
            "avg": round(sum(times) / len(times), 4),
            "min": round(min(times), 4),
            "max": round(max(times), 4),
            "count": len(times)
        }

    return {
        "serverless": calculate_stats(performance_metrics["serverless"]["response_times"]),
        "non_serverless": calculate_stats(performance_metrics["non_serverless"]["response_times"]),
        "active_instances": {
            "serverless": performance_metrics["serverless"]["instance_id"],
            "non_serverless": performance_metrics["non_serverless"]["instance_id"]
        }
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
