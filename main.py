from fastapi import FastAPI, HTTPException, Request
from fastapi.param_functions import Path as FastAPIPath  # Untuk parameter path
from pydantic import BaseModel, Field
import numpy as np
import os
import uvicorn
import onnxruntime as ort
import time
import logging
from typing import Dict, List
from huggingface_hub import snapshot_download
from pathlib import Path as PathlibPath  # Untuk operasi filesystem
import shutil
import uuid  # Ditambahkan di bagian atas file

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi model dictionary
model_sessions = {}
performance_metrics: Dict[str, Dict] = {}

def initialize_models():
    """Pastikan model benar-benar terload"""
    try:
        # Hardcode path untuk testing
        model_paths = {
            "v1": "models/v1.onnx", 
            "v2": "models/v2.onnx"
        }
        
        for ver, path in model_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file {path} not exists!")
                
            session = ort.InferenceSession(path)
            model_sessions[ver] = {
                "session": session,
                "input_name": session.get_inputs()[0].name
            }
            print(f"Successfully loaded {ver} from {path}")
            
    except Exception as e:
        print(f"CRITICAL ERROR in model loading: {str(e)}")
        raise

# Inisialisasi model
initialize_models()

app = FastAPI(
    title="Churn Prediction API",
    description="API untuk prediksi churn pelanggan dengan dukungan multi-versi model",
    version="3.0"
)

class ChurnRequest(BaseModel):
    features: List[float] = Field(..., min_items=19, max_items=19)

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0,0,1,0,45,1,0,2,0,0,0,0,0,0,1,1,2,89.85,4034.45]
            }
        }

# Inisialisasi struktur data performance_metrics
performance_metrics = {
    "serverless": {
        "response_times": [],
        "instance_id": str(uuid.uuid4())  # ID unik untuk instance
    },
    "non_serverless": {
        "response_times": [],
        "instance_id": str(uuid.uuid4())  # ID unik untuk instance
    }
}

@app.middleware("http")
async def track_performance(request: Request, call_next):
    start_time = time.time()
    
    # Tentukan mode berdasarkan path atau header
    mode = "serverless" if "serverless" in request.url.path else "non_serverless"
    
    try:
        response = await call_next(request)
    except Exception as e:
        raise e
    finally:
        process_time = time.time() - start_time
        performance_metrics[mode]["response_times"].append(process_time)
    
    return response

@app.get("/")
def home():
    return {"message": "Churn Prediction API 🚀", "mode": "serverless" if os.getenv("RAILWAY_ENVIRONMENT") else "non-serverless"}

@app.post("/predict/{version}")
async def predict_with_version(
    version: str,
    request_data: ChurnRequest,
    request: Request
):
    try:
        # Validasi versi model
        if version not in model_sessions:
            available_models = list(model_sessions.keys())
            raise HTTPException(
                status_code=404,
                detail={
                    "status": "error",
                    "error_type": "model_not_found",
                    "message": f"Model version '{version}' is not available",
                    "available_models": available_models,
                    "suggestion": f"Please use one of: {', '.join(available_models)}",
                    "request_id": str(uuid.uuid4())
                }
            )

        # Validasi input
        try:
            input_array = np.array(request_data.features, dtype=np.float32).reshape(1, -1)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "error_type": "invalid_input",
                    "message": "Invalid input features",
                    "details": str(e),
                    "expected_format": {
                        "type": "array",
                        "length": 19,
                        "content": "numeric values"
                    },
                    "example": [0,0,1,0,45,1,0,2,0,0,0,0,0,0,1,1,2,89.85,4034.45]
                }
            )

        # Prediksi
        session_info = model_sessions[version]
        outputs = session_info["session"].run(
            None,
            {session_info["input_name"]: input_array}
        )

        return {
            "status": "success",
            "prediction": int(outputs[0][0]),
            "probability": float(outputs[1][0][0]) if len(outputs) > 1 else 0.5,
            "model_version": version,
            "request_id": str(uuid.uuid4())
        }

    except HTTPException:
        raise  # Biarkan HTTPException yang sudah dibuat tetap seperti itu
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "error_type": "server_error",
                "message": "Internal server error during prediction",
                "request_id": str(uuid.uuid4()),
                "contact_support": "api-support@yourcompany.com"
            }
        )

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