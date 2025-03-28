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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi model dictionary
model_sessions = {}
performance_metrics: Dict[str, Dict] = {}

def initialize_models():
    """Download dan inisialisasi semua versi model"""
    try:
        # Unduh semua model dari Hugging Face Hub
        temp_dir = snapshot_download(
            repo_id="ajimaulana77/churn-model",
            allow_patterns=["*.onnx"]
        )
        
        # Folder target - gunakan PathlibPath
        base_dir = PathlibPath("models")
        base_dir.mkdir(exist_ok=True)
        
        for onnx_file in PathlibPath(temp_dir).glob("**/*.onnx"):
            # Ekstrak versi dari nama file
            version = onnx_file.stem[1:] if onnx_file.stem.startswith('v') else onnx_file.stem
            filename = f"v{version}.onnx"
            dest_path = base_dir / filename
            
            # Salin file model
            shutil.copy(onnx_file, dest_path)
            
            # Inisialisasi ONNX Runtime session
            model_key = f"v{version}"
            model_sessions[model_key] = {
                "session": ort.InferenceSession(str(dest_path)),
                "input_name": ort.InferenceSession(str(dest_path)).get_inputs()[0].name,
                "model_path": str(dest_path)
            }
            
            performance_metrics[model_key] = {
                "response_times": [],
                "instance_id": None
            }
            
            logger.info(f"Model versi {model_key} berhasil diinisialisasi dari {filename}")
    
    except Exception as e:
        logger.error(f"Gagal menginisialisasi model: {str(e)}")
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


@app.post("/predict/{version}")
def predict_with_version(
    version: str = FastAPIPath(..., title="Versi model", description="Versi model yang akan digunakan (contoh: v1, v2)"),
    request: ChurnRequest = None
):
    """Prediksi churn dengan model versi spesifik"""
    try:
        if version not in model_sessions:
            available_versions = list(model_sessions.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Versi {version} tidak tersedia. Versi yang ada: {available_versions}"
            )
            
        session_info = model_sessions[version]
        input_data = np.array(request.features, dtype=np.float32).reshape(1, -1)
        
        output = session_info["session"].run(
            None, 
            {session_info["input_name"]: input_data}
        )

        pred_label = int(output[0][0])
        probabilities = output[1]
        prob = probabilities[pred_label] if pred_label in probabilities else 0.0

        return {
            "prediction": pred_label,
            "probability": round(prob, 4),
            "status": "Churn" if pred_label == 1 else "Not Churn",
            "model_version": version,
            "handled_by": performance_metrics[version]["instance_id"]
        }
        
    except Exception as e:
        logger.error(f"Error prediksi dengan versi {version}: {str(e)}")
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