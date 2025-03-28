from fastapi import FastAPI, HTTPException, Request
from fastapi.param_functions import Path as FastAPIPath
from pydantic import BaseModel, Field
import numpy as np
import os
import uvicorn
import onnxruntime as ort
import time
import logging
from typing import Dict, List, Optional
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path as PathlibPath
import shutil
import uuid
import datetime
import threading
import schedule
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REPO_ID = "ajimaulana77/churn-model"  # Your Hugging Face repository
MAX_TOTAL_SIZE_MB = 200  # Maximum total size for all models
MAX_MODELS_UNDER_40MB = 5  # Max models if each <40MB
MAX_MODELS_OVER_40MB = 3  # Max models if any >40MB
REFRESH_HOURS = 6  # Auto-refresh interval

# Global variables
model_sessions = {}
performance_metrics: Dict[str, Dict] = {}

def get_model_versions() -> List[str]:
    """Get available model versions from Hugging Face Hub sorted by version number"""
    try:
        files = list_repo_files(repo_id=REPO_ID)
        onnx_files = [f for f in files if f.endswith('.onnx') and f.startswith('v')]
        versions = sorted([f.split('.')[0] for f in onnx_files], 
                         key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
        return versions
    except Exception as e:
        logger.error(f"Failed to list repo files: {str(e)}")
        return []

def get_file_size_mb(path: PathlibPath) -> float:
    """Get file size in megabytes"""
    return path.stat().st_size / (1024 * 1024)

def cleanup_models(model_dir: PathlibPath, keep_versions: List[str]):
    """Remove old model files not in keep_versions"""
    for model_file in model_dir.glob("*.onnx"):
        version = model_file.stem
        if version not in keep_versions:
            try:
                model_file.unlink()
                logger.info(f"Removed old model: {version}")
            except Exception as e:
                logger.error(f"Failed to remove {version}: {str(e)}")

def initialize_models(force_refresh: bool = False):
    """Download and initialize models with smart storage management"""
    try:
        model_dir = PathlibPath("models")
        model_dir.mkdir(exist_ok=True)
        
        # Get available versions from Hugging Face (newest first)
        available_versions = get_model_versions()
        if not available_versions:
            raise ValueError("No models found in Hugging Face Hub")
        
        logger.info(f"Available versions in HF Hub: {available_versions}")
        
        # Determine which versions to keep based on size limits
        versions_to_download = []
        total_size = 0
        model_count = 0
        max_models = MAX_MODELS_UNDER_40MB
        
        # Check if any model is over 40MB to apply stricter limit
        sample_size = None
        try:
            sample_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=available_versions[-1] + ".onnx",
                local_dir=model_dir,
                force_download=False,
                resume_download=False
            )
            sample_size = get_file_size_mb(PathlibPath(sample_path))
            if sample_size > 40:
                max_models = MAX_MODELS_OVER_40MB
            logger.info(f"Sample model size: {sample_size:.2f}MB")
        except Exception as e:
            logger.warning(f"Couldn't check model size: {str(e)}")
        
        # Select newest models within our limits
        for version in reversed(available_versions):
            if model_count >= max_models:
                break
                
            try:
                # Download model
                downloaded_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=f"{version}.onnx",
                    local_dir=model_dir,
                    force_download=force_refresh,
                    resume_download=False
                )
                
                # Check size
                model_size = get_file_size_mb(PathlibPath(downloaded_path))
                if total_size + model_size > MAX_TOTAL_SIZE_MB:
                    logger.warning(f"Skipping {version} due to size limit")
                    PathlibPath(downloaded_path).unlink(missing_ok=True)
                    continue
                    
                versions_to_download.append(version)
                total_size += model_size
                model_count += 1
                logger.info(f"Added {version} ({model_size:.2f}MB), total: {total_size:.2f}MB")
                
            except Exception as e:
                logger.error(f"Failed to process {version}: {str(e)}")
        
        # Cleanup old models not in our selected versions
        cleanup_models(model_dir, versions_to_download)
        
        # Initialize model sessions
        for version in versions_to_download:
            model_path = model_dir / f"{version}.onnx"
            try:
                session = ort.InferenceSession(str(model_path))
                model_sessions[version] = {
                    "session": session,
                    "input_name": session.get_inputs()[0].name,
                    "last_updated": datetime.datetime.now().isoformat(),
                    "file_path": str(model_path),
                    "size_mb": get_file_size_mb(model_path)
                }
                logger.info(f"Loaded model {version} ({model_sessions[version]['size_mb']:.2f}MB)")
                
            except Exception as e:
                logger.error(f"Failed to load {version}: {str(e)}")
                if version not in model_sessions:
                    raise

    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}", exc_info=True)
        if not model_sessions:
            raise RuntimeError("No models available! Service cannot start")

def refresh_models():
    """Background job to update models"""
    logger.info("Running scheduled model refresh...")
    try:
        initialize_models(force_refresh=False)
    except Exception as e:
        logger.error(f"Failed to refresh models: {str(e)}")

def schedule_runner():
    """Background scheduler"""
    schedule.every(REFRESH_HOURS).hours.do(refresh_models)
    while True:
        schedule.run_pending()
        time.sleep(60)

# Initialize models and scheduler
initialize_models()

app = FastAPI(
    title="Churn Prediction API",
    description="API untuk prediksi churn pelanggan dengan dukungan multi-versi model",
    version="3.0"
)

# Start background scheduler
@app.on_event("startup")
async def startup_event():
    threading.Thread(target=schedule_runner, daemon=True).start()
    
    # Start version checker
    async def version_checker():
        while True:
            await asyncio.sleep(3600)  # Check every hour
            try:
                current_versions = set(model_sessions.keys())
                available_versions = set(get_model_versions())
                new_versions = available_versions - current_versions
                if new_versions:
                    logger.info(f"New versions detected: {new_versions}")
            except Exception as e:
                logger.warning(f"Version check failed: {str(e)}")
    
    asyncio.create_task(version_checker())

class ChurnRequest(BaseModel):
    features: List[float] = Field(..., min_items=19, max_items=19)

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0,0,1,0,45,1,0,2,0,0,0,0,0,0,1,1,2,89.85,4034.45]
            }
        }

# Initialize performance metrics
performance_metrics = {
    "serverless": {
        "response_times": [],
        "instance_id": str(uuid.uuid4())
    },
    "non_serverless": {
        "response_times": [],
        "instance_id": str(uuid.uuid4())
    }
}

@app.middleware("http")
async def track_performance(request: Request, call_next):
    start_time = time.time()
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
    return {
        "message": "Churn Prediction API 🚀",
        "mode": "serverless" if os.getenv("RAILWAY_ENVIRONMENT") else "non-serverless",
        "available_models": list(model_sessions.keys()),
        "storage_status": {
            "max_size_mb": MAX_TOTAL_SIZE_MB,
            "current_usage_mb": sum(m["size_mb"] for m in model_sessions.values()),
            "model_count": len(model_sessions)
        }
    }

@app.post("/predict/{version}")
async def predict_with_version(
    version: str,
    request_data: ChurnRequest,
    request: Request
):
    try:
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
            "model_last_updated": model_sessions[version]["last_updated"],
            "request_id": str(uuid.uuid4())
        }

    except HTTPException:
        raise
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

@app.get("/models/status")
def get_model_status():
    """Check model status and versions"""
    return {
        "available_models": list(model_sessions.keys()),
        "model_details": {
            v: {
                "last_updated": m["last_updated"],
                "size_mb": m["size_mb"],
                "input_shape": m["session"].get_inputs()[0].shape
            } for v, m in model_sessions.items()
        },
        "storage_status": {
            "max_size_mb": MAX_TOTAL_SIZE_MB,
            "current_usage_mb": sum(m["size_mb"] for m in model_sessions.values()),
            "model_count": len(model_sessions)
        },
        "huggingface_repo": REPO_ID,
        "next_refresh": f"Every {REFRESH_HOURS} hours"
    }

@app.post("/models/refresh")
async def refresh_models_endpoint(force: bool = True):
    """Manually trigger model refresh"""
    try:
        initialize_models(force_refresh=force)
        return {
            "status": "success",
            "message": "Models refreshed successfully",
            "available_models": list(model_sessions.keys()),
            "storage_usage": f"{sum(m['size_mb'] for m in model_sessions.values()):.2f}MB"
        }
    except Exception as e:
        logger.error(f"Model refresh failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Model refresh failed",
                "details": str(e),
                "current_models": list(model_sessions.keys())
            }
        )

@app.get("/performance")
def get_performance():
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