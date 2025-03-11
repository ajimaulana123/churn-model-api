from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import os
import uvicorn
import onnxruntime as ort

# Load model ONNX
onnx_model_path = "model/random_forest_churn.onnx"
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name  # Ambil nama input dari model ONNX

# Inisialisasi FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="API buat prediksi pelanggan bakal cabut atau nggak",
    version="1.0"
)

# Struktur request dari user
class ChurnRequest(BaseModel):
    features: list = Field(
        description="List dari 19 fitur pelanggan",
        min_items=19,
        max_items=19
    )

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0, 0, 1, 0, 45, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 2, 89.85, 4034.45]
            }
        }

# Endpoint Home
@app.get("/")
def home():
    return {"message": "Welcome to Churn Prediction API 🚀"}

# Endpoint buat Prediksi
@app.post("/predict")
def predict_churn(request: ChurnRequest):
    try:
        # Ubah input ke format numpy array (float32)
        input_data = np.array(request.features, dtype=np.float32).reshape(1, -1)

        # Jalankan prediksi
        output = session.run(None, {input_name: input_data})
        print("Output model:", output)

        # Output model berbentuk: [array([0]), {0: 0.78, 1: 0.22}]
        pred_label = int(output[0][0])  # Hasil: 0 = Gak Churn, 1 = Churn
        probabilities = output[1]  # Probabilitas hasil prediksi
        
        # Ambil probabilitas berdasarkan kelas prediksi
        prob = probabilities[pred_label] if pred_label in probabilities else 0.0

        # Format response
        return {
            "prediction": pred_label,
            "probability": round(prob, 4),  # Buat lebih rapi
            "status": "Churn" if pred_label == 1 else "Gak Churn"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Jalankan API
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Ambil PORT dari Railway
    uvicorn.run(app, host="0.0.0.0", port=port)
