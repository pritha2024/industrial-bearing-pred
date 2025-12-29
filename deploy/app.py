import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# 1. initialize the app
app = FastAPI(
    title="Bearing Health API",
    description="Industrial AI Microservice for Real-time Anomaly Detection",
    version="1.0.0"
)

# 2. define the input data format (data validation)
# this ensures users don't send garbage data
class BearingFeatures(BaseModel):
    rms: float
    kurtosis: float
    skewness: float
    peak_to_peak: float
    crest_factor: float

# 3. global variables to hold the model
model = None
scaler = None

# 4. load model on startup
@app.on_event("startup")
def load_model():
    global model, scaler
    # adjust path if you are running this from inside the deploy folder vs root
    model_path = "saved_models/iso_forest.pkl"
    scaler_path = "saved_models/iso_forest_scaler.pkl"
    
    try:
        # checking if files exist to avoid ugly crashes
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print(f"[INFO] Model loaded successfully from {model_path}")
        else:
            print(f"[WARNING] Model files not found at {model_path}. API will run but predictions will fail.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")

# 5. health check endpoint (devops engineers love this)
@app.get("/health")
def health_check():
    if model is not None:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "degraded", "model_loaded": False}

# 6. the prediction endpoint
@app.post("/predict")
def predict_anomaly(features: BearingFeatures):
    """
    Receives real-time vibration features and returns health status.
    """
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # convert input json to dataframe
    input_data = pd.DataFrame([features.dict()])
    
    # scale the data (CRITICAL: must match training scaling)
    scaled_data = scaler.transform(input_data)
    
    # predict (-1 = anomaly, 1 = normal)
    prediction = model.predict(scaled_data)[0]
    
    # get anomaly score (lower = worse)
    score = model.decision_function(scaled_data)[0]
    
    # format the output for humans
    status = "CRITICAL_FAILURE" if prediction == -1 else "NORMAL_OPERATION"
    
    return {
        "machine_status": status,
        "anomaly_score": float(score),
        "maintenance_required": bool(prediction == -1)
    }

# 7. entry point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
