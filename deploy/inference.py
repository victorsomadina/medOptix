from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Union

app = FastAPI(title="MedOptix API")

MODEL = None
FEATURE_SCHEMA = None

@app.on_event("startup")
def load_artifacts():
    global MODEL, FEATURE_SCHEMA
    model_path = Path("../model/sarimax_model.pkl")
    schema_path = Path("../model/sarimax_schema.json")
    
    if model_path.exists():
        with open(model_path, 'rb') as f:
            MODEL = pickle.load(f)
    else:
        print("Model file not found at:", model_path.absolute())
    
    if schema_path.exists():
        with open(schema_path, 'r') as f:
            FEATURE_SCHEMA = json.load(f)
    else:
        print("Schema file not found at:", schema_path.absolute())


class PredictRequest(BaseModel):
    steps: int = Field(default=1, ge=1, description="Number of time steps to forecast")
    features: Dict[str, Union[int, float]] = Field(
        ..., 
        description="Dictionary of feature names and their values"
    )


@app.post("/predict")
def predict(request: PredictRequest):
    """Make predictions with automatic feature reindexing"""
    
    if MODEL is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check if sarimax_model.pkl exists in ../model/"
        )
    
    if FEATURE_SCHEMA is None:
        raise HTTPException(
            status_code=503, 
            detail="Schema not loaded. Please check if sarimax_schema.json exists in ../model/"
        )
    
    try:
        exog_df = pd.DataFrame([request.features] * request.steps)
        
        exog_df = exog_df.reindex(columns=FEATURE_SCHEMA, fill_value=0)
        
        forecast = MODEL.forecast(steps=request.steps, exog=exog_df)
        
        predictions = [max(0, round(pred)) for pred in forecast.tolist()]
        
        missing_features = [f for f in FEATURE_SCHEMA if f not in request.features.keys()]
        
        return {
            "predictions": predictions,
            "steps": request.steps,
            "features_used": list(exog_df.columns),
            "features_provided": list(request.features.keys()),
            "missing_features": missing_features,
            "note": f"{len(missing_features)} features were auto-filled with 0"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
