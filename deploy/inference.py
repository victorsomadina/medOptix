from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from pathlib import Path
import json
from datetime import date, datetime

app = FastAPI(title="MedOptix API")

MODEL = None
FEATURE_SCHEMA = None


@app.on_event("startup")
def load_artifacts():
    global MODEL, FEATURE_SCHEMA

    model_path = Path("../model/sarimax_model.pkl")
    schema_path = Path("../model/sarimax_schema.json")

    if model_path.exists():
        with open(model_path, "rb") as f:
            MODEL = pickle.load(f)
        print("Model loaded successfully")
    else:
        print("Model file not found")

    if schema_path.exists():
        with open(schema_path, "r") as f:
            FEATURE_SCHEMA = json.load(f)
        print("Schema loaded successfully")
    else:
        print("Schema file not found")


class PredictRequest(BaseModel):
    hospital: str = "Helsinki Central Hospital"
    ward: str = "ED"
    age: float = 54.89
    previous_day_occupancy: float = 34.0
    previous_day_overflow: float = 42.0
    previous_day_avg_wait: float = 227.0
    arrival_source: str = "self"
    outcome: str = "discharged"
    sex: str = "M"
    base_beds: int = 30
    effective_capacity: int = 34
    staffing_index: float = 0.927
    previous_day_discharges: float = 33.0
    previous_day_admission_rate_per_bed: float = 2.233

    steps: int = 1
    start_date: date | None = None


@app.get("/")
def root():
    return {"message": "MedOptix API", "status": "running"}


@app.post("/predict")
def predict(request: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if FEATURE_SCHEMA is None:
        raise HTTPException(status_code=503, detail="Schema not loaded")

    data = request.dict()
    steps = data.pop("steps")
    start_date = data.pop("start_date", None)

    exog_df = pd.DataFrame([data] * steps)
    exog_df = exog_df.reindex(columns=FEATURE_SCHEMA, fill_value=0)

    result = MODEL.get_forecast(steps=steps, exog=exog_df)
    forecast = result.predicted_mean.tolist()

    if start_date is not None:
        forecast_dates = pd.date_range(
            start=start_date, periods=steps, freq="D"
        ).strftime("%Y-%m-%d").tolist()
    else:
        try:
            forecast_dates = [str(idx) for idx in result.predicted_mean.index]
        except Exception:
            forecast_dates = None

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "steps": steps,
        "forecast_dates": forecast_dates,
        "forecast": forecast,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
