from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.joblib")

class CPUMetric(BaseModel):
    cpu: float

@app.post("/predict")
def predict_anomaly(metric: CPUMetric):
    prediction = model.predict([[metric.cpu]])
    is_anomaly = prediction[0] == -1
    return {"cpu": metric.cpu, "anomaly": is_anomaly}
