import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import mlflow

# Simulate CPU usage data
np.random.seed(42)
cpu_usage = np.random.normal(loc=50, scale=10, size=1000)  # normal usage
cpu_usage = np.append(cpu_usage, [100, 5, 120])  # anomalies

df = pd.DataFrame({"cpu": cpu_usage})

mlflow.set_experiment("server-monitor-anomaly")

with mlflow.start_run():
    model = IsolationForest(contamination=0.01)
    model.fit(df[["cpu"]])
    
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_param("model_type", "IsolationForest")
    joblib.dump(model, "model.joblib")
    print("Model saved.")
