import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import mlflow
import mlflow.sklearn

# ----- Simulate CPU usage data -----
np.random.seed(42)
cpu_usage = np.random.normal(loc=50, scale=10, size=1000)  # normal usage
cpu_usage = np.append(cpu_usage, [100, 5, 120])  # anomalies

df = pd.DataFrame({"cpu": cpu_usage})

# ----- MLflow experiment -----
mlflow.set_experiment("server-monitor-anomaly")

with mlflow.start_run():
    # Train model
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(df[["cpu"]])

    # Predict anomalies (-1 = anomaly, 1 = normal)
    df["prediction"] = model.predict(df[["cpu"]])
    anomalies = df[df["prediction"] == -1]

    # Log parameters and metrics
    mlflow.log_param("model_type", "IsolationForest")
    mlflow.log_param("contamination", 0.01)
    mlflow.log_metric("num_anomalies", len(anomalies))
    mlflow.log_metric("total_points", len(df))

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model")

    # Save locally with joblib
    joblib.dump(model, "model.joblib")

    print(f"Model saved. Anomalies found: {len(anomalies)}")
    print(anomalies.head())  # preview some anomalies
