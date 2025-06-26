# Server Monitor MLOps: CPU Anomaly Detection

This project implements an MLOps pipeline for detecting anomalies in server CPU usage using an Isolation Forest model. The project includes data simulation, model training, API serving, experiment tracking with MLflow, containerization with Docker, and support for Azure deployment.

## Project Structure

```
server-monitor-mlops/
├── app.py             # FastAPI app for serving predictions
├── train.py           # Model training script
├── model.joblib       # Trained model file
├── mlruns/            # MLflow experiment logs
├── requirements.txt   # Python dependencies
├── Dockerfile         # Docker image config
├── .gitignore         # Ignored files and folders
├── README.md          # Project documentation
```

## Features

- Simulates server CPU usage data with injected anomalies
- Trains an Isolation Forest model for anomaly detection
- Tracks experiments and parameters using MLflow
- Serves predictions through a REST API built with FastAPI
- Supports Docker containerization and Azure deployment

## Getting Started

### 1. Set up the environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is empty, run:

```bash
pip install fastapi uvicorn pandas scikit-learn joblib mlflow
```

### 3. Train the model

```bash
python train.py
```

This generates CPU usage data, trains the model, logs it to MLflow, and saves it as `model.joblib`.

### 4. Serve the model with FastAPI

```bash
uvicorn app:app --reload
```

Then visit:

```
http://localhost:8000/docs
```

Example input:

```json
{
  "cpu": 95
}
```

Example output:

```json
{
  "cpu": 95.0,
  "anomaly": true
}
```

### 5. Launch the MLflow UI

```bash
mlflow ui
```

Open:

```
http://localhost:5000
```

You can view training runs, parameters, and model artifacts.

## Docker Usage

### Build the Docker image

```bash
docker build -t server-monitor-api .
```

### Run the Docker container

```bash
docker run -p 8000:8000 server-monitor-api
```

Access the API at:

```
http://localhost:8000/docs
```

## Azure Deployment (Optional)

1. Push the Docker image to Azure Container Registry (ACR)
2. Create a Container Instance with `az container create`
3. Access the API via the generated DNS name

## Future Improvements

- Live CPU metric integration
- Real-time alerts for anomaly detection
- Visualization with Grafana/Prometheus
- Auto-retraining on new data
- Extend to memory, disk, and network metrics

## License

This project is for educational and demonstration purposes only.
