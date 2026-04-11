import hashlib
import random
import time
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, make_asgi_app

app = FastAPI()

# Track metrics separately per model version
REQUEST_COUNT = Counter(
    'ab_requests_total',
    'Total requests per model version',
    ['model_version', 'status']
)

PREDICTION_LATENCY = Histogram(
    'ab_prediction_latency_seconds',
    'Latency per model version',
    ['model_version'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
)

FRAUD_RATE = Counter(
    'ab_fraud_predictions_total',
    'Fraud predictions per model version',
    ['model_version']
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

def get_model_version(user_id: str, traffic_split: int = 10) -> str:
    # Hash user_id to get consistent assignment
    # Same user always gets same model
    hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    if hash_value % 100 < traffic_split:
        return "model_b"  # 10% of users
    return "model_a"      # 90% of users

def model_a_predict(features: dict) -> dict:
    # Simulate Model A — original fraud model F1=0.859
    time.sleep(random.uniform(0.001, 0.005))  # 1-5ms latency
    is_fraud = random.random() < 0.03          # 3% fraud rate
    return {"fraud": is_fraud, "model": "model_a", "confidence": 0.85}

def model_b_predict(features: dict) -> dict:
    # Simulate Model B — new model F1=0.91
    time.sleep(random.uniform(0.001, 0.008))  # slightly higher latency
    is_fraud = random.random() < 0.03
    return {"fraud": is_fraud, "model": "model_b", "confidence": 0.91}

@app.get("/predict/{user_id}")
def predict(user_id: str):
    start_time = time.time()
    
    # Determine which model this user gets
    version = get_model_version(user_id)
    
    try:
        # Route to correct model
        if version == "model_b":
            result = model_b_predict({})
        else:
            result = model_a_predict({})
        
        latency = time.time() - start_time
        
        # Log metrics per model version
        PREDICTION_LATENCY.labels(model_version=version).observe(latency)
        REQUEST_COUNT.labels(model_version=version, status='success').inc()
        if result['fraud']:
            FRAUD_RATE.labels(model_version=version).inc()
        
        return result
    
    except Exception as e:
        REQUEST_COUNT.labels(model_version=version, status='error').inc()
        raise e



for i in range(1, 200):
    user_id = f"user_{i}"
    hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    bucket = hash_value % 100
    if bucket < 10:
        print(f"{user_id}: bucket={bucket} → model_b")