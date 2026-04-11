from fastapi import FastAPI
from prometheus_client import Counter,Histogram, Gauge, make_asgi_app
import time
import random

app = FastAPI()

#COunter only goes up never down 
# Tracks total number of request 
REQUEST_COUNT = Counter(
    'fraud_api_requests_total', # Metric name
    'Total number of prediction requests', # description 
    ['status'] # label success or error 
)

# Histogram tracks distribution of values
# Perfect for latency tells you p50, p90, p99
PREDICTION_LATENCY = Histogram(
    'fraud_prediction_latency_seconds',
    'Time taken for each prediction',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]  # latency buckets in seconds
)

# Gauge  can go up or down
# Tracks current fraud prediction rate
FRAUD_RATE = Gauge(
    'fraud_prediction_rate',
    'Current rate of fraud predictions'
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/predict")
def predict():
    start_time = time.time()  # start timer
    
    try:
        # Simulate prediction  replace with real model later
        time.sleep(random.uniform(0.001, 0.01))  # simulate 1-10ms latency
        is_fraud = random.random() < 0.03         # 3% fraud rate
        
        # Record metrics
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)        # record this prediction's latency
        REQUEST_COUNT.labels(status='success').inc()  # increment success counter
        FRAUD_RATE.set(0.03 if is_fraud else 0.0)    # update fraud rate gauge
        
        return {"fraud": is_fraud, "latency_ms": latency * 1000}
    
    except Exception as e:
        REQUEST_COUNT.labels(status='error').inc()  # increment error counter
        raise e

@app.get("/health")
def health():
    return {"status": "ok"}


