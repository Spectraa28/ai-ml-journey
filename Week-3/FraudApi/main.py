from fastapi import FastAPI 
from fastapi import HTTPException
from pydantic import BaseModel
import numpy as  np
import logging
import joblib
import time


app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = joblib.load("fraud-api/fraud_model.pkl")

scaler = joblib.load("fraud-api/fraud_scaler.pkl")


class Transaction(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(transaction: Transaction):
    if len(transaction.features) != 30:
        raise HTTPException(status_code=400, detail= f"Expected 30 features, got {len(transaction.features)}")
    start = time.time()
    data  = np.array(transaction.features).reshape(1,-1)
    amount_time = np.array([[data[0][29], data[0][0]]])  # shape (1,2)
    scaled_amount_time = scaler.transform(amount_time)
    data[0][29] = scaled_amount_time[0][0]   # put Time back
    data[0][0] = scaled_amount_time[0][1]  # put Amount back
    v14 = data[0][14]
    v14_alert = bool(abs(v14) > 2.88)
    probability = model.predict_proba(data)[0][1]
    duration = time.time() - start
    logger.info(f"Prediction: fraud={bool(float(probability) >= 0.5)}, probability={float(probability):.4f}, v14_alert={v14_alert}, time taken : {duration}")
    return {
    "fraud": bool(float(probability) >= 0.5),
    "probability": float(probability),
    "confidence": "high" if float(probability) > 0.8 or float(probability) < 0.2 else "low",
    "v14_alert": v14_alert
    }


@app.get("/health")
def heath():
    try:
        dummy = np.array([0.0] * 30).reshape(1,-1)
        probability = model.predict_proba(dummy)[0][1]
        return {"status": "ok", "model":"loaded"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Model  error: {str(e)}")