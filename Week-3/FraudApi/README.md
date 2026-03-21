# fraud-ml-pipeline

XGBoost fraud detection model served as a production REST API — FastAPI + Docker + Railway

🔴 **Live API:** https://fraud-api-production-3d05.up.railway.app/health  
📊 **Swagger UI:** https://fraud-api-production-3d05.up.railway.app/docs

---

## Problem Statement

Credit card fraud costs financial institutions billions annually. The challenge is detecting fraud in real time — every transaction needs a decision in milliseconds, and the cost of missing fraud is far higher than the cost of a false alarm.

This project trains a fraud detection model on 284,807 real transactions (492 frauds, 0.173% fraud rate) and serves it as a live REST API with monitoring, input validation, and confidence scoring.

---

## Approach

**Why XGBoost?**  
After comparing 5 models, XGBoost with `scale_pos_weight=578` was selected because it achieved the best balance of precision and recall on severely imbalanced data (F1 = 0.859 ± 0.007 on cross-validation).

**Why scale_pos_weight=578?**  
The dataset has 578 normal transactions for every fraud. This parameter tells XGBoost to weight fraud cases 578x more heavily — forcing it to learn fraud patterns despite the imbalance.

**Why FastAPI?**  
Automatic Swagger UI, Pydantic validation, and async support out of the box. Zero configuration for a production-ready API.

---

## What I Tried That Failed

| Model | Why Rejected |
|---|---|
| Baseline LR | Missed 35 fraud cases — unacceptable production risk |
| Weighted LR | Generated 1,417 false alarms — operationally impossible |
| Random Forest | Good precision but caught 9 fewer frauds than XGBoost |
| XGBoost + SMOTE | Trained on 227,057 synthetic frauds — F1 dropped from 0.859 to 0.800, fake patterns may not generalize |

---

## Architecture

```
Client Request (30 features)
        ↓
FastAPI /predict endpoint
        ↓
Input Validation (Pydantic — must be exactly 30 floats)
        ↓
Feature Scaling (StandardScaler — Amount + Time only)
        ↓
XGBoost Model (scale_pos_weight=578)
        ↓
Response: fraud + probability + confidence + v14_alert
        ↓
Logger (prediction result + latency)
```

---

## API Endpoints

### POST /predict
Takes 30 transaction features, returns fraud prediction.

**Request:**
```json
{
  "features": [0.0, -1.35, -0.07, ...] // 30 floats: Time + V1-V28 + Amount
}
```

**Response:**
```json
{
  "fraud": true,
  "probability": 0.9999,
  "confidence": "high",
  "v14_alert": true
}
```

### GET /health
Tests the model is loaded and working — not just a status check.
```json
{
  "status": "ok",
  "model": "loaded"
}
```

---

## Performance Metrics

| Metric | Value |
|---|---|
| F1 Score | 0.859 ± 0.007 (CV) |
| API Latency | ~3ms per prediction |
| Fraud Rate in Dataset | 0.173% (492 / 284,807) |
| False Alarm Rate (Weighted LR, rejected) | 1,417 |
| V14 Feature Importance | ~60% of XGBoost decisions |

---

## Monitoring

**V14 Alert:** V14 is the most important feature, driving ~60% of XGBoost decisions. In fraud cases, V14 mean = -6.9 vs normal mean ≈ 0. Any transaction with `|V14| > 2.88` (3 standard deviations) triggers `v14_alert: true` — flagging it for human review regardless of model prediction.

This guards against the model missing fraud with high confidence — the most dangerous failure mode.

---

## Cost Analysis

| Item | Cost |
|---|---|
| Railway hosting | ~$5/month |
| Inference cost | $0 (XGBoost, no API fees) |
| Per prediction | ~$0.000017 at 1000 req/day |

---

## What I'd Improve With More Time

1. **Pin sklearn version** — scaler was saved with sklearn 1.7.2, container runs 1.8.0. Causes a warning on every request. Fix: retrain and save scaler with same version as container.
2. **Leaner Docker image** — XGBoost pulled nvidia-nccl-cu12 (293MB GPU library) unnecessarily. Fix: pin `xgboost==3.2.0 --no-deps` and install only required dependencies.
3. **Request logging to database** — currently logs to stdout only. Production system should persist logs for drift analysis over time.
4. **Threshold configuration** — fraud threshold (0.5) is hardcoded. Should be configurable per business context — a bank tolerates fewer false alarms than a startup.
5. **GitHub Actions CI/CD** — auto-rebuild and redeploy on every push to main.

---

## Stack

- **Model:** XGBoost (scikit-learn 1.8.0, xgboost 3.2.0)
- **API:** FastAPI 0.135.1 + Uvicorn 0.41.0
- **Validation:** Pydantic 2.12.4
- **Containerization:** Docker
- **Deployment:** Railway
- **Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)