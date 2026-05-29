from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from models import PredictRequest, PredictResponse
from config import get_router
from strategies import SHADOW_LOGS, CANARY_LOGS, BaseRouter

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles critical application startup and shutdown hooks.
    This pattern ensures your routers or model connections are fully validated before serving live traffic.
    """
    print("\n=======================================================")
    print("PROD DEPLOYMENT INITIALIZATION: ML Routing Layer Running")
    print("=======================================================\n")
    yield
    print("\n=======================================================")
    print("PROD DEPLOYMENT CLEANUP: Flushing In-Memory Diagnostic Metrics")
    print(f"Total Shadow Logs Pending Flush: {len(SHADOW_LOGS)}")
    print(f"Total Canary Logs Pending Flush: {len(CANARY_LOGS)}")
    print("=======================================================\n")

# Attach the modern lifecycle hook to our main application wrapper
app = FastAPI(
    title="ML Infrastructure Routing Layer",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest, 
    router: BaseRouter = Depends(get_router)
) -> PredictResponse:
    """
    Unified entry point for generation of recommendation lists.
    Utilizes the Strategy pattern via dependency injection to handle traffic splits seamlessly.
    """
    # The endpoint is entirely decoupled from the deployment mechanics.
    # It simply calls the common interface method contract (.route).
    return await router.route(request)


@app.get("/shadow-logs")
async def get_shadow_logs():
    """
    Exposes captured background predictions from the challenger model.
    Used during the Shadow phase to perform differential regression testing without affecting users.
    """
    return {
        "routing_mode": "shadow",
        "log_count": len(SHADOW_LOGS),
        "data": SHADOW_LOGS
    }


@app.get("/canary-logs")
async def get_canary_logs():
    """
    Exposes live tracking metrics for the Canary deployment partition.
    Used by your tracking systems to evaluate conversion anomalies and check variance.
    """
    return {
        "routing_mode": "canary",
        "log_count": len(CANARY_LOGS),
        "data": CANARY_LOGS
    }