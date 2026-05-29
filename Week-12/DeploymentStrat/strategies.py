import abc
import asyncio
import random
import time 
from models import PredictRequest, PredictResponse

SHADOW_LOGS = []
CANARY_LOGS = []

async def _mock_model_a(request:PredictRequest) -> dict:
    start_time = time.perf_counter()
    await asyncio.sleep(0.02)
    latency_ms = (time.perf_counter() - start_time) * 1000
    return{
        "recommendations": [f"prod_item_{request.item_id}_{i}" for i in range(request.top_k)],
        "model_version":"Model_A",
        "latency_ms":latency_ms
    }
    
async def _mock_model_b(request: PredictRequest) -> dict:
    start_time = time.perf_counter()
    await asyncio.sleep(0.025)
    latency_ms = (time.perf_counter() - start_time) *  1000
    return {
        "recommendations": [f"challenger_item_{request.item_id}_{i}" for i in range(request.top_k)],
        "model_version":"Model_B",
        "latency_ms":latency_ms
    }
    
class BaseRouter(abc.ABC):
    """
    Abstact base class establishing the strict  runtime contract   for all routing strategies
    """
    
    @abc.abstractmethod
    async def route(self,request: PredictRequest) -> PredictResponse:
        """Processes incoming requests and determines model routing execution paths. """
        pass
# SHADOW ROUTER
class ShadowRouter(BaseRouter):
    """
    Routes 100% of user-facing traffic to the baseline MOdel A.
    Asynchronously duplicates traffic to MOdel B in the background  for zero risk validation
    """
    
    async def route(self,request:PredictRequest) -> PredictResponse:
        # Immediately await and capture prodcution model responses to minimize user facing latency 
        prod_data = await _mock_model_b(request)
        
        # SChedule model b to be in the background without blocking return statement
        asyncio.create_task(self._execute_shadow_inference(request))
         
        return PredictResponse(**prod_data)
     
    async def _execute_shadow_inference(self,request:PredictRequest) -> None:
        """Asynchronous background execution block for processing and looggin the shadow model"""
        try:
            shadow_data = await _mock_model_b(request)
            SHADOW_LOGS.append({
                  "timestamp":time.time(),
                  "user_id":request.user_id,
                  "input_item":request.item_id,
                  "predictions":shadow_data["recommendations"],
                  "latency_ms":shadow_data["latency_ms"],
                  "status":"SUCCESS"
              })
        except Exception as e:
            SHADOW_LOGS.append({
                "timestamp":time.time(),
                "user_id":request.user_id,
                "status":f"FAILED : {str(e)}"
            })


# Canary Router
class CanaryRouter(BaseRouter):
    """
    Dynamically splits live traffic between Model A and Model B 
    based on a strictly enforced probability weight.
    """
    def __init__(self, canary_weight: float = 0.05):
        self.canary_weight = canary_weight

    async def route(self, request: PredictRequest) -> PredictResponse:
        if random.random() < self.canary_weight:
            served_data = await _mock_model_b(request)
        else:
            served_data = await _mock_model_a(request)
            
        CANARY_LOGS.append({
            "timestamp": time.time(),
            "user_id": request.user_id,
            "model_version_served": served_data["model_version"],
            "latency_ms": served_data["latency_ms"]
        })
        return PredictResponse(**served_data)