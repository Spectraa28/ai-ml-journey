import os 
from strategies import BaseRouter, ShadowRouter,CanaryRouter

ROUTING_MODE = os.getenv("ROUTING_MODE","shadow").lower().strip()
CANARY_WEIGHT = float(os.getenv("CANARY_WEIGHT","0.05"))

if ROUTING_MODE =="shadow":
    print("LOG: Initializing routing layer in [SHADOW] mode. 100% traffic to Model A")
    _active_router = ShadowRouter()
elif ROUTING_MODE== "canary":
    print(f"LOG: Initializing routing layer in [CANARY] mode. {CANARY_WEIGHT * 100}% traffic")
    _active_router = CanaryRouter(canary_weight=CANARY_WEIGHT)
else:
    raise ValueError(
        f"Invalid ROUTING_MODE '{ROUTING_MODE}' allowed values are 'shadow' or 'canary' "
    )
    

def get_router()-> BaseRouter:
    """
    reutrns the gloabbally configured active routing strategy.
    INvoked by the fastAPI presentation layer to abstract away routing mechanics
    """
    
    return _active_router