from pydantic import BaseModel , Field

class PredictRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user requesting  reccomendation")
    item_id: str = Field(..., description="The Id of the item the user is currently interacting with")
    top_k: int = Field(default=5,ge=1,le=50,description="No. of reccomended items to return")
    
class PredictResponse(BaseModel):
    recommendations: list[str] = Field(..., description="Ordered list of recommended items")
    model_version: str = Field(..., description="Identifiees which model generated the final output")
    latency_ms: float = Field(...,description="Inference and processing time in milliseconds")
    