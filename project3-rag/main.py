import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Import internal modules
from ingestion import intialize_ingestion
from pipeline import generate_answer_with_monitoring


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Initializes heavy ML models and DB connections once.
    """
    print("Initializing RAG artifacts (Models & ChromaDB)")
    # Store initialized components in app.state for access in routes
    app.state.artifacts = intialize_ingestion(
        file_path="10-K 2023.pdf", 
        company_name="Apple Inc.",
        fiscal_year="FY2023"
    )
    print("Initialization complete. Server ready.")
    yield
    # Clean up resources on shutdown
    app.state.artifacts.clear()
    print(" Resources released.")

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Financial RAG API",
    lifespan=lifespan
)

# ENdpoints 
@app.get("/health")
async def health_check():
    """Standard liveness probe."""
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus telemetry scrape point."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Handles queries by offloading to a ThreadPool.
    Passes app.state.artifacts to the pipeline logic.
    """
    loop = asyncio.get_event_loop()
    try:
        # Offload CPU-bound task to thread pool
        result = await loop.run_in_executor(
            None, 
            generate_answer_with_monitoring, 
            request.query,
            app.state.artifacts
        )
        return result
    except Exception as e:
        print(f"❌ API failure on query '{request.query}': {str(e)}")
        raise HTTPException(status_code=500, detail="Internal RAG Error")

if __name__ == "__main__":
    # Start the server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")