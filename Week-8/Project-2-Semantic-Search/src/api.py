from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
import chromadb
from fastapi.responses import Response
from prometheus_client import Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST
import time
import hashlib
from fastapi import HTTPException
from contextlib import asynccontextmanager
from src.startup import initialize_database

collection = None  # declared at module level

is_ready = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection
    global is_ready
    is_ready = True
    collection = initialize_database()  # runs on startup
    yield
    
app = FastAPI(title="CFPB Compalint Search API", lifespan=lifespan)
# Load the model and DB once at startup 
model_a = SentenceTransformer("all-MiniLM-L6-v2")
model = model_a


model_b = SentenceTransformer("all-mpnet-base-v2")      # 10% of traffic

client = chromadb.PersistentClient(path="data/chroma_db")

# Prometheus metrices 
SEARCH_LATENCY = Histogram(
    "search_latency_seconds",
    "Time taken per search query",
    buckets=[0.01,0.05,0.1,0.5]
)

SEARCH_COUNT = Counter(
    "search_requests_total",
    "Total search requests",
    ["status"]
)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/search")
def search(
    query: str,
    n_results: int = Query(default=5,le=20),
    product:str = Query(default=None)
):
    start = time.time()
    try:
        query_embedding = model.encode([query]).tolist()
        where = {"product": product} if product else None
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
            include=["documents","distances","metadatas"]
        )
        
        hits = []
        for doc,dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        ):
            hits.append({
                "score":round(1-dist, 4),
                "product": meta["product"],
                "issue":meta["issue"],
                "text": doc[:200]
            })
            
        SEARCH_LATENCY.observe(time.time() - start)
        SEARCH_COUNT.labels(status="success").inc()
        return {"query": query, "results":hits}
    
    except Exception as e:
        SEARCH_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
       
       


 
@app.get("/health")
def health():
    return {
        "status": "ready" if is_ready else "indexing",
        "collection_size": collection.count() if collection else 0,
        "default_model": "all-MiniLM-L6-v2"
    }


# A/B metrics - track per model version 
AB_SEARCH_LATENCY = Histogram(
    "ab_search_latency_seconds",
    "Search latency per  model version",
    ["model_version"],
    buckets=[0.01,0.05,0.1,0.5,1.0,2.0]
)

AB_SEARCH_SCORE = Histogram(
    "ab_search_score",
    "Top result similarity score per model version",
    ["model_version"],
    buckets=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
)

AB_REQUEST_COUNT = Counter(
    "ab_request_total",
    "Request per model version",
    ["model_version"]
)

def get_model_version(query: str, split: int = 10) -> str:
    hash_value = int(hashlib.md5(query.encode()).hexdigest(),16)
    return "model_b" if hash_value % 100 < split else "model_a"

@app.get("/search/ab")
def search_ab(
    query: str,
    n_results: int = Query(default=5,le=20)
):
    start  = time.time()
    version = get_model_version(query)
    model = model_b if version == "model_b" else model_a
    try: 
        query_embedding = model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents","distances","metadatas"]
        )
        
        hits = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        ):
            hits.append({
                "score": round(1-dist,4),
                "product":meta["product"],
                "issue":meta["issue"],
                "text": doc[:200]
            })
            
        latency = time.time() - start
        top_score = hits[0]["score"] if hits else 0 
        
        AB_SEARCH_LATENCY.labels(model_version=version).observe(latency)
        AB_SEARCH_SCORE.labels(model_version=version).observe(top_score)
        AB_REQUEST_COUNT.labels(model_version=version).inc()

        return {
            "query": query,
            "model_version": version,
            "results": hits
        }

    except Exception as e:
        raise e