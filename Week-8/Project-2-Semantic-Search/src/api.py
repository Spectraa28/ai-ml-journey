from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
import chromadb
from fastapi.responses import Response
from prometheus_client import Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST
import time

app = FastAPI(title="CFPB Compalint Search API")

# Load the model and DB once at startup 
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection("cfpb_complaints")

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
        
@app.get("/health")
def health():
    return {"status": "ok", "collection_size": collection.count()}