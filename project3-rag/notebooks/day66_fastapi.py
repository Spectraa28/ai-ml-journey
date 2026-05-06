#!/usr/bin/env python
# coding: utf-8

# In[16]:


# imports 
import os
import time 
from typing import List,Dict

import uvicorn
from fastapi import FastAPI , HTTPException , Response
from pydantic import BaseModel

import mlflow 
from prometheus_client import (
     Counter,
    Histogram,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST
)

import asyncio
from groq import Groq
from sentence_transformers import SentenceTransformer 
import chroma_db
from rank_bm25 import BM25Okapi
from day65_generation_monitoring import generate_answer_with_monitoring


# # Pydantic models (api schema )

# In[2]:


class QueryRequest(BaseModel):
    query : str


# In[4]:


class QueryResponse(BaseModel):
    answer: str
    sources : List[Dict]
    latency_ms : float
    prompt_tokens: int
    completion_tokens: int


# # Fast APi

# In[5]:


app = FastAPI (
    title="Financial RAG API",
    description="Production-grade API for querying Apple 10-K financial data. Includes integrated Prometheus monitoring and MLflow tracking.",
    version="1.0.0"
)


# In[6]:


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# In[17]:


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, generate_answer_with_monitoring, request.query
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))


# In[18]:


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# In[20]:


if __name__ == "__main__":
    
    
    print("\n" + "="*50)
    print("🔥 FINANCIAL RAG API: ONLINE")
    print("="*50)
    print(f"📍 Local Interface:  http://127.0.0.1:8000")
    print(f"💓 Health Check:     http://127.0.0.1:8000/health")
    print(f"📊 Metrics (Prom):   http://127.0.0.1:8000/metrics")
    print(f"📄 Interactive Docs: http://127.0.0.1:8000/docs")
    print("="*50 + "\n")

    
    uvicorn.run(
        "day66_fastapi:app", 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        reload=False
    )


# In[ ]:




