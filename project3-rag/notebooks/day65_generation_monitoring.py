#!/usr/bin/env python
# coding: utf-8

# # Day 65 - Llm generationn & Monitoring using mlflow and prometheus 
#  ## what we will have after today : we will be able to generate response from the llm and track the different parameter using mlflow 

# # IMPorts

# In[95]:


import time
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq
import mlflow
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST ,REGISTRY
from day64_rag_pipeline_clean import expand_financial_query, retreive_with_citations , enriched_texts, chunks, citation_formatted_answer


# # Prometheus parameter

# In[96]:
try:
    rag_requests_total = Counter("rag_requests_total", "Total RAG API requests", ["endpoint", "status_code"])
except ValueError:
    rag_requests_total = REGISTRY._names_to_collectors["rag_requests_total"]

try:
    pipeline_duration_seconds = Histogram("pipeline_duration_seconds", "Latency of the RAG pipeline in seconds", ["endpoint"], buckets=[0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
except ValueError:
    pipeline_duration_seconds = REGISTRY._names_to_collectors["pipeline_duration_seconds"]

try:
    active_processing_queries = Gauge("active_processing_queries", "Number of currently processing RAG queries")
except ValueError:
    active_processing_queries = REGISTRY._names_to_collectors["active_processing_queries"]


# # MlFlow 

# In[99]:


import os


# In[100]:


chroma_client = chromadb.PersistentClient(path="./chroma_db/")
collection = chroma_client.get_collection(name="financial_docs_v2")
count = collection.count()


if 'collection' in locals() and collection.count() > 0:
    db_data = collection.get(include=['documents', 'metadatas'])
    
    chunks = []
    for id_, text, meta in zip(db_data['ids'], db_data['documents'], db_data['metadatas']):
        sanitized_chunk = {
            "chunk_id": id_,
            "text": text,
            "company": meta.get("company", "Unknown Company"),
            "doc_type": meta.get("doc_type", "Document"),
            "fiscal_year": meta.get("fiscal_year", "N/A"),
            "section_title": meta.get("section_title", "General Section"),
            "metadata": meta # Keep for reference
        }
        
        for key, value in meta.items():
            if key not in sanitized_chunk:
                sanitized_chunk[key] = value
                
        chunks.append(sanitized_chunk)
    
    enriched_texts = db_data['documents']
    
    # Re-initialize BM25 with the cleaned texts
    tokenized_corpus = [doc.split(" ") for doc in enriched_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    print(f"✅ Synced {len(chunks)} chunks. Defensive defaults applied to all keys.")


# In[105]:


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# In[106]:


API_KEY = "####"


# In[107]:


groq_client = Groq(api_key=API_KEY)


# In[108]:


mlflow.set_experiment("financial_rag_pipeline")


# In[109]:


print("✅ Clients initialized and MLflow experiment set.")


# # Generate answer

# In[110]:


def generate_answer(query: str) -> dict:
    """
    Orchestrates the full RAG pipeline: Expansion -> Retrieval -> Guardrail -> Generation.
    """
    start_time = time.time()
    
    # Query Expansion Prepare the query for numeric/tabular search
    expanded_query = expand_financial_query(query)
    chunk_texts = [chunk['text'] for chunk in chunks]
    tokenized_corpus = [doc.split(" ") for doc in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Hybrid Retrieval Fetch relevant context from ChromaDB and BM25
    retrieved_results = retreive_with_citations(
        query=expanded_query, 
        bm25_index=bm25, 
        documents=enriched_texts, 
        chunks=chunks, 
        collection=collection,
        n_results=3,
        alpha=0.7
    )
    
    #  Guardrail Short-circuit if no relevant context is found
    if not retrieved_results:
        execution_time = (time.time() - start_time) * 1000
        return {
            "answer": "I'm sorry, I couldn't find any relevant financial data in the current documents to answer that.",
            "sources": [],
            "latency_ms": round(execution_time, 2)
        }
    
    # Prompt Assembly Using your Day 64 formatting function[cite: 1]
    # This transforms the results into the strict "[Source 1]" prompt format.
    prompt = citation_formatted_answer(query, retrieved_results)
    
    #  LLM Inference Groq execution with your assembled prompt
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a professional financial analyst. Use only the provided context."
            },
            {
                "role": "user",
                "content": prompt, 
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.1, 
        max_tokens=1024,
    )

    p_tokens = chat_completion.usage.prompt_tokens
    c_tokens = chat_completion.usage.completion_tokens
    
    answer = chat_completion.choices[0].message.content
    execution_time = (time.time() - start_time) * 1000
    
    # 6. Packaging Return structured response for the API
    return {
        "answer": answer,
        "sources": [
            {
                "chunk_id": r["chunk_id"],
                "citation": r["citation"],
                "hybrid_score": r["hybrid_score"]
            } for r in retrieved_results
        ],
        "latency_ms": round(execution_time, 2),
        "prompt_tokens": p_tokens,
        "completion_tokens": c_tokens
    }


# # Monitoring the geneation 

# In[111]:


def generate_answer_with_monitoring(query: str) -> dict:
    active_processing_queries.inc()
    start_time = time.time()

    try:
        result = generate_answer(query)

        latency_sec = time.time() - start_time
        latency_ms = latency_sec * 1000

        sources = result.get("sources",[])
        top_hybrid_score = max([s["hybrid_score"] for s in sources]) if sources else 0.0

        # cost estimation 
        prompt_t = result.get("prompt_tokens",0)
        completion_t = result.get("completion_tokens",0)
        cost_estimate  = (prompt_t + completion_t) * 0.000030

        # Mlflow audit 
        with mlflow.start_run(run_name="rag_request"):
            mlflow.log_param("query",query)
            mlflow.log_metric("latency_ms",latency_ms)
            mlflow.log_metric("top_hybrid_score",top_hybrid_score)
            mlflow.log_metric("cost_estimate_usd",cost_estimate)

            mlflow.log_param("retrieved_chunk_ids",[s["chunk_id"] for s in sources])

        pipeline_duration_seconds.labels(endpoint="generate_answer").observe(latency_sec)
        rag_requests_total.labels(endpoint="generate_answer",status_code="success").inc()

        return result

    except Exception as e:
        rag_requests_total.labels(endpoint="generate_answer",status_code="error").inc()
        print(f"❌ Critical error in monitored pipeline: {str(e)}")
        raise e 

    finally:
        active_processing_queries.dec()

# In[75]:


# TODO 
"""
Current output: "The revenue of Apple in fiscal year 2023 was 383,285."
Target output: "The total net sales of Apple Inc. for fiscal year 2023 (ended September 30, 2023) was $383,285 million."
"""
# In[ ]:




