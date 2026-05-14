import os
import time
import mlflow
import google.generativeai as genai
from prometheus_client import Counter, Histogram, Gauge

# Local Module Imports
from ingestion import initialize_ingestion
from retrieval import expand_financial_query, retrieve_with_citations, citation_formatted_answer
# --- MLflow Setup ---
mlflow.set_experiment("Financial_RAG_Production")

# --- Prometheus Metrics Definitions ---
try:
    QUERY_COUNT = Counter("rag_queries_total", "Total number of RAG queries processed")
    FAILURE_COUNT = Counter("rag_failures_total", "Total number of failed RAG queries")
    ACTIVE_REQUESTS = Gauge("rag_active_requests", "Number of queries currently being processed")
    LATENCY_HISTOGRAM = Histogram("rag_latency_seconds", "Latency of the entire RAG pipeline in seconds")
except ValueError:
    # Handles cases where metrics are already defined (common in interactive shells)
    pass

# --- Global Pipeline State ---
# Initialized once at module load
# ARTIFACTS = initialize_ingestion(
#     file_path="apple_10k_2023.htm", 
#     company_name="Apple Inc."
# )

client = genai.configure(api_key="####")
def generate_answer(query: str, artifacts: dict):
    """
    Accepts artifacts explicitly from the caller (FastAPI state).
    """
    expanded_query = expand_financial_query(query)
    
    retrieved_docs = retrieve_with_citations(
        query=expanded_query,
        bm25_index=artifacts["bm25"],        # <--- Use passed artifacts
        documents=artifacts["enriched_texts"],
        chunks=artifacts["chunks"],
        collection=artifacts["collection"]
    )
    
    final_prompt = citation_formatted_answer(query, retrieved_docs)
    
    model = genai.GenerativeModel('gemini-3.1-flash-lite') # or 'gemini-3-flash'
    
    # We set temperature to 0.0 for deterministic financial auditing
    response = model.generate_content(
        final_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
        )
    )
    
    # 5. Extract Text and Token Usage
    # Note: Gemini uses 'candidates' and 'usage_metadata'
    answer_text = response.text
    usage = {
        "prompt_tokens": response.usage_metadata.prompt_token_count,
        "completion_tokens": response.usage_metadata.candidates_token_count,
        "total_tokens": response.usage_metadata.total_token_count
    }
    
    return {
        "answer": answer_text,
        "sources": retrieved_docs,
        "usage": usage
    }

def generate_answer_with_monitoring(query: str, artifacts: dict):
    """
    Orchestrator that receives state from the API layer.
    """
    start_time = time.time()
    QUERY_COUNT.inc()
    ACTIVE_REQUESTS.inc()
    
    with mlflow.start_run(run_name=f"Query: {query[:30]}"):
        try:
            # Pass artifacts down the chain
            result = generate_answer(query, artifacts)
            
            latency_ms = (time.time() - start_time) * 1000
            LATENCY_HISTOGRAM.observe(latency_ms / 1000)
            
            mlflow.log_metric("latency_ms", latency_ms)
            
            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": result["usage"].prompt_tokens,
                "completion_tokens": result["usage"].completion_tokens
            }
        except Exception as e:
            FAILURE_COUNT.inc()
            raise e
        finally:
            ACTIVE_REQUESTS.dec()