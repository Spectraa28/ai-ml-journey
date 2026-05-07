import numpy as np
from typing import List, Dict
from rank_bm25 import BM25Okapi

def expand_financial_query(query: str) -> str:
    """
    Query expansion: bridges the vocabulary gap between natural language 
    questions and numeric financial tables/prose.
    """
    expansions = {
        "revenue": "revenue net sales total sales income",
        "profit": "profit net income earnings",
        "cash flow": "cash flow operations operating activities",
        "expenses": "expenses cost of sales operating expenses",
        "assets": "assets current assets total assets",
        "debt": "debt liabilities long-term debt",
        "earnings per share": "earnings per share EPS diluted basic",
        "net income": "net income profit earnings comprehensive income statements operations",
        "total revenue": "total revenue net sales total net sales",
    }
    
    expanded = query
    query_lower = query.lower()
    
    for term, expansion in expansions.items():
        if term in query_lower:
            expanded = f"{expanded} {expansion}"
    
    return expanded


def retrieve_with_citations(query: str, 
                             bm25_index: BM25Okapi, 
                             documents: list,
                             chunks: list,
                             collection,
                             n_results: int = 3,
                             alpha: float = 0.7) -> List[Dict]:
    """
    Performs hybrid search (Dense + Sparse) and returns results with formatted citations.
    """

    # 1. Sparse (BM25) scores
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
    
    if bm25_max > bm25_min:
        bm25_normalized = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
    else:
        bm25_normalized = bm25_scores

    # 2. Dense (Vector) scores
    dense_results = collection.query(
        query_texts=[query],
        n_results=len(documents)
    )
    
    dense_distances = np.array(dense_results["distances"][0])
    dense_similarity = 1 - dense_distances  # Convert distance to similarity
    dense_ids = dense_results["ids"][0]
    all_ids = [chunk["chunk_id"] for chunk in chunks]

    # Align dense scores with the order of the original chunks
    dense_aligned = np.zeros(len(documents))
    for i, doc_id in enumerate(all_ids):
        if doc_id in dense_ids:
            dense_idx = dense_ids.index(doc_id)
            dense_aligned[i] = dense_similarity[dense_idx]

    # 3. Hybrid Search Merging
    hybrid_scores = (alpha * dense_aligned) + ((1 - alpha) * bm25_normalized)
    top_indices = np.argsort(-hybrid_scores)[:n_results]

    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        
        # Build citation string for the LLM
        citation = (f"{chunk['company']} {chunk['doc_type']} "
                    f"{chunk['fiscal_year']} — {chunk['section_title']}")
        
        results.append({
            "text": chunk["text"],
            "citation": citation,
            "company": chunk["company"],
            "doc_type": chunk["doc_type"],
            "fiscal_year": chunk["fiscal_year"],
            "section_title": chunk["section_title"],
            "hybrid_score": round(float(hybrid_scores[idx]), 4),
            "chunk_id": chunk["chunk_id"]
        })
    
    return results

def citation_formatted_answer(query: str, retrieved_results: List[Dict]) -> str:
    """
    Constructs the final RAG prompt with numbered sources.
    """
    context_parts = []
    
    for i, result in enumerate(retrieved_results):
        context_parts.append(
            f"[Source {i+1}: {result['citation']}]\n{result['text'][:800]}"
        )
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a financial analyst assistant. Answer the question using ONLY the provided sources below.
For every fact you state, cite the source number in brackets like [Source 1].
If the sources don't contain enough information, say so explicitly.

Sources:
{context}

Question: {query}

Answer:"""
    
    return prompt