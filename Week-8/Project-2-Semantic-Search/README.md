# Semantic Search Engine — CFPB Financial Complaints

Helpdesks don't fail because agents are slow. They fail because search is 
keyword-based — it matches words, not meaning.

This engine indexes 25,000 real consumer financial complaints and finds 
semantically similar cases using transformer embeddings. Search "my bank 
charged me twice" and it surfaces relevant past complaints — even if none 
of them contain those exact words.

Built to production standards: A/B testing framework to compare embedding 
models live, MLflow experiment tracking, Prometheus monitoring, and drift 
detection on the query embedding space.

**Dataset:** 25,000 sampled from 500K+ CFPB complaints  
**Search quality:** 0.645 similarity score on mortgage queries  
**Throughput:** 37 sentences/second (MiniLM)  
**Models compared:** MiniLM vs MPNet via 90/10 traffic split  

---

## Live Demo

🔗 **API:** `https://semantic-search-v1-2.onrender.com`  
📖 **Interactive Docs:** `https://semantic-search-v1-2.onrender.com/docs`

> ⚠️ Hosted on Render free tier. First request after inactivity triggers 
> a ~10 minute re-index. If the API returns `"status": "indexing"`, wait 
> and retry.

**Demo queries to try:**
- `?query=mortgage+late+fees`
- `?query=credit+card+fraud`
- `?query=student+loan+payment+problems`
- `?query=bank+account+closed+without+notice`

---

## Quick Start

```bash
docker pull spectra2204/semantic-search:v1.1
docker run -p 8000:8000 spectra2204/semantic-search:v1.1
```

> ⚠️ First run takes ~10 minutes to index 25,000 complaints into ChromaDB.  
> Watch the logs — once you see `✅ Ingestion complete`, the API is ready.

Visit `http://localhost:8000/docs` to explore the API interactively.

---

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| Embeddings | SentenceTransformers (MiniLM) | 6x faster than MPNet with similar accuracy |
| Vector Store | ChromaDB | Persistent vector storage with metadata filtering |
| API | FastAPI | Best-in-class Python backend with auto docs |
| Experiment Tracking | MLflow | Model registry + UI for comparing runs |
| Monitoring | Prometheus | Per-model latency and similarity score tracking |
| A/B Routing | MD5 hash % 100 | Deterministic — same query always hits same model |

---

## API Endpoints

### `GET /search?query=...&n_results=5&product=...`
Semantic search using MiniLM. Returns ranked complaints matching query intent.

```json
{
  "query": "mortgage late fees",
  "results": [{
    "score": 0.645,
    "product": "Mortgage",
    "issue": "Loan servicing, payments, escrow account",
    "text": "my mortgage company charge me late fee for the last nine months..."
  }]
}
```

### `GET /search/ab?query=...`
Same search with 90/10 traffic split — 10% routed to MPNet for model 
comparison. Response includes `model_version` field showing which model 
served the request.

### `GET /health`
Returns API status and whether 25,000 complaints are indexed and ready.

### `GET /metrics`
Prometheus metrics — search latency, request counts, per-model similarity 
scores.

---

## Production Decisions

**MiniLM over MPNet** — Both models returned similar accuracy, but MiniLM 
processes queries 6x faster, making it the right default for production latency.

**Re-index on startup over persistent volume** — Cloud volume setup adds 
infrastructure complexity with minimal benefit for a 25K dataset. Re-indexing 
on cold start is rare and keeps the stack simple.

**Hash-based routing over random** — MD5 hashing ensures the same query 
always routes to the same model, making A/B comparison meaningful and 
reproducible.

---

## Project Structure

```
Project-2-Semantic-Search/
├── data/
│   └── complaints_clean.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_embeddings.ipynb
├── src/
│   ├── api.py
│   └── startup.py
├── Dockerfile
├── .dockerignore
└── requirements.txt
```

---

## Related Projects

- **Project 1 — Fraud Detection API:** XGBoost + FastAPI + Docker + Railway  
  🔗 https://fraud-ml-pipeline.onrender.com/docs  
  📦 github.com/Spectraa28/fraud-ml-pipeline
