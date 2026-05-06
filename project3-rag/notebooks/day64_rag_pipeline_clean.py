#!/usr/bin/env python
# coding: utf-8

# # Day 64 — RAG Pipeline Clean
# ## Table-to-Natural-Language Fix + Full Ingestion Pipeline
# ### Problem solved: Revenue and net income queries broken due to poor table embedding
# ### Fix: Convert financial tables to prose before embedding

# # Imports 

# In[32]:


import re 
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from bs4 import BeautifulSoup


# In[33]:


with open("apple_10k_2023.htm", 'r', encoding='utf-8' ,errors='ignore') as f:
    html  = f.read()


# In[34]:


soup = BeautifulSoup(html,'html.parser')
for tag in soup(['script','style','meta','link']):
    tag.decompose()


# # TableToNaturalLanguage

# In[35]:


class TableToNaturalLanguage:
    def __init__(self,company):
        """
        Initializes the converter with specific document metadata.
        """
        self.company = company
        
    def is_financial_table(self,table_soup):
        """
        Analyzes a table to determine if it contains financial data 
        based on keyword presence and numerical patterns.
        """
        financial_keywords = ["net", "revenue", "income", "sales", "earnings", 
                          "profit", "loss", "assets", "cash", "cost"]
        numeric_pattern =re.compile(r'\$?[\d,]+\.?\d*')

        rows = table_soup.find_all("tr")

        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue

            first_cell_text = cells[0].get_text(strip=True).lower()
            has_keyword = any(kw in first_cell_text for kw in financial_keywords)

            has_numeric_value = False
            for cell in cells[1:]:
                cell_text = cell.get_text(strip=True)
                # We look for a cell that isn't empty and matches our financial number pattern
                if cell_text and numeric_pattern.search(cell_text):
                    has_numeric_value = True
                    break

            # If a single row contains both a keyword and a financial number, it's a winner
            if has_keyword and has_numeric_value:
                return True

        return False

    
    def convert(self, raw_html):
        soup = BeautifulSoup(raw_html, "html.parser")
        if not self.is_financial_table(soup):
            return []

        rows = soup.find_all("tr")
        col_map = {} # Temporary to find years and their order
        year_pattern = re.compile(r'20\d{2}')
        results = []

        # 1. Header Scan (Still needed to find WHICH years are in the table)
        for row in rows:
            cells = row.find_all(["td", "th"])
            for i, cell in enumerate(cells):
                text = cell.get_text(strip=True)
                if year_pattern.fullmatch(text):
                    col_map[i] = text
            if col_map: break

        if not col_map: return []
        
        # The One-Liner: Get years in the order they appear left-to-right
        years_in_order = [year for idx, year in sorted(col_map.items())]

        # 2. Data Extraction with Filtered Zip
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 2: continue
            
            metric = cells[0].get_text(strip=True)
            if not metric: continue

            numeric_values = []
            for cell in cells[1:]:
                val_text = cell.get_text(strip=True)
                # Filtering out (4), 11, $, etc. while keeping 162,560
                if re.search(r'\d{3,}', val_text):
                    numeric_values.append(val_text)
            
            
            for value, year in zip(numeric_values, years_in_order):
                sentence = f"The {metric} of {self.company} in the fiscal year {year} was {value}."
                results.append(sentence)
                
        return results


# # Extraction from htmml 

# In[36]:


def extract_sections_from_xbrl_html(soup, table_converter) -> list[dict]:
    """Semantic section extraction with font-weight detection and table parsing."""
    sections, section_num = [], 0
    current_title, current_text, current_length = "Preamble", [], 0
    all_elements = soup.find_all(['span', 'div', 'p', 'ix:nonnumeric', 'ix:nonfraction', 'table'])
    
    for elem in all_elements:
        if elem.name == 'table':
            table_prose = table_converter.convert(str(elem))
            if not table_prose: continue
            text, is_header = " ".join(table_prose), False
        else:
            if elem.find(['span', 'div']): 
                continue
            text = elem.get_text(strip=True)
            if len(text) < 5: 
                continue
            style = elem.get('style', '')
            is_header = ('font-weight:700' in style and len(text) < 150 and 
                         any(text.upper().startswith(k) for k in ['ITEM', 'PART', 'CONSOLIDATED', 'NOTE']))
        
        if is_header and current_text and current_length > 200:
            section_num += 1
            sections.append({
                "section_num": section_num, 
                "section_title": current_title, 
                "text": " ".join(current_text), 
                "char_count": current_length
            })
            current_title, current_text, current_length = text[:100], [], 0
        else:
            current_text.append(text)
            current_length += len(text)
            if current_length > 1500:
                section_num += 1
                sections.append({
                    "section_num": section_num, 
                    "section_title": current_title, 
                    "text": " ".join(current_text), 
                    "char_count": current_length
                })
                current_text, current_length = [], 0
    
    if current_text:
        sections.append({
            "section_num": section_num + 1, 
            "section_title": current_title, 
            "text": " ".join(current_text), 
            "char_count": current_length
        })
    return sections


# In[37]:


def build_chunks_with_citations(sections, company, doc_type, fiscal_year, source_file):
    """Quality filtering and metadata assignment."""
    financial_signals = ["revenue", "income", "profit", "sales", "earnings", "cash", "assets", "net", "total"]
    chunks = []
    for section in sections:
        if 'http://' in section["text"] or 'https://' in section["text"]: 
            continue
        quality_score = sum(1 for w in financial_signals if w in section["text"].lower()) / len(financial_signals)
        if quality_score < 0.2: 
            continue
        chunks.append({
            "chunk_id": f"{company.lower().replace(' ', '_')}_{fiscal_year}_s{section['section_num']:03d}",
            "text": section["text"], 
            "company": company, 
            "doc_type": doc_type, 
            "fiscal_year": fiscal_year,
            "section_title": section["section_title"], 
            "source_file": source_file, 
            "section_num": section['section_num'],
            "quality_score": round(quality_score, 3)
        })
    return chunks


# # enrich chunk text

# In[38]:


def enrich_chunk_text(chunk):
    return f"{chunk['section_title']}:\n{chunk['text']}"


# In[39]:


converter = TableToNaturalLanguage(company="Apple Inc.")
sections = extract_sections_from_xbrl_html(soup, converter)
chunks = build_chunks_with_citations(sections, "Apple Inc.", "10-K", "FY2023", "aapl-20230930.htm")
enriched_texts = [enrich_chunk_text(c) for c in chunks]


# In[40]:


print(f"Sections extracted: {len(sections)}")
print(f"Chunks after quality filter: {len(chunks)}")


# # Storing in chromadb

# In[42]:


client = chromadb.PersistentClient(path="./chroma_db/")
try: 
    client.delete_collection("financial_docs_v2")
except: 
    pass
collection = client.create_collection(name="financial_docs_v2", metadata={"hnsw:space": "cosine"})


# In[43]:


# Embedding
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(enriched_texts, batch_size=32, show_progress_bar=True)


# In[44]:


collection.add(
    ids=[c["chunk_id"] for c in chunks],
    embeddings=embeddings.tolist(),
    documents=enriched_texts,
    metadatas=[{k: v for k, v in c.items() if k != 'text'} for c in chunks]
)


# In[45]:


tokenized_enriched = [text.lower().split() for text in enriched_texts]
bm25 = BM25Okapi(tokenized_enriched)


# # retrieve with citation 

# In[48]:


def retreive_with_citations(query: str, 
                             bm25_index, 
                             documents: list,
                             chunks: list,
                             collection,
                             n_results: int = 3,
                             alpha: float = 0.7) -> list[dict]:
    """
    We need citation because it will be easy for the use to verify whether the answer is genuine or not 
    """

    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
    if bm25_max > bm25_min:
        bm25_normalized = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
    else:
        bm25_normalized = bm25_scores

    # Dense scores
    dense_results = collection.query(
        query_texts=[query],
        n_results=len(documents)
    )
    dense_distances = np.array(dense_results["distances"][0])
    dense_similarity = 1 - dense_distances
    dense_ids = dense_results["ids"][0]
    all_ids = [chunk["chunk_id"] for chunk in chunks]

    dense_aligned = np.zeros(len(documents))
    for i, doc_id in enumerate(all_ids):
        if doc_id in dense_ids:
            dense_idx = dense_ids.index(doc_id)
            dense_aligned[i] = dense_similarity[dense_idx]

    # Hybrid scores
    hybrid_scores = (alpha * dense_aligned) + ((1 - alpha) * bm25_normalized)
    top_indices = np.argsort(-hybrid_scores)[:n_results]

    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        
        # Build citation string
        # WHY formatted citation: LLM pastes this directly into answer
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


# In[49]:


def expand_financial_query(query: str) -> str:
    """
    query expansion: bridges vocabulary gap between
    natural language questions and numeric financial tables
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


# In[50]:


def citation_formatted_answer(query: str,
                             retrieved_results: list[dict]) -> str:
    context_parts = []
    
    for i, result in enumerate(retrieved_results):
        context_parts.append(
            f"[Source {i+1}: {result['citation']}]\n{result['text'][:500]}"
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

