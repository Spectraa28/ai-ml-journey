# IMPOrts 
import os
import re
import bs4
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from bs4 import BeautifulSoup

# Better parsing tools 
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

def enrich_chunk_text(chunk):
    return f"{chunk['section_title']}:\n{chunk['text']}"

# INgestion with Docling 
def ingest_with_docling(file_path: str,company_name: str, doc_type: str, fiscal_year: str) -> list[dict]:
    """
    Parses a document using IBM Docling and constructs a list of structured chunks
    mapped to the downstream vector database schema.
    """
    
    print(f"Initializing Docling document converter for {file_path}")
    
    # Extract the layout aware doc conversion 
    converter = DocumentConverter()
    result = converter.convert(file_path)
    
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    # INitialize structural chunker 
    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=256)
    
    chunks = []
    
    for i, docling_chunk in enumerate(chunker.chunk(result.document)):
        text_content = docling_chunk.text.strip()
        
        if not text_content:
            continue
        
        heading_list = docling_chunk.meta.headings if docling_chunk.meta.headings else ["General Document"]
        
        section_title = heading_list[-1]
        
        citation_path = ' > '.join(heading_list)
        citation = f"{company_name} ({fiscal_year}) {doc_type} |  {citation_path}"
        
        chunk_dict = {
            "chunk_id": f"{company_name}_{fiscal_year}_chunk_{i}",
            "company": company_name,
            "doc_type": doc_type,
            "fiscal_year": fiscal_year,
            "section_title": section_title,
            "citation": citation,
            "text": text_content
        }
        
        chunks.append(chunk_dict)
        
    print(f" Parsing completed . Generated {len(chunks)} layout aware chunks")

    return  chunks



# class TableToNaturalLanguage:
#     def __init__(self, company):
#         self.company = company
        
#     def is_financial_table(self, table_soup):
#         financial_keywords = ["net", "revenue", "income", "sales", "earnings", "profit", "loss", "assets", "cash", "cost"]
#         numeric_pattern = re.compile(r'\$?[\d,]+\.?\d*')
#         rows = table_soup.find_all("tr")

#         for row in rows:
#             cells = row.find_all("td")
#             if len(cells) < 2: continue
#             first_cell_text = cells[0].get_text(strip=True).lower()
#             has_keyword = any(kw in first_cell_text for kw in financial_keywords)
#             has_numeric_value = any(re.search(numeric_pattern, c.get_text(strip=True)) for c in cells[1:])
            
#             if has_keyword and has_numeric_value:
#                 return True
#         return False

#     def convert(self, raw_html):
#         soup = BeautifulSoup(raw_html, "html.parser")
#         if not self.is_financial_table(soup): return []

#         rows = soup.find_all("tr")
#         col_map = {}
#         year_pattern = re.compile(r'20\d{2}')
#         results = []

#         # Find years in header
#         for row in rows:
#             cells = row.find_all(["td", "th"])
#             for i, cell in enumerate(cells):
#                 text = cell.get_text(strip=True)
#                 if year_pattern.fullmatch(text): col_map[i] = text
#             if col_map: break

#         if not col_map: return []
#         years_in_order = [year for idx, year in sorted(col_map.items())]

#         for row in rows:
#             cells = row.find_all("td")
#             if len(cells) < 2: continue
#             metric = cells[0].get_text(strip=True)
#             if not metric: continue
            
#             numeric_values = [c.get_text(strip=True) for c in cells[1:] if re.search(r'\d{3,}', c.get_text(strip=True))]
            
#             for value, year in zip(numeric_values, years_in_order):
#                 sentence = f"The {metric} of {self.company} in the fiscal year {year} was {value}."
#                 results.append(sentence)
#         return results

# def extract_sections_from_xbrl_html(soup, table_converter):
#     sections, section_num = [], 0
#     current_title, current_text, current_length = "Preamble", [], 0
#     all_elements = soup.find_all(['span', 'div', 'p', 'ix:nonnumeric', 'ix:nonfraction', 'table'])
    
#     for elem in all_elements:
#         if elem.name == 'table':
#             table_prose = table_converter.convert(str(elem))
#             if not table_prose: continue
#             text, is_header = " ".join(table_prose), False
#         else:
#             if elem.find(['span', 'div']): continue
#             text = elem.get_text(strip=True)
#             if len(text) < 5: continue
#             style = elem.get('style', '')
#             is_header = ('font-weight:700' in style and len(text) < 150 and 
#                          any(text.upper().startswith(k) for k in ['ITEM', 'PART', 'CONSOLIDATED', 'NOTE']))
        
#         if is_header and current_text and current_length > 200:
#             section_num += 1
#             sections.append({"section_num": section_num, "section_title": current_title, "text": " ".join(current_text), "char_count": current_length})
#             current_title, current_text, current_length = text[:100], [], 0
#         else:
#             current_text.append(text)
#             current_length += len(text)
#             if current_length > 1500:
#                 section_num += 1
#                 sections.append({"section_num": section_num, "section_title": current_title, "text": " ".join(current_text), "char_count": current_length})
#                 current_text, current_length = [], 0
    
#     if current_text:
#         sections.append({"section_num": section_num + 1, "section_title": current_title, "text": " ".join(current_text), "char_count": current_length})
#     return sections

# def build_chunks_with_citations(sections, company, doc_type, fiscal_year, source_file):
#     financial_signals = ["revenue", "income", "profit", "sales", "earnings", "cash", "assets", "net", "total"]
#     chunks = []
#     for section in sections:
#         if 'http' in section["text"]: continue
#         quality_score = sum(1 for w in financial_signals if w in section["text"].lower()) / len(financial_signals)
#         if quality_score < 0.2: continue
#         chunks.append({
#             "chunk_id": f"{company.lower().replace(' ', '_')}_{fiscal_year}_s{section['section_num']:03d}",
#             "text": section["text"], 
#             "company": company, "doc_type": doc_type, "fiscal_year": fiscal_year,
#             "section_title": section["section_title"], "source_file": source_file, 
#             "section_num": section['section_num'], "quality_score": round(quality_score, 3)
#         })
#     return chunks



# def initialize_ingestion(file_path: str, company_name: str):
#     """
#     Orchestrates the ingestion, embedding, and storage.
#     Ensures work is only performed if the database is empty.
#     """
#     chroma_client = chromadb.PersistentClient(path="./chroma_db/")
#     collection = chroma_client.get_or_create_collection(name="financial_docs_v3", metadata={"hnsw:space": "cosine"})
#     model = SentenceTransformer('all-MiniLM-L6-v2')
    
#     if collection.count() > 0:
#         print(f"Cache Hit: Found existing collection with {collection.count()} chunks. Skipping HTML parsing and embedding pipeline.")
        
#         # Pull raw texts and metadata objects straight from the vector database
#         stored_data = collection.get(include=["documents", "metadatas"])
#         enriched_texts = stored_data["documents"]
        
#         chunks = []
#         for meta, doc in zip(stored_data["metadatas"], stored_data["documents"]):
#             chunk_dict = meta.copy()
#             chunk_dict["text"] = doc  # Inject document string back into the pipeline payload
#             chunks.append(chunk_dict)
    
#     else:
#         print("Cache Miss / First-time run: Initializing full document processing pipeline...")
        
#         # 2. Heavy HTML Extraction & Transformation Pipeline
#         with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#             html = f.read()

#         soup = BeautifulSoup(html, 'html.parser')
#         for tag in soup(['script', 'style', 'meta', 'link']):
#             tag.decompose()

#         converter = TableToNaturalLanguage(company=company_name)
#         sections = extract_sections_from_xbrl_html(soup, converter)
#         chunks = build_chunks_with_citations(sections, company_name, "10-K", "FY2023", file_path)
#         enriched_texts = [enrich_chunk_text(c) for c in chunks]

#         # 3. Vector Storage Allocation
#         print("Generating heavy semantic embedding weights...")
#         embeddings = model.encode(enriched_texts, batch_size=32, show_progress_bar=True)
#         collection.add(
#             ids=[c["chunk_id"] for c in chunks],
#             embeddings=embeddings.tolist(),
#             documents=enriched_texts,
#             metadatas=[{k: v for k, v in c.items() if k != 'text'} for c in chunks]
#         )
        
        
        
def intialize_ingestion(file_path: str, company_name: str, doc_type:str = "10-k",fiscal_year: str = "FY2023"):
    """
        Orchestrates the ingestion , embedding , and storage.
        Legacy BeautifulSoup  parsing replaced with layout-aware IBM Docling 
    """
        
    chroma_client = chromadb.PersistentClient(path="./chroma_db/")
        
    collection = chroma_client.get_or_create_collection(name="financial_docs_v3",metadata={"hnsw:space":"cosine"})
        
    model = SentenceTransformer('all-MiniLM-L6-v2')
        
    if collection.count() > 0:
        print(f"Cache hit:  found collection with {collection.count()} chunks. Skipping Docling parsing.")
            
        stored_data = collection.get(include=["documents","metadatas"])
        enriched_texts = stored_data["documents"]
        
        chunks =[]
        for meta, doc in zip(stored_data["metadatas"], stored_data["documents"]):
            chunk_dict = meta.copy()
            chunk_dict["text"] = doc
            chunks.append(chunk_dict)
    
    else:
        print("Cache miss / first time run")
        
        converter = DocumentConverter()
        result = converter.convert(file_path)
        
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        chunker = HybridChunker(tokenizer=tokenizer, max_tokens=256)
        
        chunks = []
        for i , docling_chunk in enumerate(chunker.chunk(result.document)):
            text_content = docling_chunk.text.strip()
            if not text_content or len(text_content) < 50:
                continue
            
            heading_list = docling_chunk.meta.headings if docling_chunk.meta.headings else ["General  Document"]
            section_title = heading_list[-1]
            citation_path = ' > '.join(heading_list)
            citation = f"{company_name} ({fiscal_year}) {doc_type} | {citation_path}"
            
            chunks.append({
                "chunk_id":f"{company_name}_{fiscal_year}_chunk_{i}",
                "company": company_name,
                "doc_type":doc_type,
                "fiscal_year": fiscal_year,
                "section_title":section_title,
                "citation": citation,
                "text": text_content
            })
        
        print(f"✅ Docling parsing complete. Generated {len(chunks)} chunks.")
        
        enriched_texts = [enrich_chunk_text(c) for c in chunks]
        
        print("Generating heavy semantic embedding weights.... ")
        embeddings = model.encode(enriched_texts, batch_size=32, show_progress_bar=True)
        collection.add(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=embeddings.tolist(),
            documents=enriched_texts,
            metadatas=[{k:  v for k , v in c.items() if k != 'text'} for c in chunks]
        )
        
    # 4. Shared High-Speed In-Memory BM25 Index Regeneration
    tokenized_enriched = [text.lower().split() for text in enriched_texts]
    bm25 = BM25Okapi(tokenized_enriched)

    return {
        "chunks": chunks,
        "enriched_texts": enriched_texts,
        "collection": collection,
        "model": model,
        "bm25": bm25
    }