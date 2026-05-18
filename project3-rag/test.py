from ingestion import ingest_with_docling
import json 

path = "10-K 2023.pdf"

result = ingest_with_docling("10-K 2023.pdf","apple","pdf","2023")

print("💾 Saving preview checkpoint to disk...")
with open("docling_chunks_preview.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4)
        
print("✅ Saved first 5 and last 5 chunks to docling_chunks_preview.json. Safe to inspect.")


import json
with open("docling_chunks_preview.json") as f:
    # Load full chunks not just preview
    pass

short_chunks = [c for c in result if len(c["text"]) < 50]
print(f"Short chunks under 50 chars: {len(short_chunks)}")
print(f"Percentage: {len(short_chunks)/len(result)*100:.1f}%")