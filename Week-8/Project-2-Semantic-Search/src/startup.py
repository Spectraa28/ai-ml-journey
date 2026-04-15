import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

def initialize_database():
    print("Starting database initialization...")
    
    # Step 1: connect to chroma, get or create collection
    client = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_or_create_collection(name="cfpb_complaints")
    
    # Step 2: check if already indexed
    current_count = collection.count()
    
    # Step 3: if populated → print "already indexed"
    if current_count > 0:
        print(f"✅ Database already indexed with {current_count} records. Skipping ingestion.")
        return collection
        
    # Step 3: if empty → load CSV → encode → store
    print("⚠️ Database empty. Starting ingestion process (this may take a few minutes)...")
    
    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2") # Or whichever model won your MLflow comparison
    
    # Load data
    df = pd.read_csv('data/complaints_clean.csv')
    
    # Define batch logic here...
    batch_size = 1000

    for i in range(0, len(df), batch_size):
        batch_end = min(i + batch_size, len(df))
        batch_texts = df['clean_narrative'].iloc[i:batch_end].tolist()
        
        # Encode only this batch — not all 25k
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=64,
            show_progress_bar=True
        ).tolist()
        
        collection.add(
            documents=batch_texts,
            embeddings=batch_embeddings,
            ids=[str(x) for x in df['complaint_id'].iloc[i:batch_end].tolist()],
            metadatas=[
                {"product": row['product'], "issue": row['issue']}
                for _, row in df.iloc[i:batch_end].iterrows()
            ]
        )
        
        print(f"Stored {batch_end}/{len(df)} complaints")
    
    print(f"✅ Ingestion complete. Database now holds {collection.count()} records.")
    return collection