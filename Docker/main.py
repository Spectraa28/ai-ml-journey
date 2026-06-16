from fastapi import FastAPI
import psycopg2
import os

app = FastAPI()

@app.get("/health")
def health():
    try:
        conn =  psycopg2.connect(os.getenv("DATABASE_URL"))
        conn.close()
        return {"status": "ok" , "db":"connected"}
    except Exception as e:
        return {"status": "ok", "db": str(e)}
    
    
@app.get("/ping")
def ping():
    return {"message": "pong"}