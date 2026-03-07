from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import redis
import json
import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.db.models import DomainScan, DATABASE_URL

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Phishing Detection API", version="1.0.0")

# Database configuration
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
    redis_client.ping()
except redis.ConnectionError:
    logging.warning("Could not connect to Redis from API layer.")
    redis_client = None

API_TOKEN = os.getenv("API_TOKEN", "default_secret")
api_key_header = APIKeyHeader(name="X-API-Token", auto_error=True)

def verify_token(api_key_header: str = Security(api_key_header)):
    if api_key_header != API_TOKEN:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key_header

class ScanRequest(BaseModel):
    domain: str
    target_cse: str = "Unknown"

@app.post("/api/scan")
def trigger_scan(request: ScanRequest, token: str = Depends(verify_token)):
    """
    Push a domain directly into the processing queue.
    """
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis queue is offline")
        
    job_data = json.dumps({
        "domain": request.domain,
        "source": "api_manual",
        "target_cse": request.target_cse
    })
    redis_client.rpush("domain_analysis_queue", job_data)
    
    return {"status": "ok", "message": f"Domain {request.domain} queued for analysis"}

@app.get("/api/suspected")
def get_suspected_domains(token: str = Depends(verify_token)):
    """
    Retrieve currently suspected domains from the database.
    """
    session = SessionLocal()
    try:
        results = session.query(DomainScan).filter(DomainScan.label == "Suspected").all()
        return [
            {
                "domain": r.domain,
                "target_cse": r.target_cse,
                "scan_date": r.scan_date,
                "confidence": r.confidence,
                "is_live": r.is_live
            } for r in results
        ]
    finally:
        session.close()

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "redis_connected": redis_client is not None,
        "db_url": DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
