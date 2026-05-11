from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from fastapi.responses import Response
import redis
import json
import logging
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine, text
import sys
import os
from jose import JWTError, jwt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from herald.db.models import DomainScan, User, DATABASE_URL, SessionLocal, init_db, Whitelist

# Initialize database tables
init_db()
from herald.core.auth import verify_password, get_password_hash, create_access_token, SECRET_KEY, ALGORITHM
from herald.utils.logging_config import setup_logging
from herald.utils.export import generate_pdf_report
import structlog

setup_logging()
logger = structlog.get_logger(__name__)

from fastapi.middleware.cors import CORSMiddleware

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

app = FastAPI(title="Phishing Detection API", version="1.0.0")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
    redis_client.ping()
except redis.ConnectionError:
    logging.warning("Could not connect to Redis from API layer.")
    redis_client = None

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")

@app.get("/")
def read_root():
    return {
        "name": "HERALD Phishing Detection API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs"
    }

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

class ScanRequest(BaseModel):
    domain: str
    target_cse: str = "Unknown"

# Auth Schemas
class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "analyst"

class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    is_active: bool

    class Config:
        from_attributes = True

@app.post("/api/auth/register", response_model=UserResponse)
def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    # Simple registration for P0/MVP. In production, this should be admin-only or restricted.
    existing_user = db.query(User).filter(User.username == user_in.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_pw = get_password_hash(user_in.password)
    new_user = User(
        username=user_in.username, 
        hashed_password=hashed_pw,
        role=user_in.role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/api/auth/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/scan")
@limiter.limit("60/minute")
def trigger_scan(request: Request, scan_req: ScanRequest, current_user: User = Depends(get_current_user)):
    """
    Push a domain directly into the processing queue.
    """
    if not redis_client:
        logger.error("redis_offline", action="trigger_scan")
        raise HTTPException(status_code=500, detail="Redis queue is offline")
        
    job_data = json.dumps({
        "domain": scan_req.domain,
        "source": "api_manual",
        "target_cse": scan_req.target_cse
    })
    redis_client.rpush("domain_analysis_queue", job_data)
    logger.info("domain_queued", domain=scan_req.domain, user=current_user.username)
    
    return {"status": "ok", "message": f"Domain {scan_req.domain} queued for analysis"}

@app.get("/api/suspected")
def get_suspected_domains(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Retrieve currently suspected domains from the database.
    """
    results = db.query(DomainScan).filter(DomainScan.label == "Suspected").all()
    return [
        {
            "domain": r.domain,
            "target_cse": r.target_cse,
            "scan_date": r.scan_date,
            "confidence": r.confidence,
            "is_live": r.is_live,
            "analyst_verdict": r.analyst_verdict
        } for r in results
    ]

@app.get("/api/detections")
def get_recent_detections(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Retrieve the 50 most recent detections from the database.
    """
    results = db.query(DomainScan).order_by(DomainScan.scan_date.desc()).limit(50).all()
    return [
        {
            "domain": r.domain,
            "label": r.label,
            "confidence": r.confidence,
            "target_cse": r.target_cse,
            "source": r.source,
            "scan_date": r.scan_date,
            "analyst_verdict": r.analyst_verdict
        } for r in results
    ]

class FeedbackRequest(BaseModel):
    domain: str
    verdict: str  # 'TP', 'FP', 'Escalated'

@app.post("/api/feedback")
def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    scan = db.query(DomainScan).filter(DomainScan.domain == request.domain).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Domain scan not found")
    
    scan.analyst_verdict = request.verdict
    db.commit()
    logger.info("feedback_submitted", domain=request.domain, verdict=request.verdict)
    return {"status": "ok", "message": "Feedback recorded"}

@app.get("/api/admin/failed-jobs")
def get_failed_jobs(current_user: User = Depends(get_current_user)):
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis queue is offline")
    
    failed_jobs = redis_client.lrange("failed_jobs", 0, -1)
    return {"count": len(failed_jobs), "jobs": [json.loads(job) for job in failed_jobs]}

@app.post("/api/admin/failed-jobs/retry")
def retry_failed_jobs(current_user: User = Depends(get_current_user)):
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis queue is offline")
        
    failed_jobs = redis_client.lrange("failed_jobs", 0, -1)
    if not failed_jobs:
        return {"status": "ok", "requeued": 0}
        
    redis_client.delete("failed_jobs")
    
    for job_str in failed_jobs:
        job = json.loads(job_str)
        job["retries"] = 0 # reset retries
        redis_client.rpush("domain_analysis_queue", json.dumps(job))
        
    logger.info("failed_jobs_requeued", count=len(failed_jobs))
    return {"status": "ok", "requeued": len(failed_jobs)}

class WhitelistCreate(BaseModel):
    domain: str
    reason: str = None

@app.get("/api/whitelist")
def get_whitelist(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    results = db.query(Whitelist).all()
    return [{"domain": r.domain, "added_by": r.added_by, "added_on": r.added_on, "reason": r.reason} for r in results]

@app.post("/api/whitelist")
def add_to_whitelist(item: WhitelistCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    existing = db.query(Whitelist).filter(Whitelist.domain == item.domain).first()
    if existing:
        raise HTTPException(status_code=400, detail="Domain already in whitelist")
    
    new_entry = Whitelist(domain=item.domain, reason=item.reason)
    db.add(new_entry)
    db.commit()
    logger.info("whitelist_added", domain=item.domain, reason=item.reason)
    return {"status": "ok", "message": f"{item.domain} added to whitelist"}

@app.delete("/api/whitelist/{domain}")
def remove_from_whitelist(domain: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    entry = db.query(Whitelist).filter(Whitelist.domain == domain).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Domain not found in whitelist")
    
    db.delete(entry)
    db.commit()
    logger.info("whitelist_removed", domain=domain)
    return {"status": "ok", "message": f"{domain} removed from whitelist"}

@app.get("/api/export/{domain}/json")
def export_json(domain: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    scan = db.query(DomainScan).filter(DomainScan.domain == domain).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Domain not found")
        
    return {
        "domain": scan.domain,
        "target_cse": scan.target_cse,
        "source": scan.source,
        "scan_date": scan.scan_date,
        "label": scan.label,
        "confidence": scan.confidence,
        "is_live": scan.is_live,
        "analyst_verdict": scan.analyst_verdict
    }

@app.get("/api/export/{domain}/pdf")
def export_pdf(domain: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    scan = db.query(DomainScan).filter(DomainScan.domain == domain).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Domain not found")
        
    pdf_buffer = generate_pdf_report(scan)
    headers = {
        'Content-Disposition': f'attachment; filename="herald_report_{domain}.pdf"'
    }
    return Response(content=pdf_buffer.getvalue(), media_type="application/pdf", headers=headers)

@app.get("/api/health")
def health_check(db: Session = Depends(get_db)):
    health = {
        "status": "healthy",
        "services": {
            "redis": {"connected": redis_client is not None},
            "database": {"connected": False},
            "worker": {"status": "unknown"}
        }
    }
    
    # Check DB
    try:
        db.execute(text("SELECT 1"))
        health["services"]["database"]["connected"] = True
    except Exception as e:
        health["status"] = "degraded"
        health["services"]["database"]["error"] = str(e)
        
    # Check Redis Queue Depth and Worker
    if redis_client:
        try:
            health["services"]["redis"]["queue_depth"] = redis_client.llen("domain_analysis_queue")
            health["services"]["redis"]["failed_jobs_count"] = redis_client.llen("failed_jobs")
            
            last_seen = redis_client.get("worker:last_seen")
            if last_seen:
                health["services"]["worker"]["last_seen"] = last_seen
                # If last seen > 5 minutes ago, mark as unhealthy
                from datetime import datetime
                last_seen_dt = datetime.fromisoformat(last_seen)
                if (datetime.utcnow() - last_seen_dt).total_seconds() > 300:
                    health["services"]["worker"]["status"] = "stale"
                    health["status"] = "degraded"
                else:
                    health["services"]["worker"]["status"] = "active"
            else:
                health["services"]["worker"]["status"] = "not_seen"
        except Exception as e:
            health["services"]["redis"]["error"] = str(e)
            
    return health

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
