from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse, Response
import redis
import json
import logging
import time
import uuid
from datetime import datetime
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
from herald.monitoring.metrics import metrics
from herald.monitoring.resilience import CircuitBreakerConfig, RedisCircuitBreaker
from herald.monitoring.redis_queue import DOMAIN_ANALYSIS_QUEUE, VISUAL_ANALYSIS_QUEUE, RedisReliableQueue
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

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

domain_queue = RedisReliableQueue(redis_client, DOMAIN_ANALYSIS_QUEUE) if redis_client else None
visual_queue = RedisReliableQueue(redis_client, VISUAL_ANALYSIS_QUEUE) if redis_client else None
visual_circuit = (
    RedisCircuitBreaker(redis_client, CircuitBreakerConfig(name="visual_analysis"))
    if redis_client
    else None
)
DOMAIN_QUEUE_MAX_READY = int(os.getenv("DOMAIN_QUEUE_MAX_READY", "5000"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    trace_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    bind_contextvars(trace_id=trace_id)
    started_at = time.monotonic()
    try:
        response = await call_next(request)
        metrics.increment("herald_http_requests_total", method=request.method, path=request.url.path, status=str(response.status_code))
        response.headers["x-request-id"] = trace_id
        return response
    finally:
        metrics.observe("herald_http_request_seconds", time.monotonic() - started_at, method=request.method, path=request.url.path)
        clear_contextvars()

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
    if not domain_queue:
        logger.error("redis_offline", action="trigger_scan")
        raise HTTPException(status_code=500, detail="Redis queue is offline")

    domain_depth = domain_queue.depth()
    metrics.gauge("herald_queue_depth", domain_depth["ready"], queue=DOMAIN_ANALYSIS_QUEUE.ready, state="ready")
    if domain_depth["ready"] >= DOMAIN_QUEUE_MAX_READY:
        logger.warning("domain_queue_pressure_rejected", ready_depth=domain_depth["ready"], max_ready=DOMAIN_QUEUE_MAX_READY)
        metrics.increment("herald_scan_rejected_total", reason="queue_pressure")
        raise HTTPException(status_code=429, detail="Scan queue is under pressure; retry later")
        
    trace_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    job_id = domain_queue.enqueue({
        "domain": scan_req.domain,
        "source": "api_manual",
        "target_cse": scan_req.target_cse,
        "trace_id": trace_id,
    })
    logger.info("domain_queued", domain=scan_req.domain, user=current_user.username, job_id=job_id)
    
    return {"status": "ok", "job_id": job_id, "message": f"Domain {scan_req.domain} queued for analysis"}


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    if domain_queue:
        for state, value in domain_queue.depth().items():
            metrics.gauge("herald_queue_depth", value, queue=DOMAIN_ANALYSIS_QUEUE.ready, state=state)
    if visual_queue:
        for state, value in visual_queue.depth().items():
            metrics.gauge("herald_queue_depth", value, queue=VISUAL_ANALYSIS_QUEUE.ready, state=state)
    if visual_circuit:
        circuit_state = visual_circuit.state()
        metrics.gauge("herald_circuit_failures", circuit_state["failures"], circuit="visual_analysis")
        metrics.gauge("herald_circuit_open", 1 if circuit_state["state"] == "open" else 0, circuit="visual_analysis")

    return PlainTextResponse(metrics.render_prometheus(), media_type="text/plain; version=0.0.4")

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
    
    failed_jobs = redis_client.lrange(DOMAIN_ANALYSIS_QUEUE.dlq, 0, -1)
    return {"count": len(failed_jobs), "jobs": [json.loads(job) for job in failed_jobs]}

@app.post("/api/admin/failed-jobs/retry")
def retry_failed_jobs(current_user: User = Depends(get_current_user)):
    if not domain_queue:
        raise HTTPException(status_code=500, detail="Redis queue is offline")
        
    requeued = domain_queue.drain_dlq_to_ready()
    logger.info("failed_jobs_requeued", count=requeued)
    return {"status": "ok", "requeued": requeued}

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
            if domain_queue:
                health["services"]["redis"]["domain_queue"] = domain_queue.depth()
            if visual_queue:
                health["services"]["redis"]["visual_queue"] = visual_queue.depth()
            if visual_circuit:
                health["services"]["visual_circuit"] = visual_circuit.state()
            
            last_seen = redis_client.get("worker:domain:last_seen")
            if last_seen:
                health["services"]["worker"]["last_seen"] = last_seen
                # If last seen > 5 minutes ago, mark as unhealthy
                last_seen_dt = datetime.fromisoformat(last_seen)
                if (datetime.utcnow() - last_seen_dt).total_seconds() > 300:
                    health["services"]["worker"]["status"] = "stale"
                    health["status"] = "degraded"
                else:
                    health["services"]["worker"]["status"] = "active"
            else:
                health["services"]["worker"]["status"] = "not_seen"

            visual_last_seen = redis_client.get("worker:visual:last_seen")
            health["services"]["visual_worker"] = {"status": "not_seen"}
            if visual_last_seen:
                health["services"]["visual_worker"]["last_seen"] = visual_last_seen
                visual_last_seen_dt = datetime.fromisoformat(visual_last_seen)
                if (datetime.utcnow() - visual_last_seen_dt).total_seconds() > 300:
                    health["services"]["visual_worker"]["status"] = "stale"
                    health["status"] = "degraded"
                else:
                    health["services"]["visual_worker"]["status"] = "active"
        except Exception as e:
            health["services"]["redis"]["error"] = str(e)
            
    return health

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
