from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Security, WebSocket, WebSocketDisconnect, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse, Response
import asyncio
import redis.asyncio as aioredis
import redis
import json
import logging
import time
import uuid
from datetime import datetime
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine, text
import sys
import os
from jose import JWTError, jwt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from sqlalchemy import inspect
from herald.db.models import DomainScan, User, DATABASE_URL, SessionLocal, init_db, Whitelist, engine

# Initialize database tables
from herald.core.auth import verify_password, get_password_hash, create_access_token, SECRET_KEY, ALGORITHM
from herald.utils.logging_config import setup_logging
from herald.utils.export import generate_pdf_report
from herald.monitoring.metrics import metrics
from herald.monitoring.resilience import CircuitBreakerConfig, RedisCircuitBreaker
from herald.monitoring.redis_queue import DOMAIN_ANALYSIS_QUEUE, VISUAL_ANALYSIS_QUEUE, RedisReliableQueue
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("database_connected", database="sqlite")
    inspector = inspect(engine)
    
    if not inspector.has_table("domain_scans"):
        logger.warning("database_initialized")
        return
        
    columns = [col['name'] for col in inspector.get_columns("domain_scans")]
    required_columns = ["lifecycle_state", "screenshot_path", "ocr_text", "dns_records"]
    
    missing_columns = [col for col in required_columns if col not in columns]
    if missing_columns:
        logger.error("schema_mismatch_detected", missing=missing_columns)
        logger.error("Schema mismatch detected! Please run `python setup_db.py` to migrate.")
        sys.exit(1)
    else:
        logger.info("schema_validated")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
logger = structlog.get_logger()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

_redis_client = None
_domain_queue = None
_visual_queue = None
_visual_circuit = None

def get_redis():
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
            _redis_client.ping()
        except redis.ConnectionError:
            logging.warning("Could not connect to Redis from API layer.")
            _redis_client = None
    return _redis_client

def get_domain_queue():
    global _domain_queue
    if _domain_queue is None:
        client = get_redis()
        _domain_queue = RedisReliableQueue(client, DOMAIN_ANALYSIS_QUEUE) if client else None
    return _domain_queue

def get_visual_queue():
    global _visual_queue
    if _visual_queue is None:
        client = get_redis()
        _visual_queue = RedisReliableQueue(client, VISUAL_ANALYSIS_QUEUE) if client else None
    return _visual_queue

def get_visual_circuit():
    global _visual_circuit
    if _visual_circuit is None:
        client = get_redis()
        _visual_circuit = (
            RedisCircuitBreaker(client, CircuitBreakerConfig(name="visual_analysis"))
            if client
            else None
        )
    return _visual_circuit

def __getattr__(name):
    if name == 'redis_client':
        return get_redis()
    elif name == 'domain_queue':
        return get_domain_queue()
    elif name == 'visual_queue':
        return get_visual_queue()
    elif name == 'visual_circuit':
        return get_visual_circuit()
    raise AttributeError(f"module {__name__} has no attribute {name}")

DOMAIN_QUEUE_MAX_READY = int(os.getenv("DOMAIN_QUEUE_MAX_READY", "5000"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.pubsub_task = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Start the pubsub listener only when the first client connects
        if not self.pubsub_task or self.pubsub_task.done():
            self.pubsub_task = asyncio.create_task(self.listen_to_redis())

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                # If sending fails, we just drop the connection later via disconnect
                pass

    async def listen_to_redis(self):
        try:
            # We use async redis for the websocket pub/sub listener
            async_redis = await aioredis.from_url(f"redis://{REDIS_HOST}:6379/0", decode_responses=True)
            pubsub = async_redis.pubsub()
            await pubsub.subscribe("herald.telemetry")
            
            logger.info("telemetry_pubsub_listener_started")
            async for message in pubsub.listen():
                if message["type"] == "message":
                    # Broadcast the JSON string directly to all clients
                    await self.broadcast(message["data"])
                    
                # Exit loop if no clients left
                if not self.active_connections:
                    break
        except Exception as e:
            logger.error("telemetry_pubsub_listener_failed", error=str(e))
        finally:
            if 'async_redis' in locals():
                await async_redis.aclose()

manager = ConnectionManager()

@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection open, handle client heartbeats if needed
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)



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

class InvestigateRequest(BaseModel):
    url: str

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
    domain_queue = get_domain_queue()
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

@app.post("/api/investigate")
@limiter.limit("60/minute")
def investigate_url(request: Request, inv_req: InvestigateRequest, current_user: User = Depends(get_current_user)):
    """
    Push a URL into the real investigation pipeline.
    """
    domain_queue = get_domain_queue()
    if not domain_queue:
        logger.error("redis_offline", action="investigate_url")
        raise HTTPException(status_code=500, detail="Redis queue is offline")

    domain_depth = domain_queue.depth()
    if domain_depth["ready"] >= DOMAIN_QUEUE_MAX_READY:
        raise HTTPException(status_code=429, detail="Investigation queue is under pressure; retry later")
        
    trace_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    
    # Normalize URL to domain
    domain = inv_req.url
    if "://" in domain:
        domain = domain.split("://")[1].split("/")[0]
        
    job_id = domain_queue.enqueue({
        "domain": domain,
        "original_url": inv_req.url,
        "source": "api_investigate",
        "target_cse": "Unknown",
        "trace_id": trace_id,
        "lifecycle_state": "QUEUED"
    })
    logger.info("investigation_queued", url=inv_req.url, domain=domain, user=current_user.username, job_id=job_id)
    
    return {
        "status": "ok",
        "job_id": job_id,
        "trace_id": trace_id,
        "domain": domain,
        "lifecycle_state": "QUEUED"
    }


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    domain_queue = get_domain_queue()
    visual_queue = get_visual_queue()
    visual_circuit = get_visual_circuit()
    
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
    


@app.get("/api/admin/failed-jobs")
def get_failed_jobs(current_user: User = Depends(get_current_user)):
    redis_client = get_redis()
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis queue is offline")
    
    failed_jobs = redis_client.lrange(DOMAIN_ANALYSIS_QUEUE.dlq, 0, -1)
    return {"count": len(failed_jobs), "jobs": [json.loads(job) for job in failed_jobs]}

@app.post("/api/admin/failed-jobs/retry")
def retry_failed_jobs(current_user: User = Depends(get_current_user)):
    domain_queue = get_domain_queue()
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

        "analyst_verdict": scan.analyst_verdict,
        "lifecycle_state": scan.lifecycle_state,
        "screenshot_path": scan.screenshot_path,
        "ocr_text": scan.ocr_text,
        "dns_records": scan.dns_records
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
def health_check():
    """Lightweight liveness check."""
    return {"status": "ok"}

@app.get("/api/ready")
def readiness_check(db: Session = Depends(get_db)):
    """Readiness check: Redis, DB, telemetry, queues, websockets."""
    status = "ok"
    reasons = []

    # Database
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        db_ok = False
        status = "degraded"
        reasons.append(f"db_error: {str(e)}")

    # Redis
    redis_client = get_redis()
    redis_ok = redis_client is not None
    if not redis_ok:
        status = "degraded"
        reasons.append("redis_disconnected")
        
    return {
        "status": status,
        "database": "connected" if db_ok else "disconnected",
        "redis": "connected" if redis_ok else "disconnected",
        "telemetry_subsystem": "ready" if manager.pubsub_task else "inactive",
        "websocket_clients": len(manager.active_connections),
        "reasons": reasons
    }

@app.get("/api/metrics-summary")
def metrics_summary():
    """Operational metrics summaries for dashboard consumption."""
    summary = {
        "queues": {},
        "workers": {},
        "browser_pressure": {},
        "circuit_breakers": {}
    }

    redis_client = get_redis()
    if not redis_client:
        return {"status": "error", "message": "Redis unavailable"}

    # Queue Metrics
    domain_queue = get_domain_queue()
    visual_queue = get_visual_queue()
    if domain_queue:
        summary["queues"]["lexical"] = domain_queue.depth()
    if visual_queue:
        summary["queues"]["visual"] = visual_queue.depth()
        
    # Active Worker Counts (Lightweight via Redis last_seen)
    domain_last_seen = redis_client.get("worker:domain:last_seen")
    visual_last_seen = redis_client.get("worker:visual:last_seen")
    
    def is_active(ts_str):
        if not ts_str: return False
        try:
            return (datetime.utcnow() - datetime.fromisoformat(ts_str)).total_seconds() < 120
        except: return False

    summary["workers"]["lexical_active"] = 1 if is_active(domain_last_seen) else 0
    summary["workers"]["visual_active"] = 1 if is_active(visual_last_seen) else 0

    # Browser Pressure Metrics (Aggregated from Visual Worker Telemetry)
    # Since we are using lightweight Redis counters:
    summary["browser_pressure"] = {
        "concurrent_sessions": int(redis_client.get("browser:active_sessions") or 0),
        "saturation_pct": int(redis_client.get("browser:saturation_pct") or 0),
        "crashes_1m": int(redis_client.get("browser:crashes_1m") or 0),
        "timeouts_1m": int(redis_client.get("browser:timeouts_1m") or 0)
    }

    visual_circuit = get_visual_circuit()
    if visual_circuit:
        summary["circuit_breakers"]["visual_analysis"] = visual_circuit.state()

    return summary

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
