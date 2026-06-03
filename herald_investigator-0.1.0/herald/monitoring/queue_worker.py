import os
import time
from datetime import datetime

import redis
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars
from sqlalchemy.exc import SQLAlchemyError

from herald.db.models import DomainScan, SessionLocal, Whitelist
from herald.monitoring.metrics import Timer, metrics
from herald.monitoring.redis_queue import DOMAIN_ANALYSIS_QUEUE, VISUAL_ANALYSIS_QUEUE, RedisReliableQueue
from herald.detection.engine import DetectionEngine
from herald.utils.logging_config import setup_logging
from herald.telemetry.stream import TelemetryStream
from herald.telemetry.emitter import TelemetryEmitter
from herald.core.security import validate_url_safe, SSRFProtectionError

setup_logging()
logger = structlog.get_logger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
IDEMPOTENCY_TTL_SECONDS = int(os.getenv("IDEMPOTENCY_TTL_SECONDS", "86400"))
VISUAL_QUEUE_MAX_READY = int(os.getenv("VISUAL_QUEUE_MAX_READY", "1000"))


def build_redis_client():
    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=0,
        decode_responses=True,
        socket_timeout=30,
        socket_connect_timeout=5,
        health_check_interval=30,
    )
    client.ping()
    return client


redis_client = None
domain_queue = None
visual_queue = None
engine = None
emitter = None


def init_worker():
    global redis_client, domain_queue, visual_queue, engine, emitter
    if redis_client is not None:
        return

    try:
        redis_client = build_redis_client()
    except redis.RedisError:
        logger.warning("redis_connection_failed", service="queue_worker")
        redis_client = None

    domain_queue = RedisReliableQueue(redis_client, DOMAIN_ANALYSIS_QUEUE) if redis_client else None
    visual_queue = RedisReliableQueue(redis_client, VISUAL_ANALYSIS_QUEUE) if redis_client else None
    engine = DetectionEngine(scorer_type="ml")

    # Initialize Telemetry
    telemetry_stream = TelemetryStream(redis_client)
    TelemetryEmitter.initialize(telemetry_stream, worker_type="domain_worker")
    emitter = TelemetryEmitter.get()

def normalize_domain(domain: str) -> str:
    return domain.lower().strip().replace("www.", "")


def should_skip_duplicate(domain: str, job: dict) -> bool:
    init_worker()
    if not redis_client:
        return False

    if int(job.get("attempts", 0)) > 0:
        return False

    today = datetime.utcnow().strftime("%Y-%m-%d")
    seen_key = f"domain:seen:{normalize_domain(domain)}:{today}"
    is_new = redis_client.setnx(seen_key, "1")
    if is_new:
        redis_client.expire(seen_key, IDEMPOTENCY_TTL_SECONDS)
        return False

    return True


def upsert_domain_scan(session, *, domain: str, label: str, confidence: float, target_cse: str, source: str, lifecycle_state: str = "PROCESSING") -> None:
    existing = session.query(DomainScan).filter_by(domain=domain).first()
    if existing:
        existing.label = label
        existing.confidence = confidence
        existing.target_cse = target_cse
        existing.scan_date = datetime.utcnow()
        existing.source = source
        existing.lifecycle_state = lifecycle_state
        return

    session.add(
        DomainScan(
            domain=domain,
            label=label,
            confidence=confidence,
            target_cse=target_cse,
            source=source,
            scan_date=datetime.utcnow(),
            lifecycle_state=lifecycle_state
        )
    )


def enqueue_visual_analysis(domain: str, result: dict, source: str) -> None:
    init_worker()
    if not visual_queue:
        logger.warning("visual_queue_unavailable", domain=domain)
        metrics.increment("herald_visual_enqueue_degraded_total", reason="queue_unavailable")
        return

    visual_depth = visual_queue.depth()
    metrics.gauge("herald_queue_depth", visual_depth["ready"], queue=VISUAL_ANALYSIS_QUEUE.ready, state="ready")
    if visual_depth["ready"] >= VISUAL_QUEUE_MAX_READY:
        logger.warning(
            "visual_queue_pressure_degraded",
            domain=domain,
            ready_depth=visual_depth["ready"],
            max_ready=VISUAL_QUEUE_MAX_READY,
        )
        metrics.increment("herald_visual_enqueue_degraded_total", reason="queue_pressure")
        return

    visual_queue.enqueue(
        {
            "domain": domain,
            "target_cse": result.get("target_cse", "Unknown"),
            "initial_confidence": result.get("ml_confidence_adjusted", result.get("ml_confidence", 0.0)),
            "source": source,
            "parent_analysis_type": result.get("analysis_type"),
        }
    )
    logger.info("visual_analysis_queued", domain=domain, target_cse=result.get("target_cse", "Unknown"))


def process_domain(job_data: dict) -> None:
    init_worker()
    domain = job_data.get("domain")
    if not domain:
        raise ValueError("job is missing required domain")

    target_cse = job_data.get("target_cse", "Unknown")
    source = job_data.get("source", "manual")

    bind_contextvars(trace_id=job_data.get("trace_id"), job_id=job_data.get("job_id"))
    logger.info("processing_domain", domain=domain, target_cse=target_cse, source=source)

    queue_wait_ms = int((time.time() - job_data.get("enqueued_at", time.time())) * 1000) if job_data.get("enqueued_at") else 0
    emitter.emit_event(
        event_type="JOB_ACCEPTED", 
        payload={"domain": domain, "job_id": job_data.get("job_id")}, 
        priority="LOW", 
        trace_id=job_data.get("trace_id")
    )

    session = SessionLocal()
    start_time = time.time()
    
    # SSRF Protection Check
    original_url = job_data.get("original_url", f"http://{domain}")
    try:
        validate_url_safe(original_url)
    except SSRFProtectionError as exc:
        logger.warning("domain_rejected_ssrf", domain=domain, error=str(exc))
        upsert_domain_scan(session, domain=domain, label="Rejected", confidence=1.0, target_cse=target_cse, source=source, lifecycle_state="FAILED")
        session.commit()
        session.close()
        clear_contextvars()
        return

    try:
        with Timer("herald_domain_processing_seconds", worker="domain"):
            clean_domain = normalize_domain(domain)
            is_whitelisted = session.query(Whitelist).filter(Whitelist.domain == clean_domain).first()

            if is_whitelisted:
                label = "Clean"
                confidence = 0.01
                result = {"analysis_type": "Whitelist"}
                logger.info("domain_whitelisted_intercept", domain=domain)
            else:
                from herald.detection.models import display_verdict
                detection_res = engine.score(domain)
                label = display_verdict(detection_res.verdict)
                confidence = detection_res.confidence
                result = {
                    "analysis_type": detection_res.model_version,
                    "ml_confidence": detection_res.features.get("ml_confidence", confidence),
                    "ml_confidence_adjusted": detection_res.features.get("ml_confidence_adjusted", confidence),
                    "visual_analysis_required": detection_res.features.get("visual_analysis_required", False),
                    "target_cse": detection_res.features.get("target_cse") or target_cse,
                }

            upsert_domain_scan(
                session,
                domain=domain,
                label=label,
                confidence=confidence,
                target_cse=target_cse,
                source=source,
            )
            session.commit()

            if result.get("visual_analysis_required"):
                enqueue_visual_analysis(domain, result, source)

        duration_ms = int((time.time() - start_time) * 1000)
        emitter.emit_trace_span(
            name="Lexical Analysis", 
            status="OK", 
            duration_ms=duration_ms, 
            queue_wait_ms=queue_wait_ms, 
            trace_id=job_data.get("trace_id"),
            retry_count=int(job_data.get("attempts", 0))
        )
        emitter.emit_verdict(
            domain=domain, 
            verdict=label.upper(), 
            confidence=confidence, 
            trace_id=job_data.get("trace_id")
        )

        logger.info(
            "domain_processed",
            domain=domain,
            label=label,
            confidence=confidence,
            analysis_type=result.get("analysis_type"),
        )
        metrics.increment("herald_jobs_processed_total", worker="domain", status="success")
    except SQLAlchemyError as exc:
        session.rollback()
        metrics.increment("herald_jobs_processed_total", worker="domain", status="database_error")
        logger.exception("database_save_failed", domain=domain)
        
        emitter.emit_event(
            event_type="DATABASE_PERSISTENCE_FAILED", 
            payload={"domain": domain, "error": str(exc)}, 
            priority="HIGH", 
            severity="CRITICAL",
            trace_id=job_data.get("trace_id")
        )
        
        if "OperationalError" in str(type(exc)) or "no such column" in str(exc):
            raise ValueError(f"SCHEMA_MISMATCH: {str(exc)}") from exc
            
        raise
    finally:
        session.close()
        clear_contextvars()


def start_queue_worker() -> None:
    init_worker()
    logger.info("worker_started", queue=DOMAIN_ANALYSIS_QUEUE.ready)
    if not redis_client or not domain_queue:
        logger.error("redis_not_available", service="queue_worker")
        return

    while True:
        leased_job_json = None
        job_data = None
        try:
            redis_client.set("worker:domain:last_seen", datetime.utcnow().isoformat())
            queue_result = domain_queue.dequeue(timeout=5)
            if not queue_result:
                continue

            leased_job_json, job_data = queue_result
            domain = job_data.get("domain", "unknown")

            if should_skip_duplicate(domain, job_data):
                logger.info("domain_skipped_idempotent", domain=domain)
                metrics.increment("herald_jobs_skipped_total", worker="domain", reason="duplicate")
                domain_queue.ack(leased_job_json)
                continue

            process_domain(job_data)
            domain_queue.ack(leased_job_json)
            metrics.increment("herald_jobs_ack_total", worker="domain")
        except Exception as exc:
            metrics.increment("herald_jobs_processed_total", worker="domain", status="error")
            logger.error(
                "queue_worker_error",
                job_id=job_data.get("job_id") if job_data else None,
                domain=job_data.get("domain") if job_data else None,
                error=str(exc),
            )
            if leased_job_json and job_data:
                emitter.emit_event(
                    event_type="RETRY_TRIGGERED", 
                    payload={"domain": job_data.get("domain"), "error": str(exc)}, 
                    priority="MEDIUM", 
                    severity="WARNING",
                    trace_id=job_data.get("trace_id")
                )
                if "SCHEMA_MISMATCH" in str(exc):
                    domain_queue.send_to_dlq(leased_job_json, job_data, exc)
                else:
                    domain_queue.retry_or_dlq(leased_job_json, job_data, exc)
            time.sleep(1)


if __name__ == "__main__":
    start_queue_worker()
