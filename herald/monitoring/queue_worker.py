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
from herald.predict_with_fallback import PhishingPredictorV3
from herald.utils.logging_config import setup_logging

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
        socket_timeout=5,
        socket_connect_timeout=5,
        health_check_interval=30,
    )
    client.ping()
    return client


try:
    redis_client = build_redis_client()
except redis.RedisError:
    logger.warning("redis_connection_failed", service="queue_worker")
    redis_client = None

domain_queue = RedisReliableQueue(redis_client, DOMAIN_ANALYSIS_QUEUE) if redis_client else None
visual_queue = RedisReliableQueue(redis_client, VISUAL_ANALYSIS_QUEUE) if redis_client else None
predictor = PhishingPredictorV3()


def normalize_domain(domain: str) -> str:
    return domain.lower().strip().replace("www.", "")


def should_skip_duplicate(domain: str, job: dict) -> bool:
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


def upsert_domain_scan(session, *, domain: str, label: str, confidence: float, target_cse: str, source: str) -> None:
    existing = session.query(DomainScan).filter_by(domain=domain).first()
    if existing:
        existing.label = label
        existing.confidence = confidence
        existing.target_cse = target_cse
        existing.scan_date = datetime.utcnow()
        existing.source = source
        return

    session.add(
        DomainScan(
            domain=domain,
            label=label,
            confidence=confidence,
            target_cse=target_cse,
            source=source,
            scan_date=datetime.utcnow(),
        )
    )


def enqueue_visual_analysis(domain: str, result: dict, source: str) -> None:
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
    domain = job_data.get("domain")
    if not domain:
        raise ValueError("job is missing required domain")

    target_cse = job_data.get("target_cse", "Unknown")
    source = job_data.get("source", "manual")

    bind_contextvars(trace_id=job_data.get("trace_id"), job_id=job_data.get("job_id"))
    logger.info("processing_domain", domain=domain, target_cse=target_cse, source=source)

    session = SessionLocal()
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
                result = predictor.predict(domain, cse_name=target_cse, include_visual=False)
                label = result.get("status", "Unknown")
                confidence = float(result.get("ml_confidence_adjusted", result.get("ml_confidence", 0.0)))

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

        logger.info(
            "domain_processed",
            domain=domain,
            label=label,
            confidence=confidence,
            analysis_type=result.get("analysis_type"),
        )
        metrics.increment("herald_jobs_processed_total", worker="domain", status="success")
    except SQLAlchemyError:
        session.rollback()
        metrics.increment("herald_jobs_processed_total", worker="domain", status="database_error")
        logger.exception("database_save_failed", domain=domain)
        raise
    finally:
        session.close()
        clear_contextvars()


def start_queue_worker() -> None:
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
                domain_queue.retry_or_dlq(leased_job_json, job_data, exc)
            time.sleep(1)


if __name__ == "__main__":
    start_queue_worker()
