import os
import time
from multiprocessing import Process, Queue
from queue import Empty
from datetime import datetime

import redis
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars
from sqlalchemy.exc import SQLAlchemyError

from herald.db.models import DomainScan, SessionLocal
from herald.monitoring.metrics import Timer, metrics
from herald.monitoring.resilience import CircuitBreakerConfig, CircuitOpenError, RedisCircuitBreaker
from herald.monitoring.redis_queue import RedisReliableQueue, VISUAL_ANALYSIS_QUEUE
from herald.utils.logging_config import setup_logging

setup_logging()
logger = structlog.get_logger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
VISUAL_ANALYSIS_TIMEOUT_SECONDS = int(os.getenv("VISUAL_ANALYSIS_TIMEOUT_SECONDS", "45"))
CIRCUIT_FAILURE_THRESHOLD = int(os.getenv("VISUAL_CIRCUIT_FAILURE_THRESHOLD", "5"))
CIRCUIT_RESET_SECONDS = int(os.getenv("VISUAL_CIRCUIT_RESET_SECONDS", "300"))


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
    logger.warning("redis_connection_failed", service="visual_worker")
    redis_client = None

visual_queue = RedisReliableQueue(redis_client, VISUAL_ANALYSIS_QUEUE, lease_seconds=900) if redis_client else None
visual_circuit = (
    RedisCircuitBreaker(
        redis_client,
        CircuitBreakerConfig(
            name="visual_analysis",
            failure_threshold=CIRCUIT_FAILURE_THRESHOLD,
            reset_seconds=CIRCUIT_RESET_SECONDS,
        ),
    )
    if redis_client
    else None
)


def run_visual_analysis_child(result_queue: Queue, domain: str, target_cse: str, initial_confidence: float) -> None:
    try:
        from herald.predict_with_fallback import PhishingPredictorV3

        predictor = PhishingPredictorV3()
        result_queue.put({"ok": True, "result": predictor.analyze_visual_fallback(domain, target_cse, initial_confidence)})
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})


def run_visual_analysis_with_timeout(domain: str, target_cse: str, initial_confidence: float) -> dict:
    result_queue: Queue = Queue(maxsize=1)
    process = Process(
        target=run_visual_analysis_child,
        args=(result_queue, domain, target_cse, initial_confidence),
        daemon=True,
    )
    process.start()
    process.join(VISUAL_ANALYSIS_TIMEOUT_SECONDS)

    if process.is_alive():
        process.terminate()
        process.join(5)
        raise TimeoutError(f"visual analysis exceeded {VISUAL_ANALYSIS_TIMEOUT_SECONDS}s")

    try:
        payload = result_queue.get_nowait()
    except Empty as exc:
        raise RuntimeError("visual analysis returned no result") from exc

    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "visual analysis failed"))

    return payload["result"]


def apply_visual_result(job_data: dict) -> None:
    domain = job_data.get("domain")
    if not domain:
        raise ValueError("job is missing required domain")

    target_cse = job_data.get("target_cse", "Unknown")
    initial_confidence = float(job_data.get("initial_confidence", 0.0))

    bind_contextvars(trace_id=job_data.get("trace_id"), job_id=job_data.get("job_id"))
    logger.info("visual_processing_started", domain=domain, target_cse=target_cse)

    if visual_circuit:
        visual_circuit.ensure_allowed()

    with Timer("herald_visual_processing_seconds", worker="visual"):
        visual_result = run_visual_analysis_with_timeout(domain, target_cse, initial_confidence)
        is_confirmed = bool(visual_result.get("cv_ocr_confirmed"))

    session = SessionLocal()
    try:
        scan = session.query(DomainScan).filter_by(domain=domain).first()
        if not scan:
            logger.warning("visual_scan_record_missing", domain=domain)
            return

        if is_confirmed:
            scan.label = "Phishing"
            scan.confidence = float(visual_result.get("final_confidence", 1.0))
        else:
            scan.label = "Suspected"
            scan.confidence = float(visual_result.get("final_confidence", initial_confidence))

        scan.scan_date = datetime.utcnow()
        session.commit()
        if visual_circuit:
            visual_circuit.record_success()
        metrics.increment("herald_jobs_processed_total", worker="visual", status="success")
        logger.info(
            "visual_processing_finished",
            domain=domain,
            label=scan.label,
            confidence=scan.confidence,
            status=visual_result.get("cv_ocr_status"),
        )
    except SQLAlchemyError:
        session.rollback()
        metrics.increment("herald_jobs_processed_total", worker="visual", status="database_error")
        logger.exception("visual_database_save_failed", domain=domain)
        raise
    finally:
        session.close()
        clear_contextvars()


def mark_visual_degraded(job_data: dict, reason: str) -> None:
    domain = job_data.get("domain")
    if not domain:
        return

    session = SessionLocal()
    try:
        scan = session.query(DomainScan).filter_by(domain=domain).first()
        if scan:
            scan.label = "Suspected"
            scan.scan_date = datetime.utcnow()
            session.commit()
        logger.warning("visual_analysis_degraded", domain=domain, reason=reason)
        metrics.increment("herald_visual_degraded_total", reason=reason)
    finally:
        session.close()


def start_visual_worker() -> None:
    logger.info("worker_started", queue=VISUAL_ANALYSIS_QUEUE.ready)
    if not redis_client or not visual_queue:
        logger.error("redis_not_available", service="visual_worker")
        return

    while True:
        leased_job_json = None
        job_data = None
        try:
            redis_client.set("worker:visual:last_seen", datetime.utcnow().isoformat())
            queue_result = visual_queue.dequeue(timeout=5)
            if not queue_result:
                continue

            leased_job_json, job_data = queue_result
            bind_contextvars(trace_id=job_data.get("trace_id"), job_id=job_data.get("job_id"))
            try:
                apply_visual_result(job_data)
            except CircuitOpenError as exc:
                mark_visual_degraded(job_data, "circuit_open")
                logger.warning("visual_circuit_open_degraded", error=str(exc))
            except TimeoutError as exc:
                if visual_circuit:
                    visual_circuit.record_failure()
                mark_visual_degraded(job_data, "timeout")
                logger.warning("visual_timeout_degraded", error=str(exc))
            except Exception:
                if visual_circuit:
                    visual_circuit.record_failure()
                raise
            visual_queue.ack(leased_job_json)
            metrics.increment("herald_jobs_ack_total", worker="visual")
        except Exception as exc:
            metrics.increment("herald_jobs_processed_total", worker="visual", status="error")
            logger.error(
                "visual_worker_error",
                job_id=job_data.get("job_id") if job_data else None,
                domain=job_data.get("domain") if job_data else None,
                error=str(exc),
            )
            if leased_job_json and job_data:
                visual_queue.retry_or_dlq(leased_job_json, job_data, exc)
            time.sleep(1)
        finally:
            clear_contextvars()


if __name__ == "__main__":
    start_visual_worker()
