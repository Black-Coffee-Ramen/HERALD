import redis
import json
import time
import os
import sys
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime

# Setup structured logging
from herald.utils.logging_config import setup_logging
import structlog

setup_logging()
logger = structlog.get_logger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from herald.predict_with_fallback import PhishingPredictorV3
from herald.db.models import DomainScan, DATABASE_URL, Whitelist

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
    redis_client.ping()
except redis.ConnectionError:
    logger.warning("redis_connection_failed", service="queue_worker")
    redis_client = None

def clear_failed_jobs():
    if redis_client:
        count = redis_client.llen("failed_jobs")
        redis_client.delete("failed_jobs")
        logger.info("failed_jobs_cleared", count=count)

# Initialize DB and predictor at module level
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
predictor = PhishingPredictorV3()

def process_domain(job_data):
    domain = job_data.get('domain')
    target_cse = job_data.get('target_cse', 'Unknown')
    source = job_data.get('source', 'manual')
    
    logger.info("processing_domain", domain=domain, target_cse=target_cse)
    
    # Save to PostgreSQL
    session = SessionLocal()
    try:
        # Check Whitelist First
        clean_domain = domain.lower().strip().replace('www.', '')
        is_whitelisted = session.query(Whitelist).filter(Whitelist.domain == clean_domain).first()
        
        if is_whitelisted:
            label = 'Clean'
            confidence = 0.01
            logger.info("domain_whitelisted_intercept", domain=domain)
        else:
            # Run ML prediction
            result = predictor.predict(domain)
            label = result.get('status', 'Unknown')
            confidence = float(result.get('ml_confidence', 0.0))
            
        # Check if domain already exists
        existing = session.query(DomainScan).filter_by(domain=domain).first()
        if existing:
            existing.label = label
            existing.confidence = confidence
            existing.target_cse = target_cse
            existing.scan_date = datetime.utcnow()
            existing.source = source
        else:
            scan = DomainScan(
                domain=domain,
                label=label,
                confidence=confidence,
                target_cse=target_cse,
                source=source,
                scan_date=datetime.utcnow()
            )
            session.add(scan)
        session.commit()
        logger.info("domain_processed", domain=domain, label=label, confidence=confidence)
        
        # Update heartbeat in Redis
        if redis_client:
            redis_client.set("worker:last_seen", datetime.utcnow().isoformat())
    except Exception as e:
        session.rollback()
        logger.error("database_save_failed", domain=domain, error=str(e))
        raise
    finally:
        session.close()

def start_queue_worker():
    logger.info("worker_started", queue="domain_analysis_queue")
    if not redis_client:
        logger.error("redis_not_available", service="queue_worker")
        return

    while True:
        try:
            if redis_client:
                redis_client.set("worker:last_seen", datetime.utcnow().isoformat())
            # Block until an item is available in the queue
            queue_result = redis_client.blpop("domain_analysis_queue", timeout=5)
            if not queue_result:
                continue

            _, job_json = queue_result
            job_data = json.loads(job_json)
            
            domain = job_data.get('domain', 'unknown')
            retries = job_data.get('retries', 0)
            
            # Exponential backoff check
            available_after = job_data.get('available_after', 0)
            if time.time() < available_after:
                # Job is not ready yet, push back and sleep briefly
                redis_client.rpush("domain_analysis_queue", json.dumps(job_data))
                time.sleep(1)
                continue
                
            # Idempotency check (only on first attempt)
            if retries == 0 and redis_client:
                today = datetime.utcnow().strftime("%Y-%m-%d")
                seen_key = f"domain:seen:{domain}:{today}"
                is_new = redis_client.setnx(seen_key, "1")
                if not is_new:
                    logger.info("domain_skipped_idempotent", domain=domain)
                    continue
                redis_client.expire(seen_key, 86400) # expire in 24 hours

            try:
                process_domain(job_data)
            except Exception as e:
                logger.error("processing_error", domain=domain, error=str(e))
                retries += 1
                job_data['retries'] = retries
                job_data['error'] = str(e)
                
                if retries >= 3:
                    logger.error("job_failed_final", domain=domain, retries=retries)
                    redis_client.rpush("failed_jobs", json.dumps(job_data))
                else:
                    # Exponential backoff: 30s, then 120s
                    delay = 30 if retries == 1 else 120
                    job_data['available_after'] = time.time() + delay
                    logger.info("job_requeued_delayed", domain=domain, retries=retries, delay=delay)
                    redis_client.rpush("domain_analysis_queue", json.dumps(job_data))

        except Exception as e:
            logger.error("queue_worker_error", error=str(e))
            time.sleep(5)

if __name__ == "__main__":
    start_queue_worker()
