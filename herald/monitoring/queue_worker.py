import redis
import json
import logging
import time
import os
import sys

from herald.predict_with_fallback import PhishingPredictorV3
from herald.db.models import DomainScan, DATABASE_URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s %(asctime)s - %(message)s')

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
    redis_client.ping()
except redis.ConnectionError:
    logging.warning("Could not connect to Redis from Queue Worker.")
    redis_client = None

def clear_failed_jobs():
    if redis_client:
        count = redis_client.llen("failed_jobs")
        redis_client.delete("failed_jobs")
        logging.info(f"Cleared {count} failed jobs from dead letter queue")

# Initialize DB and predictor at module level (outside the function)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
predictor = PhishingPredictorV3()

def process_domain(domain_data):
    domain = domain_data.get('domain')
    target_cse = domain_data.get('target_cse', 'Unknown')
    source = domain_data.get('source', 'manual')
    
    logging.info(f"Processing domain: {domain}")
    
    # Run ML prediction
    result = predictor.predict(domain)
    
    label = result.get('status', 'Unknown')
    confidence = float(result.get('ml_confidence', 0.0))
    
    # Save to PostgreSQL
    session = SessionLocal()
    try:
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
        logging.info(f"Successfully processed and saved {domain} → {label} ({confidence:.2f})")
    except Exception as e:
        session.rollback()
        logging.error(f"Database save failed for {domain}: {e}")
        raise
    finally:
        session.close()

def start_queue_worker():
    logging.info("Starting Queue Worker to process domain_analysis_queue...")
    if not redis_client:
        logging.error("Redis client not available. Exiting Queue Worker.")
        return

    while True:
        try:
            # Block until an item is available in the queue
            queue_result = redis_client.blpop("domain_analysis_queue", timeout=5)
            if not queue_result:
                continue

            _, job_json = queue_result
            job_data = json.loads(job_json)
            
            domain = job_data.get('domain', 'unknown')
            retries = job_data.get('retries', 0)

            try:
                process_domain(job_data)
            except Exception as e:
                logging.error(f"Error processing {domain}: {e}")
                retries += 1
                job_data['retries'] = retries
                job_data['error'] = str(e)
                
                if retries >= 3:
                    logging.error(f"Domain {domain} failed 3 times. Moving to failed_jobs dead letter queue.")
                    redis_client.rpush("failed_jobs", json.dumps(job_data))
                else:
                    logging.info(f"Re-queueing {domain} (Attempt {retries}/3)")
                    # Give it a small delay before putting it back (or put it at the back)
                    redis_client.rpush("domain_analysis_queue", json.dumps(job_data))

        except Exception as e:
            logging.error(f"Queue worker encountered an error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    start_queue_worker()
