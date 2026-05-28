from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, timedelta
import redis
import json
import logging
import os

from herald.db.models import DomainScan, SessionLocal

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

# Connect to Redis — use env var so it works both locally and in Docker
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

_redis_client = None

def get_redis_client():
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
            _redis_client.ping()
            logging.info("Scheduler connected to Redis.")
        except redis.ConnectionError:
            logging.warning("Could not connect to Redis. Ensure Redis is running for production.")
            _redis_client = None
    return _redis_client

def __getattr__(name):
    if name == 'redis_client':
        return get_redis_client()
    raise AttributeError(f"module {__name__} has no attribute {name}")

def check_suspected_domains(n_days=7):
    """
    Query the database for 'Suspected' domains older than n_days and push them
    back to the Redis queue for full re-classification.
    """
    logging.info(f"Running periodic check for 'Suspected' domains older than {n_days} days...")
    session = SessionLocal()
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=n_days)
        suspected_domains = session.query(DomainScan).filter(
            DomainScan.label == 'Suspected',
            DomainScan.scan_date < cutoff_date
        ).all()
        logging.info(f"Found {len(suspected_domains)} suspected domains requiring re-scan.")
        if redis_client:
            for ds in suspected_domains:
                job_data = json.dumps({
                    "domain": ds.domain,
                    "source": "scheduler_rescan",
                    "target_cse": ds.target_cse
                })
                redis_client.rpush("domain_analysis_queue", job_data)
                ds.scan_date = datetime.utcnow()
            session.commit()
    except Exception as e:
        logging.error(f"Error querying database: {e}")
    finally:
        session.close()

# Module-level scheduler — importable by run_workers.py
n_days_interval = int(os.getenv("RESCAN_INTERVAL_DAYS", 7))

scheduler = BlockingScheduler()
scheduler.add_job(
    func=check_suspected_domains,
    trigger=IntervalTrigger(hours=24),
    args=[n_days_interval],
    id='recheck_suspected_domains',
    name=f"Recheck suspected domains older than {n_days_interval} days",
    replace_existing=True
)

if __name__ == "__main__":
    logging.info("Started APScheduler for Suspected domain re-monitoring.")
    check_suspected_domains(n_days=n_days_interval)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
