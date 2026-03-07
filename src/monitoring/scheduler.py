from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, timedelta
import redis
import json
import logging
import sys
import os

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.db.models import DomainScan, DATABASE_URL

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

# Connect to Redis
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logging.info("Connected to Redis queue.")
except redis.ConnectionError:
    logging.warning("Could not connect to Redis. Ensure Redis is running for production.")
    redis_client = None

# Connect DB
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def check_suspected_domains(n_days=7):
    """
    Query the database for 'Suspected' domains older than n_days and push them
    back to the Redis queue for full re-classification.
    """
    logging.info(f"Running periodic check for 'Suspected' domains older than {n_days} days...")
    
    session = SessionLocal()
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=n_days)
        
        # Find suspected domains checked before cutoff_date
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
                
                # Update scan date so we don't spam it until it fails again
                ds.scan_date = datetime.utcnow()
            
            session.commit()
            
    except Exception as e:
        logging.error(f"Error querying database: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    
    # Run the job every 24 hours (can be adjusted)
    n_days_interval = int(os.getenv("RESCAN_INTERVAL_DAYS", 7))
    
    scheduler.add_job(
        func=check_suspected_domains,
        trigger=IntervalTrigger(hours=24),
        args=[n_days_interval],
        id='recheck_suspected_domains',
        name=f"Recheck suspected domains older than {n_days_interval} days",
        replace_existing=True
    )
    
    logging.info("Started APScheduler for Suspected domain re-monitoring.")
    
    # Run it once immediately on startup
    check_suspected_domains(n_days=n_days_interval)
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
