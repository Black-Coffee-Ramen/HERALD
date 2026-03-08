import threading
import time
import logging

from herald.ingestion.certstream_monitor import start_certstream
from herald.ingestion.new_domains_monitor import start_polling
from herald.ingestion.social_monitor import run_social_monitor as start_social_monitor
from herald.monitoring.scheduler import scheduler, check_suspected_domains
from herald.monitoring.queue_worker import start_queue_worker

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s %(asctime)s - %(message)s')

def run_certstream():
    logging.info("Starting Certstream thread...")
    start_certstream()

def run_new_domains_polling():
    logging.info("Starting New Domains polling thread...")
    start_polling(interval_hours=24)

def run_scheduler():
    logging.info("Starting APScheduler thread...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass

def run_queue_worker():
    logging.info("Starting Queue Worker thread...")
    start_queue_worker()

def run_social_monitor():
    logging.info("Starting Social Monitor thread...")
    start_social_monitor()

def start_all_workers():
    # Wait a few seconds for Redis/DB to spin up in Docker before starting loops
    time.sleep(5)
    logging.info("Initializing Workers for Phishing Detection Platform...")

    t1 = threading.Thread(target=run_certstream, daemon=True)
    t2 = threading.Thread(target=run_new_domains_polling, daemon=True)
    t3 = threading.Thread(target=run_scheduler, daemon=True)
    t4 = threading.Thread(target=run_queue_worker, daemon=True)
    t5 = threading.Thread(target=run_social_monitor, daemon=True)

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()

    logging.info("All worker threads launched. Awaiting termination signal.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down workers...")

if __name__ == "__main__":
    start_all_workers()
