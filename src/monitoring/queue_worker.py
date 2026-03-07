import redis
import json
import logging
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s %(asctime)s - %(message)s')

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
    redis_client.ping()
except redis.ConnectionError:
    logging.warning("Could not connect to Redis from Queue Worker.")
    redis_client = None

def process_domain(domain_data):
    # This is where we would call the Enhanced Phishing Pipeline
    # For now, we simulate processing that might fail
    domain = domain_data.get('domain')
    logging.info(f"Processing domain: {domain}")
    
    # Simulate a failure condition randomly or intentionally based on domain name
    if "fail" in domain.lower():
        raise Exception(f"Simulated failure for {domain}")
    
    # Success path
    logging.info(f"Successfully processed {domain}")

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
