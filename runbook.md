# HERALD Production Runbook

This document outlines the standard operating procedures for the HERALD Phishing Detection System. 
It provides step-by-step instructions for troubleshooting and resolving common operational issues.

## System Architecture Overview
- **API Node:** FastAPI server handling ingress (Port 8000).
- **Worker Node:** Background queue workers for ML processing.
- **Redis Node:** In-memory queue broker and cache (Port 6379 internally).
- **Database Node:** PostgreSQL / SQLite storing detections and configurations.

---

## 1. Service Crash: API Node (`herald-api`)

**Symptoms:**
- Health monitor script sends an "API Offline" alert.
- Requests to `http://localhost:8000/api/health` return connection refused or 5xx errors.

**Recovery Steps:**
1. Check the logs to identify the crash reason:
   ```bash
   docker-compose logs --tail=100 api
   ```
2. If it's an Out-Of-Memory (OOM) kill, the Docker daemon will automatically attempt to restart it. Monitor if it enters a crash loop.
3. Manually restart the API service:
   ```bash
   docker-compose restart api
   ```
4. Verify recovery by pinging `/api/health`.

---

## 2. Service Crash: Worker Node (`herald-worker`)

**Symptoms:**
- `/api/health` reports the worker status as `stale` (last seen > 5 minutes ago).
- The Redis `queue_depth` is continuously increasing without dropping.

**Recovery Steps:**
1. Check the worker logs for unhandled exceptions or ML model loading errors:
   ```bash
   docker-compose logs --tail=100 worker
   ```
2. Restart the worker service:
   ```bash
   docker-compose restart worker
   ```
3. Monitor `/api/health` to ensure `queue_depth` begins decreasing and worker `last_seen` timestamp updates.

---

## 3. Service Crash: Redis Queue (`herald-redis`)

**Symptoms:**
- API node returns `500 Internal Server Error: Redis queue is offline` when attempting to scan.
- Worker node logs display `redis_connection_failed`.

**Recovery Steps:**
1. Restart the Redis service:
   ```bash
   docker-compose restart redis
   ```
2. **Data Integrity Check:** We have AOF persistence enabled. Upon restart, Redis will reload the queue state from `appendonly.aof`. 
3. Check the `failed_jobs` queue. Any jobs that were mid-processing during the crash might need to be requeued:
   ```bash
   curl -X POST http://localhost:8000/api/admin/failed-jobs/retry
   ```

---

## 4. High False Positive Rate Alert

**Symptoms:**
- The weekly report indicates precision has dropped below 90%.
- Analysts are manually marking an unusually high number of `FP` (False Positive) verdicts.

**Recovery Steps:**
1. **Immediate Mitigation:** Do not retrain the model immediately. Instead, add the highly-targeted legitimate domains to the whitelist to stop the bleeding:
   ```bash
   curl -X POST http://localhost:8000/api/whitelist -H "Content-Type: application/json" -d '{"domain": "legit-domain.com", "reason": "FP Spike Mitigation"}'
   ```
2. Extract the recent False Positives from the database.
3. Analyze if the FPs share a common new network pattern or specific keyword. 
4. Accumulate at least 500 new confirmed ground-truth labels before initiating an ML retraining cycle.

---

## 5. Model Rollback Procedure

If a newly deployed ML model performs poorly in production:
1. Re-link the active model symlink to the previous version:
   ```bash
   ln -sf models/ensemble_v6.joblib models/ensemble_active.joblib
   ```
2. Restart the worker container so it loads the old model into memory:
   ```bash
   docker-compose restart worker
   ```
