import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import redis
import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class QueueNames:
    ready: str
    processing: str
    delayed: str
    dlq: str


DOMAIN_ANALYSIS_QUEUE = QueueNames(
    ready="domain_analysis_queue",
    processing="domain_analysis_queue:processing",
    delayed="domain_analysis_queue:delayed",
    dlq="domain_analysis_queue:dlq",
)

VISUAL_ANALYSIS_QUEUE = QueueNames(
    ready="visual_analysis_queue",
    processing="visual_analysis_queue:processing",
    delayed="visual_analysis_queue:delayed",
    dlq="visual_analysis_queue:dlq",
)


class RedisReliableQueue:
    """
    Redis list based queue with a small reliability envelope:
    - BRPOPLPUSH moves work into a processing list before execution.
    - ack removes the exact processing payload only after successful handling.
    - delayed retries are kept in a sorted set until their due timestamp.
    - stale processing leases are requeued after worker crashes.

    This is not a Kafka replacement, but it closes the largest loss window in
    the current Redis queue while keeping the migration small.
    """

    def __init__(
        self,
        client: redis.Redis,
        names: QueueNames,
        *,
        lease_seconds: int = 300,
        max_retries: int = 3,
    ):
        self.client = client
        self.names = names
        self.lease_seconds = lease_seconds
        self.max_retries = max_retries

    def enqueue(self, payload: dict[str, Any], *, delay_seconds: int = 0) -> str:
        job = dict(payload)
        job.setdefault("job_id", str(uuid.uuid4()))
        job.setdefault("trace_id", job["job_id"])
        job.setdefault("attempts", 0)
        job["enqueued_at"] = time.time()
        job_json = json.dumps(job, sort_keys=True)

        if delay_seconds > 0:
            self.client.zadd(self.names.delayed, {job_json: time.time() + delay_seconds})
        else:
            self.client.lpush(self.names.ready, job_json)

        return job["job_id"]

    def dequeue(self, *, timeout: int = 5) -> Optional[tuple[str, dict[str, Any]]]:
        self.promote_due_jobs(limit=100)
        self.reclaim_expired(limit=100)

        job_json = self.client.brpoplpush(
            self.names.ready,
            self.names.processing,
            timeout=timeout,
        )
        if not job_json:
            return None

        try:
            job = json.loads(job_json)
        except json.JSONDecodeError:
            logger.error("queue_payload_decode_failed", queue=self.names.ready)
            self.client.lrem(self.names.processing, 1, job_json)
            self.client.rpush(self.names.dlq, job_json)
            return None

        job["lease_expires_at"] = time.time() + self.lease_seconds
        leased_json = json.dumps(job, sort_keys=True)
        pipe = self.client.pipeline()
        pipe.lrem(self.names.processing, 1, job_json)
        pipe.lpush(self.names.processing, leased_json)
        pipe.execute()
        return leased_json, job

    def ack(self, leased_job_json: str) -> None:
        self.client.lrem(self.names.processing, 1, leased_job_json)

    def retry_or_dlq(self, leased_job_json: str, job: dict[str, Any], error: Exception) -> None:
        self.client.lrem(self.names.processing, 1, leased_job_json)

        attempts = int(job.get("attempts", 0)) + 1
        job["attempts"] = attempts
        job["last_error"] = str(error)
        job["last_failed_at"] = time.time()
        job.pop("lease_expires_at", None)

        if attempts >= self.max_retries:
            self.client.rpush(self.names.dlq, json.dumps(job, sort_keys=True))
            logger.error(
                "queue_job_dead_lettered",
                queue=self.names.ready,
                job_id=job.get("job_id"),
                attempts=attempts,
                error=str(error),
            )
            return

        delay = min(30 * (2 ** (attempts - 1)), 300)
        self.enqueue(job, delay_seconds=delay)
        logger.info(
            "queue_job_retried",
            queue=self.names.ready,
            job_id=job.get("job_id"),
            attempts=attempts,
            delay_seconds=delay,
        )

    def promote_due_jobs(self, *, limit: int = 100) -> int:
        now = time.time()
        due_jobs = self.client.zrangebyscore(self.names.delayed, 0, now, start=0, num=limit)
        if not due_jobs:
            return 0

        pipe = self.client.pipeline()
        for job_json in due_jobs:
            pipe.zrem(self.names.delayed, job_json)
            pipe.lpush(self.names.ready, job_json)
        pipe.execute()
        return len(due_jobs)

    def reclaim_expired(self, *, limit: int = 100) -> int:
        processing_jobs = self.client.lrange(self.names.processing, 0, limit - 1)
        reclaimed = 0
        now = time.time()

        for job_json in processing_jobs:
            try:
                job = json.loads(job_json)
            except json.JSONDecodeError:
                self.client.lrem(self.names.processing, 1, job_json)
                self.client.rpush(self.names.dlq, job_json)
                continue

            if float(job.get("lease_expires_at", now + 1)) > now:
                continue

            job.pop("lease_expires_at", None)
            pipe = self.client.pipeline()
            pipe.lrem(self.names.processing, 1, job_json)
            pipe.lpush(self.names.ready, json.dumps(job, sort_keys=True))
            pipe.execute()
            reclaimed += 1

        return reclaimed

    def depth(self) -> dict[str, int]:
        return {
            "ready": self.client.llen(self.names.ready),
            "processing": self.client.llen(self.names.processing),
            "delayed": self.client.zcard(self.names.delayed),
            "dlq": self.client.llen(self.names.dlq),
        }

    def drain_dlq_to_ready(self) -> int:
        jobs = self.client.lrange(self.names.dlq, 0, -1)
        if not jobs:
            return 0

        pipe = self.client.pipeline()
        pipe.delete(self.names.dlq)
        for job_json in jobs:
            try:
                job = json.loads(job_json)
            except json.JSONDecodeError:
                continue
            job["attempts"] = 0
            job.pop("last_error", None)
            job.pop("last_failed_at", None)
            job.pop("lease_expires_at", None)
            pipe.lpush(self.names.ready, json.dumps(job, sort_keys=True))
        pipe.execute()
        return len(jobs)
