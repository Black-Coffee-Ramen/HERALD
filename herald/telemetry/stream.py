import json
import logging
from typing import Optional
import structlog
from redis import Redis
from herald.telemetry.schemas import EventEnvelope

logger = structlog.get_logger(__name__)

class TelemetryStream:
    """
    Redis Pub/Sub abstraction for telemetry event transportation.
    """
    def __init__(self, redis_client: Optional[Redis], channel: str = "herald.telemetry"):
        self.redis = redis_client
        self.channel = channel

    def publish(self, envelope: EventEnvelope) -> None:
        """
        Synchronous publish used by workers.
        Fail-open: If Redis is unavailable, log locally and skip emission to prevent worker crashes.
        """
        if not self.redis:
            return
        
        try:
            # Pydantic v2 compatible dict dump
            payload_str = envelope.model_dump_json()
            self.redis.publish(self.channel, payload_str)
        except Exception as e:
            # Operational degradation fallback: local log only
            logger.warning("telemetry_stream_publish_failed", error=str(e), event_type=envelope.event_type)

