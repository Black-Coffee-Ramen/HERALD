import time
from typing import Any, Optional
import structlog
from herald.telemetry.schemas import EventEnvelope, PriorityLevel, SeverityLevel
from herald.telemetry.stream import TelemetryStream

logger = structlog.get_logger(__name__)

class TelemetryEmitter:
    """
    Singleton service for cleanly emitting telemetry envelopes across the platform.
    """
    _instance: Optional['TelemetryEmitter'] = None
    
    def __init__(self, stream: TelemetryStream, worker_type: str = "unknown"):
        self.stream = stream
        self.worker_type = worker_type

    @classmethod
    def initialize(cls, stream: TelemetryStream, worker_type: str) -> 'TelemetryEmitter':
        cls._instance = cls(stream, worker_type)
        return cls._instance

    @classmethod
    def get(cls) -> 'TelemetryEmitter':
        if not cls._instance:
            # Fallback mock/noop stream if not initialized
            cls._instance = cls(TelemetryStream(None), "uninitialized_worker")
        return cls._instance

    def emit_event(
        self, 
        event_type: str, 
        payload: Any, 
        priority: PriorityLevel = "LOW", 
        severity: SeverityLevel = "INFO",
        trace_id: Optional[str] = None,
        degraded_state: bool = False
    ) -> None:
        """
        Constructs the standard EventEnvelope and dispatches it via the stream.
        """
        try:
            envelope = EventEnvelope(
                event_type=event_type,
                payload=payload,
                worker_type=self.worker_type,
                severity=severity,
                telemetry_priority=priority,
                trace_id=trace_id,
                degraded_state=degraded_state,
                source_service="herald.backend"
            )
            
            # Operational JSON logging mimicking the envelope (Structured Logging task)
            logger.info(
                "telemetry_event_emitted",
                event_id=envelope.event_id,
                event_type=envelope.event_type,
                trace_id=envelope.trace_id,
                telemetry_priority=envelope.telemetry_priority,
                severity=envelope.severity
            )
            
            self.stream.publish(envelope)
        except Exception as e:
            logger.warning("telemetry_emission_failed", error=str(e), event_type=event_type)

    def emit_trace_span(
        self, 
        name: str, 
        status: str = "OK", 
        duration_ms: int = 0, 
        queue_wait_ms: int = 0,
        trace_id: Optional[str] = None,
        message: str = "",
        retry_count: int = 0
    ) -> None:
        """
        Helper method to specifically emit worker lifecycle spans.
        """
        payload = {
            "id": f"span-{time.time_ns()}",
            "name": name,
            "status": status,
            "workerType": self.worker_type,
            "durationMs": duration_ms,
            "queueWaitMs": queue_wait_ms,
            "startTime": time.time() * 1000 - duration_ms,
            "message": message,
            "retryCount": retry_count
        }
        
        priority: PriorityLevel = "LOW"
        severity: SeverityLevel = "INFO"
        
        if status == "ERROR":
            priority = "MEDIUM"
            severity = "ERROR"
        elif retry_count > 0:
            priority = "MEDIUM"
            severity = "WARNING"
            
        self.emit_event(
            event_type="TRACE_SPAN_COMPLETED", 
            payload=payload, 
            priority=priority, 
            severity=severity,
            trace_id=trace_id
        )

    def emit_verdict(self, domain: str, verdict: str, confidence: float, trace_id: Optional[str] = None) -> None:
        """
        Helper method to specifically emit critical threat intelligence verdicts.
        """
        payload = {
            "id": f"threat-{time.time_ns()}",
            "domain": domain,
            "verdict": verdict,
            "confidence": confidence,
            "source": self.worker_type,
            "timestamp": time.time() * 1000
        }
        
        # Threat Intelligence is always HIGH priority
        self.emit_event(
            event_type="THREAT_DETECTED", 
            payload=payload, 
            priority="HIGH", 
            severity="WARNING" if verdict in ["MALICIOUS", "SUSPICIOUS"] else "INFO",
            trace_id=trace_id
        )
