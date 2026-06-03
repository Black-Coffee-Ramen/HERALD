from pydantic import BaseModel, Field
from typing import Optional, Any, Literal
from datetime import datetime
import uuid

SeverityLevel = Literal["INFO", "WARNING", "ERROR", "CRITICAL"]
PriorityLevel = Literal["HIGH", "MEDIUM", "LOW"]

class EventEnvelope(BaseModel):
    """
    Standard Telemetry Event Envelope.
    Guarantees strict schema consistency between FastAPI backend and Next.js frontend.
    """
    event_id: str = Field(default_factory=lambda: f"evt-{uuid.uuid4().hex[:12]}")
    event_type: str
    trace_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    worker_type: str
    severity: SeverityLevel
    telemetry_priority: PriorityLevel
    degraded_state: bool = False
    source_service: str
    payload: Any
    version: str = "1.0"
