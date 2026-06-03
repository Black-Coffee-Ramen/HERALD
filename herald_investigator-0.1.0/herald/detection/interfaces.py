from typing import Protocol
from herald.detection.models import DetectionResult

class Scorer(Protocol):
    def score(self, domain: str) -> DetectionResult:
        ...
