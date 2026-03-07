"""
HERALD: AI-Based Phishing Detection for Critical Infrastructure.

Provides tools to detect phishing attempts in emails and URLs using a
combination of rule-based heuristics and machine-learning models.
"""

from herald.detector import PhishingDetector
from herald.result import DetectionResult

__all__ = ["PhishingDetector", "DetectionResult"]
__version__ = "0.1.0"
