"""
CertusDoc data models — shared across all agents and pipeline stages.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class ForgeryType(Enum):
    """Types of forgery that can be detected."""
    NONE = "none"
    TEXT_SPLICING = "text_splicing"
    COPY_MOVE = "copy_move"
    METADATA_TAMPERING = "metadata_tampering"
    FONT_MISMATCH = "font_mismatch"
    IMAGE_MANIPULATION = "image_manipulation"
    JPEG_RECOMPRESSION = "jpeg_recompression"
    UNKNOWN = "unknown"


class RiskLevel(Enum):
    """Risk classification based on Document Integrity Score."""
    AUTHENTIC = "AUTHENTIC"       # DIS >= 0.80
    LOW_RISK = "LOW RISK"         # 0.65 <= DIS < 0.80
    MEDIUM_RISK = "MEDIUM RISK"   # 0.40 <= DIS < 0.65
    HIGH_RISK = "HIGH RISK"       # DIS < 0.40


@dataclass
class Document:
    """Represents an ingested document ready for analysis."""
    file_path: str
    file_name: str
    file_size_bytes: int
    pages: list[np.ndarray]              # List of page images (BGR, 300 DPI)
    ocr_text: list[str]                  # OCR text per page
    ocr_confidence: list[float]          # Average OCR confidence per page (0-100)
    ocr_word_data: list[list[dict]]      # Per-word OCR data (text, bbox, conf) per page
    metadata: dict                        # PDF/EXIF metadata
    original_format: str                  # "pdf", "png", "jpg", etc.


@dataclass
class AgentFinding:
    """A single finding from a detection agent."""
    description: str
    severity: float          # 0.0 (benign) to 1.0 (critical)
    region: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h) bounding box
    page: int = 0


@dataclass
class AgentResult:
    """Standardized output from any detection agent."""
    agent_name: str
    score: float                          # 0.0 = definitely forged, 1.0 = definitely authentic
    reliability_weight: float             # 0.0 to 1.0, how much to trust this agent
    findings: list[AgentFinding] = field(default_factory=list)
    heatmap: Optional[np.ndarray] = None  # Anomaly heatmap (grayscale, same dims as input)
    processing_time_ms: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class ForensicReport:
    """Final output of the CertusDoc pipeline."""
    document: Document
    agent_results: list[AgentResult]
    dis_score: float                      # Document Integrity Score (0-1)
    risk_level: RiskLevel
    primary_forgery_type: ForgeryType
    recommended_action: str
    fused_heatmap: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0

    @staticmethod
    def classify_risk(dis: float) -> RiskLevel:
        if dis >= 0.80:
            return RiskLevel.AUTHENTIC
        elif dis >= 0.65:
            return RiskLevel.LOW_RISK
        elif dis >= 0.40:
            return RiskLevel.MEDIUM_RISK
        else:
            return RiskLevel.HIGH_RISK
