"""
Base agent interface. All detection agents inherit from this.
"""
from abc import ABC, abstractmethod
from certusdoc.models import Document, AgentResult


class BaseAgent(ABC):
    """Abstract base class for all CertusDoc detection agents."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def analyze(self, document: Document) -> AgentResult:
        """
        Analyze a document and return detection results.
        
        Args:
            document: Ingested document with pages, OCR data, and metadata.
            
        Returns:
            AgentResult with score, reliability weight, findings, and optional heatmap.
        """
        pass

    def _compute_reliability(self, document: Document) -> float:
        """
        Compute how reliable this agent's output is for the given document.
        Override in subclasses for agent-specific reliability logic.
        """
        return 1.0
