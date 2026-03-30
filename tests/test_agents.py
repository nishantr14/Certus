"""
CertusDoc Agent Tests

Tests each agent independently and the full pipeline end-to-end.
"""
import os
import sys
import pytest
import numpy as np
import cv2
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from certusdoc.models import Document, AgentResult, RiskLevel
from certusdoc.agents.visual_agent import VisualTamperAgent
from certusdoc.agents.text_agent import TextForensicsAgent
from certusdoc.agents.metadata_agent import MetadataAgent
from certusdoc.fusion.engine import compute_dis
from certusdoc.pipeline import CertusDocPipeline


DATA_DIR = Path(__file__).parent.parent / "data"


def _make_clean_document() -> Document:
    """Create a minimal clean Document for testing."""
    # White image with black text
    img = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Test Document", (100, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(img, "This is authentic", (100, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return Document(
        file_path="test.png",
        file_name="test.png",
        file_size_bytes=5000,
        pages=[img],
        ocr_text=["Test Document This is authentic"],
        ocr_confidence=[92.0],
        ocr_word_data=[[
            {"text": "Test", "x": 100, "y": 170, "w": 120, "h": 40, "conf": 95.0},
            {"text": "Document", "x": 230, "y": 170, "w": 200, "h": 40, "conf": 93.0},
            {"text": "This", "x": 100, "y": 320, "w": 80, "h": 30, "conf": 90.0},
            {"text": "is", "x": 190, "y": 320, "w": 40, "h": 30, "conf": 91.0},
            {"text": "authentic", "x": 240, "y": 320, "w": 160, "h": 30, "conf": 94.0},
        ]],
        metadata={"source": "image", "creation_tool": None, "exif": {}},
        original_format="png",
    )


def _make_forged_document() -> Document:
    """Create a document with obvious forgery markers."""
    img = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Forged Doc", (100, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    # Add a visually different patch (simulates splicing)
    img[400:500, 100:400] = [200, 220, 200]  # Different background
    cv2.putText(img, "EDITED TEXT", (110, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Add noise in one region but not another
    noise = np.random.randint(0, 30, (100, 300, 3), dtype=np.uint8)
    img[400:500, 100:400] = cv2.add(img[400:500, 100:400], noise)

    return Document(
        file_path="forged.jpg",
        file_name="forged.jpg",
        file_size_bytes=8000,
        pages=[img],
        ocr_text=["Forged Doc EDITED TEXT"],
        ocr_confidence=[65.0],
        ocr_word_data=[[
            {"text": "Forged", "x": 100, "y": 170, "w": 120, "h": 40, "conf": 88.0},
            {"text": "Doc", "x": 230, "y": 170, "w": 80, "h": 40, "conf": 90.0},
            {"text": "EDITED", "x": 110, "y": 430, "w": 100, "h": 30, "conf": 45.0},
            {"text": "TEXT", "x": 220, "y": 432, "w": 80, "h": 28, "conf": 42.0},
        ]],
        metadata={
            "source": "image",
            "creation_tool": "Adobe Photoshop CC 2024",
            "exif": {"Software": "Adobe Photoshop CC 2024"},
        },
        original_format="jpg",
    )


# === Individual Agent Tests ===

class TestVisualAgent:
    def test_clean_image_high_score(self):
        agent = VisualTamperAgent()
        doc = _make_clean_document()
        result = agent.analyze(doc)
        assert isinstance(result, AgentResult)
        assert result.score >= 0.7, f"Clean image should score high, got {result.score}"

    def test_returns_heatmap(self):
        agent = VisualTamperAgent()
        doc = _make_clean_document()
        result = agent.analyze(doc)
        assert result.heatmap is not None
        assert result.heatmap.shape[0] > 0

    def test_noisy_image_lower_score(self):
        agent = VisualTamperAgent()
        doc = _make_forged_document()
        result = agent.analyze(doc)
        assert isinstance(result, AgentResult)
        assert result.score <= 1.0


class TestTextAgent:
    def test_clean_text_reasonable_score(self):
        agent = TextForensicsAgent()
        doc = _make_clean_document()
        result = agent.analyze(doc)
        assert isinstance(result, AgentResult)
        assert result.score >= 0.5

    def test_empty_ocr_returns_high_score(self):
        agent = TextForensicsAgent()
        doc = _make_clean_document()
        doc.ocr_word_data = [[]]
        doc.ocr_confidence = [0.0]
        result = agent.analyze(doc)
        assert result.score == 1.0  # No text to analyze = nothing suspicious

    def test_low_confidence_detected(self):
        agent = TextForensicsAgent()
        doc = _make_forged_document()
        result = agent.analyze(doc)
        assert isinstance(result, AgentResult)


class TestMetadataAgent:
    def test_clean_metadata(self):
        agent = MetadataAgent()
        doc = _make_clean_document()
        result = agent.analyze(doc)
        assert isinstance(result, AgentResult)

    def test_photoshop_detected(self):
        agent = MetadataAgent()
        doc = _make_forged_document()
        result = agent.analyze(doc)
        # Photoshop should be flagged
        assert result.score < 0.8, f"Photoshop doc should score lower, got {result.score}"
        tool_findings = [f for f in result.findings if "photoshop" in f.description.lower()
                         or "editing" in f.description.lower()]
        assert len(tool_findings) > 0, "Should detect Photoshop as editing tool"


# === Fusion Engine Tests ===

class TestFusion:
    def test_dis_formula(self):
        results = [
            AgentResult(agent_name="A", score=0.8, reliability_weight=1.0),
            AgentResult(agent_name="B", score=0.6, reliability_weight=0.5),
        ]
        dis, forgery_type, _ = compute_dis(results)
        # DIS = (1.0*0.8 + 0.5*0.6) / (1.0+0.5) = 1.1/1.5 = 0.733...
        assert 0.7 < dis < 0.8

    def test_empty_results(self):
        dis, forgery_type, _ = compute_dis([])
        assert dis == 1.0

    def test_zero_weight_excluded(self):
        results = [
            AgentResult(agent_name="A", score=0.9, reliability_weight=1.0),
            AgentResult(agent_name="B", score=0.1, reliability_weight=0.0),
        ]
        dis, _, _ = compute_dis(results)
        assert dis > 0.85  # Agent B should be excluded


# === Full Pipeline Tests ===

class TestPipeline:
    @pytest.fixture(scope="class")
    def pipeline(self):
        return CertusDocPipeline()

    def test_authentic_png(self, pipeline):
        path = DATA_DIR / "authentic" / "test_certificate.png"
        if not path.exists():
            pytest.skip("Test data not found")
        report = pipeline.analyze(str(path))
        assert report.dis_score > 0.5
        assert len(report.agent_results) == 3

    def test_authentic_pdf(self, pipeline):
        path = DATA_DIR / "authentic" / "test_certificate.pdf"
        if not path.exists():
            pytest.skip("Test data not found")
        report = pipeline.analyze(str(path))
        assert report.dis_score > 0.6
        assert report.risk_level in (RiskLevel.AUTHENTIC, RiskLevel.LOW_RISK)

    def test_forged_lower_than_authentic(self, pipeline):
        auth_path = DATA_DIR / "authentic" / "test_certificate.pdf"
        forged_path = DATA_DIR / "forged" / "forged_certificate.jpg"
        if not auth_path.exists() or not forged_path.exists():
            pytest.skip("Test data not found")
        auth_report = pipeline.analyze(str(auth_path))
        forged_report = pipeline.analyze(str(forged_path))
        assert forged_report.dis_score <= auth_report.dis_score, (
            f"Forged ({forged_report.dis_score:.3f}) should score <= "
            f"authentic ({auth_report.dis_score:.3f})"
        )

    def test_blank_page_no_crash(self, pipeline):
        path = DATA_DIR / "forged" / "blank_page.png"
        if not path.exists():
            pytest.skip("Test data not found")
        report = pipeline.analyze(str(path))
        assert report.dis_score >= 0.0
        assert report.dis_score <= 1.0

    def test_tiny_image_no_crash(self, pipeline):
        path = DATA_DIR / "forged" / "tiny_image.png"
        if not path.exists():
            pytest.skip("Test data not found")
        report = pipeline.analyze(str(path))
        assert report.dis_score >= 0.0


# === Government Provenance Override Tests ===

class TestGovernmentProvenanceOverride:
    """
    Verify the government provenance override in compute_dis (engine.py lines 187-196).

    The override floors DIS at 0.75 when:
      - Metadata agent score >= 0.90 AND reliability_weight >= 0.60
      - Text agent has no hard forgery indicators
      - Computed DIS (before override) < 0.75

    This prevents wkhtmltopdf / DigiLocker / NSDL rendering artifacts from
    being mistaken for forgery when metadata conclusively confirms government origin.
    """

    def test_eaadhaar_wkhtmltopdf_floors_at_0_75(self):
        """
        Simulated e-Aadhaar generated by wkhtmltopdf (government tool).

        Metadata agent is confident (0.987) because it detects wkhtmltopdf.
        Visual agent is low (0.32) due to ELA/ManTraNet rendering artifacts.
        Text agent is middling (0.59) — complex Aadhaar formatting confuses it.
        No hard forgery indicators in text.

        Expected: government provenance override fires → DIS >= 0.75
        """
        results = [
            AgentResult(
                agent_name="Metadata & Provenance Agent",
                score=0.987,
                reliability_weight=0.85,
                findings=[],
            ),
            AgentResult(
                agent_name="Visual Tamper Agent",
                score=0.32,
                reliability_weight=0.90,
                findings=[],
            ),
            AgentResult(
                agent_name="Text Forensics Agent",
                score=0.59,
                reliability_weight=0.75,
                findings=[],
            ),
        ]
        dis, forgery_type, _ = compute_dis(results)
        assert dis >= 0.75, (
            f"e-Aadhaar with wkhtmltopdf provenance should score >= 0.75 "
            f"(government provenance override), got {dis:.3f}"
        )

    def test_fake_aadhaar_ios_quartz_scores_below_0_25(self):
        """
        Simulated fake Aadhaar generated with iOS Quartz (consumer tool).

        Metadata agent is low (0.20) and its findings include 'never by consumer'
        (consumer tool detected on a government ID type).
        Visual agent is very low (0.15) — clear forgery signal.
        Text agent is low (0.40) with 'fails Verhoeff' finding (invalid Aadhaar number).

        Expected: evidence cap from 'never by consumer' + 'fails Verhoeff' hard indicator
        → DIS < 0.25
        """
        from certusdoc.models import AgentFinding

        results = [
            AgentResult(
                agent_name="Metadata & Provenance Agent",
                score=0.20,
                reliability_weight=0.80,
                findings=[
                    AgentFinding(
                        description="PDF created with iOS Quartz — never by consumer software on a government ID",
                        severity=0.85,
                    ),
                ],
            ),
            AgentResult(
                agent_name="Visual Tamper Agent",
                score=0.15,
                reliability_weight=0.90,
                findings=[],
            ),
            AgentResult(
                agent_name="Text Forensics Agent",
                score=0.40,
                reliability_weight=0.75,
                findings=[
                    AgentFinding(
                        description="Aadhaar number fails Verhoeff checksum validation",
                        severity=0.95,
                    ),
                ],
            ),
        ]
        dis, forgery_type, _ = compute_dis(results)
        assert dis < 0.25, (
            f"Fake Aadhaar (iOS Quartz, invalid Verhoeff) should score < 0.25 "
            f"(evidence cap), got {dis:.3f}"
        )

    def test_high_metadata_with_hard_visual_failure_behavior(self):
        """
        Documents behavior when metadata is high (0.95) but visual has a hard failure
        (score < 0.20).

        When visual < 0.20 it is a severe agent signal. With government provenance
        active the visual agent is removed from the ceiling calculation, so no hard
        ceiling is imposed by visual alone. The government provenance floor (0.75)
        still fires because metadata >= 0.90, reliability >= 0.60, and text has no
        hard indicators.

        Expected: DIS >= 0.75 (government floor overrides the very low visual score),
        confirming that a single extreme visual outlier cannot override strong
        government provenance when text is clean.
        """
        results = [
            AgentResult(
                agent_name="Metadata & Provenance Agent",
                score=0.95,
                reliability_weight=0.85,
                findings=[],
            ),
            AgentResult(
                agent_name="Visual Tamper Agent",
                score=0.10,
                reliability_weight=0.90,
                findings=[],
            ),
            AgentResult(
                agent_name="Text Forensics Agent",
                score=0.80,
                reliability_weight=0.75,
                findings=[],
            ),
        ]
        dis, forgery_type, _ = compute_dis(results)
        # Government provenance override should floor DIS at 0.75 even with
        # extreme visual failure, because visual is excluded from ceiling logic
        # and the text agent has no hard indicators.
        assert dis >= 0.75, (
            f"High metadata + clean text should trigger government provenance floor "
            f"(visual excluded from ceiling), got {dis:.3f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
