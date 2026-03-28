"""
Forensic PDF Report Generator

Generates a professional PDF report using reportlab with:
- Document summary (filename, size, pages, processing time)
- DIS score with color coding (red/yellow/green)
- Per-agent breakdown with findings
- Heatmap overlay image if available
- Recommended action
"""
import html
import io
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak,
)

from certusdoc.models import ForensicReport, RiskLevel


def generate_report(report: ForensicReport, output_path: Optional[str] = None) -> bytes:
    """
    Generate a forensic PDF report from a ForensicReport.

    Args:
        report: The completed ForensicReport from the pipeline.
        output_path: Optional file path to save the PDF. If None, returns bytes.

    Returns:
        PDF content as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=25 * mm, bottomMargin=20 * mm,
        leftMargin=20 * mm, rightMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "Title2", parent=styles["Title"], fontSize=20, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "FindingText", parent=styles["Normal"], fontSize=9,
        leftIndent=12, textColor=colors.HexColor("#4a5568"),
    ))
    styles.add(ParagraphStyle(
        "SectionHead", parent=styles["Heading2"], fontSize=13,
        spaceAfter=8, spaceBefore=14,
    ))

    elements = []

    # === Header ===
    elements.append(Paragraph("CertusDoc Forensic Report", styles["Title2"]))
    elements.append(Paragraph(
        "Multi-Agent Document Forgery Detection System",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 12))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#1a1a2e")))
    elements.append(Spacer(1, 12))

    # === Document Summary ===
    elements.append(Paragraph("Document Summary", styles["SectionHead"]))

    doc_info = [
        ["Filename:", report.document.file_name],
        ["File Size:", f"{report.document.file_size_bytes / 1024:.1f} KB"],
        ["Pages:", str(len(report.document.pages))],
        ["Format:", report.document.original_format.upper()],
        ["Processing Time:", f"{report.processing_time_ms:.0f} ms"],
    ]
    if report.document.ocr_confidence:
        avg_conf = sum(report.document.ocr_confidence) / len(report.document.ocr_confidence)
        doc_info.append(["OCR Confidence:", f"{avg_conf:.1f}%"])

    summary_table = Table(doc_info, colWidths=[120, 350])
    summary_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 16))

    # === DIS Score ===
    elements.append(Paragraph("Document Integrity Score", styles["SectionHead"]))

    dis_color = _get_risk_color(report.risk_level)
    risk_text = html.escape(report.risk_level.value)
    forgery_text = html.escape(report.primary_forgery_type.value.replace("_", " ").title())

    score_data = [
        [
            Paragraph(f'<font size="28" color="{dis_color}"><b>{report.dis_score:.2f}</b></font>',
                       styles["Normal"]),
            Paragraph(
                f'<font size="14" color="{dis_color}"><b>{risk_text}</b></font><br/>'
                f'<font size="9">Forgery type: {forgery_text}</font>',
                styles["Normal"],
            ),
        ]
    ]
    score_table = Table(score_data, colWidths=[100, 370])
    score_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(_get_risk_bg(report.risk_level))),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(dis_color)),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    elements.append(score_table)
    elements.append(Spacer(1, 16))

    # === Per-Agent Breakdown ===
    elements.append(Paragraph("Agent Analysis Breakdown", styles["SectionHead"]))

    for result in sorted(report.agent_results, key=lambda r: r.score):
        agent_color = "#e53e3e" if result.score < 0.4 else (
            "#ed8936" if result.score < 0.7 else "#48bb78"
        )

        safe_name = html.escape(result.agent_name)
        agent_header = [
            [
                Paragraph(f'<b>{safe_name}</b>', styles["Normal"]),
                Paragraph(
                    f'<font color="{agent_color}"><b>Score: {result.score:.2f}</b></font> '
                    f'| Reliability: {result.reliability_weight:.2f} '
                    f'| Time: {result.processing_time_ms:.0f}ms',
                    styles["Normal"],
                ),
            ]
        ]
        agent_table = Table(agent_header, colWidths=[180, 290])
        agent_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(agent_table)

        for finding in result.findings[:5]:
            severity_marker = "!!!" if finding.severity > 0.7 else (
                "!!" if finding.severity > 0.4 else "!"
            )
            safe_desc = html.escape(finding.description)
            elements.append(Paragraph(
                f"{severity_marker} {safe_desc}",
                styles["FindingText"],
            ))

        elements.append(Spacer(1, 8))

    # === Heatmap ===
    if report.fused_heatmap is not None:
        elements.append(Paragraph("Anomaly Heatmap", styles["SectionHead"]))
        try:
            heatmap_img = _array_to_rl_image(report.fused_heatmap, max_width=450)
            if heatmap_img:
                elements.append(heatmap_img)
                elements.append(Spacer(1, 8))

            # Overlay on original if available
            if report.document.pages:
                original = report.document.pages[0]
                original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                heatmap_colored = cv2.applyColorMap(report.fused_heatmap, cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                heatmap_resized = cv2.resize(
                    heatmap_rgb,
                    (original_rgb.shape[1], original_rgb.shape[0]),
                )
                overlay = cv2.addWeighted(original_rgb, 0.6, heatmap_resized, 0.4, 0)
                overlay_img = _array_to_rl_image(overlay, max_width=450)
                if overlay_img:
                    elements.append(Paragraph(
                        "<i>Heatmap overlaid on original document</i>",
                        styles["FindingText"],
                    ))
                    elements.append(overlay_img)
        except Exception as e:
            logger.warning(f"Heatmap rendering failed: {e}")

    elements.append(Spacer(1, 16))

    # === Recommended Action ===
    elements.append(Paragraph("Recommended Action", styles["SectionHead"]))
    safe_action = html.escape(report.recommended_action)
    elements.append(Paragraph(safe_action, styles["Normal"]))
    elements.append(Spacer(1, 20))

    # === Footer ===
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#cbd5e0")))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        '<font size="8" color="#a0aec0">'
        'Generated by CertusDoc — Multi-Agent Document Forgery Detection System'
        '</font>',
        styles["Normal"],
    ))

    # Build PDF
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    if output_path:
        Path(output_path).write_bytes(pdf_bytes)
        logger.info(f"Report saved to {output_path}")

    return pdf_bytes


def _get_risk_color(risk: RiskLevel) -> str:
    """Return hex color for risk level."""
    return {
        RiskLevel.AUTHENTIC: "#38a169",
        RiskLevel.LOW_RISK: "#48bb78",
        RiskLevel.MEDIUM_RISK: "#ed8936",
        RiskLevel.HIGH_RISK: "#e53e3e",
    }.get(risk, "#4a5568")


def _get_risk_bg(risk: RiskLevel) -> str:
    """Return background hex color for risk level."""
    return {
        RiskLevel.AUTHENTIC: "#e8f5e9",
        RiskLevel.LOW_RISK: "#e8f5e9",
        RiskLevel.MEDIUM_RISK: "#fff3e0",
        RiskLevel.HIGH_RISK: "#ffe0e0",
    }.get(risk, "#f7fafc")


def _array_to_rl_image(
    arr: np.ndarray, max_width: float = 450
) -> Optional[RLImage]:
    """Convert a numpy array to a reportlab Image object."""
    from PIL import Image as PILImage

    if arr is None:
        return None

    if len(arr.shape) == 2:
        pil = PILImage.fromarray(arr, mode="L")
    else:
        pil = PILImage.fromarray(arr)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)

    img_w, img_h = pil.size
    scale = min(1.0, max_width / img_w)
    return RLImage(buf, width=img_w * scale, height=img_h * scale)
