"""
CertusDoc Streamlit Dashboard
Matches the mockup from CertusDoc_final.pdf page 4
"""
import streamlit as st
import tempfile
import os
import time
from pathlib import Path

import numpy as np
import cv2

# Page config
st.set_page_config(
    page_title="CertusDoc",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS to match the proposal mockup
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .score-card {
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
    }
    .high-risk { background-color: #ffe0e0; border-left: 4px solid #e53e3e; }
    .medium-risk { background-color: #fff3e0; border-left: 4px solid #ed8936; }
    .low-risk { background-color: #e8f5e9; border-left: 4px solid #48bb78; }
    .authentic { background-color: #e8f5e9; border-left: 4px solid #38a169; }
    .agent-card {
        padding: 1rem;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    .finding-text { font-size: 0.85rem; color: #4a5568; }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<div class="main-header">🔍 CertusDoc</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Document forgery detection & forensic analysis</div>',
        unsafe_allow_html=True,
    )

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a document for analysis",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
        help="Supported formats: PDF, PNG, JPG, TIFF, BMP",
    )

    if uploaded_file is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded_file.name).suffix
        ) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            # Show file info
            file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
            st.markdown(f"**📄 {uploaded_file.name}** — {file_size_mb:.1f} MB")

            # Run analysis
            with st.spinner("Analyzing document..."):
                start = time.time()
                report = run_analysis(tmp_path)
                elapsed = time.time() - start

            st.markdown(f"*Analysis complete in {elapsed:.1f} seconds*")
            st.divider()

            # === Results Section ===
            render_results(report)

        finally:
            os.unlink(tmp_path)


def run_analysis(file_path: str):
    """Run the CertusDoc pipeline."""
    from certusdoc.pipeline import CertusDocPipeline

    pipeline = CertusDocPipeline()
    return pipeline.analyze(file_path)


def render_results(report):
    """Render the forensic report in the dashboard."""

    # Top-level metrics row
    col1, col2, col3 = st.columns(3)

    # DIS Score — use risk-colored card
    risk_class = report.risk_level.value.lower().replace(" ", "-")
    dis_html = (
        f'<div class="score-card {risk_class}">'
        f'<div style="font-size: 0.8rem; color: #666;">Document integrity score</div>'
        f'<div style="font-size: 2.5rem; font-weight: 700;">{report.dis_score:.2f}</div>'
        f'<div style="font-size: 0.9rem; font-weight: 600;">{report.risk_level.value}</div>'
        f'</div>'
    )
    with col1:
        st.markdown(dis_html, unsafe_allow_html=True)

    # Forgery type
    forgery_display = report.primary_forgery_type.value.replace("_", " ").title()
    forgery_html = (
        f'<div class="score-card" style="background-color: #f7fafc; border-left: 4px solid #4299e1;">'
        f'<div style="font-size: 0.8rem; color: #666;">Forgery type detected</div>'
        f'<div style="font-size: 1.3rem; font-weight: 700;">{forgery_display}</div>'
        f'</div>'
    )
    with col2:
        st.markdown(forgery_html, unsafe_allow_html=True)

    # Agents completed
    completed = len([r for r in report.agent_results if r.reliability_weight > 0])
    total = len(report.agent_results)
    converged = len([r for r in report.agent_results if r.score < 0.6])
    agents_html = (
        f'<div class="score-card" style="background-color: #f7fafc; border-left: 4px solid #9f7aea;">'
        f'<div style="font-size: 0.8rem; color: #666;">Agents completed</div>'
        f'<div style="font-size: 2.5rem; font-weight: 700;">{completed} / {total}</div>'
        f'<div style="font-size: 0.9rem;">{converged} agent(s) flagged issues</div>'
        f'</div>'
    )
    with col3:
        st.markdown(agents_html, unsafe_allow_html=True)

    st.divider()

    # Per-agent breakdown
    st.subheader("Per-agent breakdown")

    for result in sorted(report.agent_results, key=lambda r: r.score):
        score_color = "#e53e3e" if result.score < 0.4 else (
            "#ed8936" if result.score < 0.7 else "#48bb78"
        )

        with st.container():
            agent_html = (
                f'<div class="agent-card">'
                f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                f'<div style="font-weight: 600;">{result.agent_name}</div>'
                f'<div style="background-color: {score_color}; color: white; '
                f'padding: 2px 10px; border-radius: 12px; font-size: 0.85rem;">'
                f'Score: {result.score:.2f}</div>'
                f'</div></div>'
            )
            st.markdown(agent_html, unsafe_allow_html=True)

            # Show findings
            if result.findings:
                for finding in result.findings[:3]:  # Show top 3
                    st.markdown(
                        f'<div class="finding-text">• {finding.description}</div>',
                        unsafe_allow_html=True,
                    )

            # Reliability info
            st.caption(
                f"Reliability weight: {result.reliability_weight:.2f} | "
                f"Processing time: {result.processing_time_ms:.0f}ms"
            )
            st.markdown("---")

    # Recommended Action
    if report.risk_level.value in ("HIGH RISK", "MEDIUM RISK"):
        st.error(f"**Recommended action:** {report.recommended_action}")
    elif report.risk_level.value == "LOW RISK":
        st.warning(f"**Recommended action:** {report.recommended_action}")
    else:
        st.success(f"**Result:** {report.recommended_action}")

    # Heatmap visualization
    if report.fused_heatmap is not None:
        st.subheader("Anomaly Heatmap")
        heatmap_colored = cv2.applyColorMap(report.fused_heatmap, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay on original document
        if report.document.pages:
            original = report.document.pages[0]
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            heatmap_resized = cv2.resize(
                heatmap_rgb,
                (original_rgb.shape[1], original_rgb.shape[0]),
            )
            overlay = cv2.addWeighted(original_rgb, 0.6, heatmap_resized, 0.4, 0)

            col1, col2 = st.columns(2)
            with col1:
                st.image(original_rgb, caption="Original Document", use_container_width=True)
            with col2:
                st.image(overlay, caption="Anomaly Overlay", use_container_width=True)
        else:
            st.image(heatmap_rgb, caption="Anomaly Heatmap", use_container_width=True)

    # Export options
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📄 Export PDF Report"):
            from certusdoc.report.generator import generate_report
            pdf_bytes = generate_report(report)
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"certusdoc_{report.document.file_name}.pdf",
                mime="application/pdf",
            )
    with col2:
        if st.button("🔍 View Heatmap"):
            if report.fused_heatmap is not None:
                st.image(
                    cv2.applyColorMap(report.fused_heatmap, cv2.COLORMAP_JET),
                    caption="Full Anomaly Heatmap",
                )
    with col3:
        if st.button("{ } Export JSON"):
            import json
            export = {
                "file_name": report.document.file_name,
                "dis_score": report.dis_score,
                "risk_level": report.risk_level.value,
                "forgery_type": report.primary_forgery_type.value,
                "recommended_action": report.recommended_action,
                "agents": [
                    {
                        "name": r.agent_name,
                        "score": r.score,
                        "reliability": r.reliability_weight,
                        "findings": [f.description for f in r.findings],
                    }
                    for r in report.agent_results
                ],
                "processing_time_ms": report.processing_time_ms,
            }
            st.json(export)


if __name__ == "__main__":
    main()
