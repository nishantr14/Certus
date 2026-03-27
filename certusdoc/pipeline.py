"""
CertusDoc Pipeline Orchestrator

Runs all four stages:
1. Document Ingestion
2. Multi-Agent Detection (parallel)
3. Weighted Trust Fusion
4. Output (ForensicReport)
"""
import time
from pathlib import Path
from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from certusdoc.models import Document, ForensicReport, AgentResult, RiskLevel, ForgeryType
from certusdoc.ingestion.ingest import ingest_document
from certusdoc.agents.visual_agent import VisualTamperAgent
from certusdoc.agents.text_agent import TextForensicsAgent
from certusdoc.agents.metadata_agent import MetadataAgent
from certusdoc.fusion.engine import compute_dis


class CertusDocPipeline:
    """Main orchestrator for the CertusDoc forgery detection system."""

    def __init__(self, trufor_model_path: str = None):
        """
        Initialize the pipeline with all detection agents.
        
        Args:
            trufor_model_path: Optional path to TruFor model weights.
        """
        logger.info("Initializing CertusDoc pipeline...")

        self.visual_agent = VisualTamperAgent(trufor_model_path=trufor_model_path)
        self.text_agent = TextForensicsAgent()
        self.metadata_agent = MetadataAgent()

        self.agents = [self.visual_agent, self.text_agent, self.metadata_agent]

        logger.info(f"Pipeline ready with {len(self.agents)} agents")

    def analyze(self, file_path: Union[str, Path]) -> ForensicReport:
        """
        Run the full CertusDoc pipeline on a document.
        
        Args:
            file_path: Path to the document file (PDF or image).
            
        Returns:
            ForensicReport with full analysis results.
        """
        total_start = time.time()

        # Stage 1: Ingestion
        logger.info("=" * 60)
        logger.info("STAGE 1: Document Ingestion")
        logger.info("=" * 60)
        document = ingest_document(file_path)
        logger.info(
            f"Ingested: {document.file_name} | "
            f"{len(document.pages)} page(s) | "
            f"{document.file_size_bytes / 1024:.1f} KB | "
            f"OCR confidence: {sum(document.ocr_confidence) / max(1, len(document.ocr_confidence)):.1f}%"
        )

        # Stage 2: Multi-Agent Detection (parallel)
        logger.info("=" * 60)
        logger.info("STAGE 2: Multi-Agent Detection")
        logger.info("=" * 60)
        agent_results = self._run_agents_parallel(document)

        # Stage 3: Weighted Trust Fusion
        logger.info("=" * 60)
        logger.info("STAGE 3: Weighted Trust Fusion")
        logger.info("=" * 60)
        dis_score, forgery_type, fused_heatmap = compute_dis(agent_results, document)
        risk_level = ForensicReport.classify_risk(dis_score)

        # Generate recommended action
        recommended_action = self._generate_recommendation(
            dis_score, risk_level, forgery_type, agent_results
        )

        total_elapsed = (time.time() - total_start) * 1000

        # Stage 4: Build Report
        logger.info("=" * 60)
        logger.info("STAGE 4: Report Generation")
        logger.info("=" * 60)
        logger.info(f"DIS: {dis_score:.4f} | Risk: {risk_level.value} | "
                     f"Forgery: {forgery_type.value}")
        logger.info(f"Total processing time: {total_elapsed:.0f}ms")

        report = ForensicReport(
            document=document,
            agent_results=agent_results,
            dis_score=dis_score,
            risk_level=risk_level,
            primary_forgery_type=forgery_type,
            recommended_action=recommended_action,
            fused_heatmap=fused_heatmap,
            processing_time_ms=total_elapsed,
        )

        return report

    def _run_agents_parallel(self, document: Document) -> list[AgentResult]:
        """Run all detection agents in parallel using ThreadPoolExecutor."""
        results = []

        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            future_to_agent = {
                executor.submit(agent.analyze, document): agent
                for agent in self.agents
            }

            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(
                        f"  ✓ {agent.name}: score={result.score:.3f} "
                        f"({len(result.findings)} findings, "
                        f"{result.processing_time_ms:.0f}ms)"
                    )
                except Exception as e:
                    logger.error(f"  ✗ {agent.name} failed: {e}")
                    # Return a neutral result for failed agents
                    results.append(AgentResult(
                        agent_name=agent.name,
                        score=0.5,
                        reliability_weight=0.0,  # Zero weight = excluded from fusion
                        findings=[],
                        details={"error": str(e)},
                    ))

        return results

    def _generate_recommendation(
        self,
        dis: float,
        risk: RiskLevel,
        forgery_type: ForgeryType,
        results: list[AgentResult],
    ) -> str:
        """Generate a human-readable recommended action."""
        if risk == RiskLevel.AUTHENTIC:
            return "Document appears authentic. No action required."

        if risk == RiskLevel.LOW_RISK:
            return (
                "Minor inconsistencies detected. Manual review recommended "
                "if document is being used for high-stakes verification."
            )

        # Count how many agents flagged issues
        flagging_agents = [r for r in results if r.score < 0.6]
        agent_names = [r.agent_name for r in flagging_agents]
        convergence = f"{len(flagging_agents)}/{len(results)} agents converged"

        if risk == RiskLevel.HIGH_RISK:
            return (
                f"Flag immediately. Do not accept as valid document. "
                f"{convergence} on detecting {forgery_type.value}. "
                f"Flagging agents: {', '.join(agent_names)}. "
                f"Escalate to forensic examiner. Request original document "
                f"from issuing authority."
            )

        # MEDIUM_RISK
        return (
            f"Document shows signs of {forgery_type.value}. "
            f"{convergence}. "
            f"Recommend manual verification before accepting. "
            f"Cross-check with issuing authority if possible."
        )
