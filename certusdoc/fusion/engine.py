"""
Weighted Trust Fusion Engine (Stage 3)

Implements: DIS = Σ(Rᵢ × Sᵢ) / ΣRᵢ

Where:
- Rᵢ = reliability weight of agent i (dynamic, based on input quality)
- Sᵢ = integrity score from agent i (0 = forged, 1 = authentic)
- DIS = Document Integrity Score (final output)
"""
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from certusdoc.models import AgentResult, ForgeryType, Document


def compute_dis(
    agent_results: list[AgentResult],
    document: "Document | None" = None,
) -> tuple[float, ForgeryType, Optional[np.ndarray]]:
    """
    Compute the Document Integrity Score by weighted fusion of agent results.
    Applies dynamic reliability adjustments based on document characteristics.

    Args:
        agent_results: List of AgentResult from all detection agents.
        document: Optional Document for dynamic weight adjustment.

    Returns:
        Tuple of (dis_score, primary_forgery_type, fused_heatmap)
    """
    if not agent_results:
        logger.warning("No agent results to fuse")
        return 1.0, ForgeryType.NONE, None

    # Apply dynamic reliability adjustments
    adjusted_results = _apply_dynamic_weights(agent_results, document)

    # Compute weighted DIS
    weighted_sum = 0.0
    weight_sum = 0.0

    for result, adj_weight in adjusted_results:
        s = result.score
        weighted_sum += adj_weight * s
        weight_sum += adj_weight

        logger.info(
            f"  {result.agent_name}: score={s:.3f}, "
            f"reliability={result.reliability_weight:.2f}, "
            f"adjusted={adj_weight:.2f}, weighted={adj_weight*s:.3f}"
        )

    dis = weighted_sum / weight_sum if weight_sum > 0 else 1.0
    dis = np.clip(dis, 0.0, 1.0)

    # === WhatsApp image handling ===
    # If WhatsApp-compressed, the metadata agent weight is already 0.1,
    # so it won't appear in active_agents (w > 0.1 filter below).
    # For WhatsApp images, visual agent is the primary signal — if it
    # flags issues, tighten the DIS more aggressively since WhatsApp
    # compression already introduces artifacts (the agent accounts for this).
    is_whatsapp = _is_whatsapp_image(document)

    # === HARD CEILING RULES ===
    # Graduated ceilings based on how strongly agents flag issues.
    # Stronger visual/text signals → tighter DIS cap.
    active_agents = [r for r, w in adjusted_results if w > 0.1]
    flagging_agents = [r for r in active_agents if r.score < 0.6]
    moderate_agents = [r for r in active_agents if 0.3 <= r.score < 0.50]
    severe_agents = [r for r in active_agents if r.score < 0.3]

    # === Cross-agent trust ===
    # When metadata confirms legitimate origin AND text is clean,
    # visual anomalies alone are less conclusive (e.g., wkhtmltopdf
    # rendering artifacts trigger ELA but aren't forgery).
    meta_result = next((r for r in active_agents
                        if "metadata" in r.agent_name.lower()), None)
    text_result = next((r for r in active_agents
                        if "text" in r.agent_name.lower()), None)
    visual_result = next((r for r in active_agents
                          if "visual" in r.agent_name.lower()), None)

    # Cross-agent trust requires POSITIVE legitimacy evidence from metadata,
    # not just neutral absence-of-evidence (image-only inputs score ~0.82
    # with no actual metadata). Check reliability > 0.6 (needs multiple
    # metadata fields present) AND score >= 0.80.
    metadata_confirms_legit = (meta_result and meta_result.score >= 0.80
                               and meta_result.reliability_weight >= 0.60)
    text_is_clean = text_result and text_result.score >= 0.75
    visual_only_flag = (visual_result and visual_result.score < 0.6
                        and metadata_confirms_legit and text_is_clean)

    # If ONLY the visual agent flags and metadata+text agree it's legit,
    # boost the visual score for ceiling calculation (soften its impact).
    effective_flagging = list(flagging_agents)
    effective_moderate = list(moderate_agents)
    effective_severe = list(severe_agents)

    if visual_only_flag:
        # Remove visual from the flagging/severity lists for ceiling calc
        effective_flagging = [r for r in flagging_agents
                              if "visual" not in r.agent_name.lower()]
        effective_moderate = [r for r in moderate_agents
                              if "visual" not in r.agent_name.lower()]
        effective_severe = [r for r in severe_agents
                            if "visual" not in r.agent_name.lower()]
        logger.info(f"  Cross-agent trust: metadata ({meta_result.score:.2f}) + "
                     f"text ({text_result.score:.2f}) confirm legitimacy — "
                     f"visual anomaly ({visual_result.score:.2f}) softened")

    ceiling = None
    if len(effective_severe) >= 2:
        ceiling = 0.30
    elif len(effective_severe) >= 1 and len(effective_flagging) >= 2:
        ceiling = 0.35
    elif len(effective_flagging) >= 2:
        # Two agents flagging — strong signal
        ceiling = 0.45
    elif len(effective_severe) >= 1:
        # One agent with very strong signal
        ceiling = 0.45
    elif len(effective_moderate) >= 1 and len(effective_flagging) >= 2:
        ceiling = 0.50
    elif len(effective_moderate) >= 1:
        # One agent with moderate signal (score 0.30-0.50)
        ceiling = 0.55

    if ceiling is not None and dis > ceiling:
        worst = min(active_agents, key=lambda r: r.score)
        logger.info(f"  DIS ceiling {ceiling} applied: {len(flagging_agents)} agent(s) "
                     f"scored <0.6 ({worst.agent_name}: {worst.score:.3f})")
        dis = ceiling

    # WhatsApp: if visual agent flags strong issues, apply extra penalty
    # Only for clear forgery signals (score < 0.40), not mild JPEG artifacts
    if is_whatsapp and visual_result and visual_result.score < 0.40:
        wa_ceiling = min(dis, visual_result.score + 0.10)
        if wa_ceiling < dis:
            logger.info(f"  WhatsApp visual penalty: DIS {dis:.3f} → {wa_ceiling:.3f} "
                         f"(visual={visual_result.score:.3f})")
            dis = wa_ceiling

    # Agent convergence bonus/penalty (applied after ceiling)
    low_score_agents = [r for r, _ in adjusted_results if r.score < 0.5]
    high_score_agents = [r for r, _ in adjusted_results if r.score > 0.8]

    if len(low_score_agents) >= 2 and len(active_agents) >= 2:
        convergence_penalty = 0.05 * len(low_score_agents)
        dis = max(0.0, dis - convergence_penalty)
        logger.info(f"  Convergence penalty: -{convergence_penalty:.2f} "
                     f"({len(low_score_agents)} agents flagged issues)")
    elif len(high_score_agents) == len(active_agents) and len(active_agents) >= 2:
        convergence_bonus = 0.03
        dis = min(1.0, dis + convergence_bonus)

    dis = float(np.clip(dis, 0.0, 1.0))

    # Determine primary forgery type
    forgery_type = _determine_forgery_type(agent_results, dis)

    # Fuse heatmaps (if available)
    fused_heatmap = _fuse_heatmaps(agent_results)

    logger.info(f"DIS = {dis:.4f} | Forgery type: {forgery_type.value}")

    return float(dis), forgery_type, fused_heatmap


def _is_whatsapp_image(document: "Document | None") -> bool:
    """
    Detect if an image was likely shared via WhatsApp or similar messaging apps.

    WhatsApp strips all EXIF data, re-encodes as JPEG with aggressive compression,
    and resizes to mobile-friendly dimensions. This is extremely common in India
    where documents are shared via WhatsApp.

    Heuristics:
    - Image (not PDF)
    - JPEG format
    - No EXIF data at all
    - File size under 500KB (WhatsApp compresses to ~50-250KB)
    - Dimensions suggest mobile capture/resize (max dim <= 1600px)
    """
    if document is None:
        return False

    meta = document.metadata
    if meta.get("source") != "image":
        return False

    # Must be JPEG
    fmt = (meta.get("format") or "").upper()
    original = document.original_format.lower()
    if fmt not in ("JPEG", "JPG") and original not in ("jpg", "jpeg"):
        return False

    # No EXIF data (WhatsApp strips it completely)
    exif = meta.get("exif", {})
    if exif:
        return False

    # No creation tool (since EXIF Software tag is gone)
    if meta.get("creation_tool"):
        return False

    # Small file size (WhatsApp aggressive compression)
    if document.file_size_bytes > 500_000:
        return False

    # Mobile-typical dimensions: WhatsApp caps at ~1600px on longest side
    w = meta.get("width", 0)
    h = meta.get("height", 0)
    if w > 0 and h > 0:
        max_dim = max(w, h)
        min_dim = min(w, h)
        # WhatsApp images are typically 640-1600px, with aspect ratios
        # consistent with phone cameras (roughly 3:4, 9:16, etc.)
        if max_dim > 1920:
            return False
        # Very small images (<200px) aren't typical WhatsApp shares
        if min_dim < 200:
            return False

    return True


def _apply_dynamic_weights(
    agent_results: list[AgentResult],
    document: "Document | None" = None,
) -> list[tuple[AgentResult, float]]:
    """
    Adjust agent reliability weights based on document characteristics.

    Rules:
    - If image DPI is low (small image), reduce visual agent weight
    - If OCR confidence is below 50%, reduce text agent weight
    - If metadata is sparse, reduce metadata agent weight
    - If WhatsApp-compressed image detected, reduce metadata weight to 0.1
    - If an agent failed (reliability=0), keep it at 0
    """
    adjusted = []
    is_whatsapp = _is_whatsapp_image(document)

    if is_whatsapp:
        logger.info("  WhatsApp/messaging image detected: no EXIF, JPEG, "
                     f"<500KB ({document.file_size_bytes/1024:.0f}KB), "
                     f"mobile dimensions — metadata weight reduced to 0.1")

    for result in agent_results:
        weight = result.reliability_weight

        if weight == 0.0:
            adjusted.append((result, 0.0))
            continue

        agent_lower = result.agent_name.lower()

        if document is not None:
            # === WhatsApp image handling ===
            # WhatsApp strips metadata and adds JPEG artifacts.
            # Metadata is useless → reduce weight to 0.1.
            # Visual and text agents should drive the score.
            if is_whatsapp:
                if "metadata" in agent_lower:
                    weight = 0.1
                elif "visual" in agent_lower:
                    # Visual is primary signal for WhatsApp images;
                    # ensure it's not penalized for resolution
                    weight = max(weight, 0.7)

            # Visual agent: penalize for low-resolution input
            if "visual" in agent_lower and document.pages:
                page = document.pages[0]
                h, w = page.shape[:2]
                pixel_count = h * w
                # Below ~1MP (e.g. 1000x1000), visual analysis is less reliable
                if pixel_count < 500_000:
                    weight *= 0.4
                    logger.debug(f"Visual agent weight reduced: low resolution ({w}x{h})")
                elif pixel_count < 1_000_000:
                    weight *= 0.7
                    logger.debug(f"Visual agent weight reduced: medium resolution ({w}x{h})")

            # Text agent: penalize for low OCR confidence
            if "text" in agent_lower and document.ocr_confidence:
                avg_conf = float(np.mean(document.ocr_confidence))
                if avg_conf < 30:
                    weight *= 0.2
                    logger.debug(f"Text agent weight reduced: very low OCR ({avg_conf:.0f}%)")
                elif avg_conf < 50:
                    weight *= 0.5
                    logger.debug(f"Text agent weight reduced: low OCR ({avg_conf:.0f}%)")

            # Metadata agent: penalize for sparse metadata
            # (skip if already handled by WhatsApp detection)
            if "metadata" in agent_lower and not is_whatsapp:
                meta = document.metadata
                available = sum(1 for k in ["creation_tool", "creation_date",
                                             "modification_date", "exif",
                                             "embedded_fonts", "producer"]
                                if meta.get(k))
                if available <= 1:
                    weight *= 0.5
                    logger.debug(f"Metadata agent weight reduced: sparse metadata ({available} fields)")

        adjusted.append((result, float(np.clip(weight, 0.0, 1.0))))

    return adjusted


def _determine_forgery_type(
    agent_results: list[AgentResult], dis: float
) -> ForgeryType:
    """
    Determine the most likely forgery type based on which agents
    flagged issues and what findings they reported.
    """
    if dis >= 0.80:
        return ForgeryType.NONE

    # Find the agent with the lowest score (strongest detection signal)
    worst_agent = min(agent_results, key=lambda r: r.score)

    # Map agent findings to forgery types
    agent_name = worst_agent.agent_name.lower()
    findings_text = " ".join(f.description.lower() for f in worst_agent.findings)

    if "visual" in agent_name:
        if "ela" in findings_text or "compression" in findings_text:
            return ForgeryType.JPEG_RECOMPRESSION
        if "copy" in findings_text:
            return ForgeryType.COPY_MOVE
        if "splicing" in findings_text or "noise" in findings_text:
            return ForgeryType.IMAGE_MANIPULATION
        return ForgeryType.IMAGE_MANIPULATION

    if "text" in agent_name:
        if "font" in findings_text:
            return ForgeryType.FONT_MISMATCH
        if "confidence" in findings_text or "spacing" in findings_text:
            return ForgeryType.TEXT_SPLICING
        return ForgeryType.TEXT_SPLICING

    if "metadata" in agent_name:
        return ForgeryType.METADATA_TAMPERING

    return ForgeryType.UNKNOWN


def _fuse_heatmaps(agent_results: list[AgentResult]) -> Optional[np.ndarray]:
    """
    Fuse heatmaps from all agents that produced them.
    Uses reliability-weighted averaging.
    """
    heatmaps_with_weights = []

    for result in agent_results:
        if result.heatmap is not None:
            heatmaps_with_weights.append((result.heatmap, result.reliability_weight))

    if not heatmaps_with_weights:
        return None

    if len(heatmaps_with_weights) == 1:
        return heatmaps_with_weights[0][0]

    # Resize all heatmaps to match the largest one
    max_h = max(h.shape[0] for h, _ in heatmaps_with_weights)
    max_w = max(h.shape[1] for h, _ in heatmaps_with_weights)

    fused = np.zeros((max_h, max_w), dtype=np.float64)
    total_weight = 0.0

    for heatmap, weight in heatmaps_with_weights:
        resized = cv2.resize(heatmap, (max_w, max_h), interpolation=cv2.INTER_LINEAR)
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        fused += resized.astype(np.float64) * weight
        total_weight += weight

    if total_weight > 0:
        fused /= total_weight

    return fused.astype(np.uint8)
