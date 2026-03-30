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
    # Strong metadata confirmation (>= 0.95) indicates government-issued doc
    # with known tool — this is very strong evidence of legitimacy.
    metadata_strongly_legit = (meta_result and meta_result.score >= 0.95
                                and meta_result.reliability_weight >= 0.60)
    # Government-tool provenance override (>= 0.90): a document provably
    # created by wkhtmltopdf / DigiLocker / NSDL CANNOT be forged at the
    # metadata level. Visual anomalies are rendering artifacts, not forgery.
    metadata_govt_provenance = (meta_result and meta_result.score >= 0.90
                                and meta_result.reliability_weight >= 0.60)
    text_is_clean = text_result and text_result.score >= 0.75
    # For very strong metadata, relax text threshold — e-Aadhaars etc. have
    # complex formatting that legitimately confuses the text agent
    text_is_acceptable = text_result and text_result.score >= 0.50

    # Check that text agent has no hard forgery indicators (invalid Aadhaar,
    # QR mismatch, etc.) — needed for the government provenance override.
    text_has_hard_indicators = False
    if text_result:
        for f in text_result.findings:
            desc = f.description.lower()
            if any(kw in desc for kw in (
                "fails verhoeff", "invalid aadhaar", "contradicts ocr",
                "qr mismatch", "qr code contradicts",
            )):
                text_has_hard_indicators = True
                break
    # Also check metadata agent findings for QR contradictions
    if meta_result and not text_has_hard_indicators:
        for f in meta_result.findings:
            desc = f.description.lower()
            if "contradicts ocr" in desc or "qr mismatch" in desc:
                text_has_hard_indicators = True
                break

    visual_only_flag = (visual_result and visual_result.score < 0.6
                        and metadata_confirms_legit and text_is_clean)

    # Extended cross-agent trust: very strong metadata + acceptable text
    # should still soften visual false positives (e.g., wkhtmltopdf rendering
    # artifacts on e-Aadhaar that ManTraNet flags as manipulation)
    visual_meta_override = (visual_result and visual_result.score < 0.6
                             and metadata_strongly_legit and text_is_acceptable
                             and not visual_only_flag)

    # If ONLY the visual agent flags and metadata+text agree it's legit,
    # boost the visual score for ceiling calculation (soften its impact).
    effective_flagging = list(flagging_agents)
    effective_moderate = list(moderate_agents)
    effective_severe = list(severe_agents)

    # Government provenance: visual agent is entirely excluded from ceiling logic
    govt_provenance_active = (metadata_govt_provenance and not text_has_hard_indicators)

    if govt_provenance_active or visual_only_flag or visual_meta_override:
        # Remove visual from the flagging/severity lists for ceiling calc
        effective_flagging = [r for r in flagging_agents
                              if "visual" not in r.agent_name.lower()]
        effective_moderate = [r for r in moderate_agents
                              if "visual" not in r.agent_name.lower()]
        effective_severe = [r for r in severe_agents
                            if "visual" not in r.agent_name.lower()]
        if govt_provenance_active:
            trust_reason = "government provenance (meta≥0.90, no hard text indicators)"
        elif visual_only_flag:
            trust_reason = "metadata+text"
        else:
            trust_reason = "strong metadata+acceptable text"
        logger.info(f"  Cross-agent trust ({trust_reason}): "
                     f"metadata ({meta_result.score:.2f}) confirms legitimacy — "
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

    # === Government provenance override ===
    # When metadata agent confirms a known government tool (score >= 0.90)
    # AND text agent has no hard forgery indicators, visual anomalies are
    # rendering artifacts (wkhtmltopdf, DigiLocker, NSDL). Floor at 0.75.
    if metadata_govt_provenance and not text_has_hard_indicators and dis < 0.75:
        govt_floor = 0.75
        logger.info(f"  Government provenance override: DIS {dis:.3f} → {govt_floor:.3f} "
                     f"(metadata={meta_result.score:.2f} confirms govt tool, "
                     f"no hard text indicators — visual anomalies are rendering artifacts)")
        dis = govt_floor

    # === Strong metadata floor (fallback for edge cases) ===
    # When metadata STRONGLY confirms legitimacy (>= 0.95, government tool),
    # visual false positives from ManTraNet/ELA on PDF-rendered docs should
    # not sink the score below a reasonable floor.
    elif visual_meta_override and dis < 0.45:
        meta_floor = 0.45
        logger.info(f"  Strong metadata floor: DIS {dis:.3f} → {meta_floor:.3f} "
                     f"(metadata={meta_result.score:.2f} strongly confirms legitimacy)")
        dis = meta_floor

    # WhatsApp-specific scoring adjustments
    if is_whatsapp:
        # WhatsApp compression introduces visual artifacts (ELA/noise anomalies).
        # If text agent is clean/acceptable (>= 0.65), trust it more and soften visual.
        if visual_result and text_result:
            if text_result.score >= 0.65 and visual_result.score >= 0.40:
                # Text is acceptable + visual is only mildly flagging → likely compression
                # artifacts on a real document. Ensure DIS reaches Authentic range if no hard indicators.
                if not text_has_hard_indicators:
                    wa_floor = 0.86
                else:
                    wa_floor = 0.60
                    
                if dis < wa_floor:
                    logger.info(f"  WhatsApp clean floor: DIS {dis:.3f} → {wa_floor:.3f} "
                                 f"(text={text_result.score:.3f} clean, visual={visual_result.score:.3f} mild)")
                    dis = wa_floor

        # If visual agent flags STRONG issues (< 0.40), apply extra penalty
        # — this means ManTraNet or ELA found real tampering, not just artifacts
        if visual_result and visual_result.score < 0.40:
            wa_ceiling = min(dis, visual_result.score + 0.10)
            if wa_ceiling < dis:
                logger.info(f"  WhatsApp visual penalty: DIS {dis:.3f} → {wa_ceiling:.3f} "
                             f"(visual={visual_result.score:.3f})")
                dis = wa_ceiling

    # === EVIDENCE-BASED SCORING ===
    # Scan all agent findings for hard indicators, soft indicators, and
    # authentic signals. This complements the score-based ceiling logic
    # with semantic understanding of what was actually found.
    dis = _apply_evidence_based_adjustments(
        dis, agent_results, active_agents,
        metadata_strongly_legit=bool(metadata_strongly_legit),
        is_whatsapp=is_whatsapp,
    )

    # Re-apply floors after evidence scoring (evidence penalties may have
    # dropped DIS below floors that were set earlier)
    if metadata_govt_provenance and not text_has_hard_indicators and dis < 0.75:
        logger.info(f"  Government provenance floor re-applied: DIS {dis:.3f} → 0.750")
        dis = 0.75
    if is_whatsapp and visual_result and text_result:
        if text_result.score >= 0.65 and visual_result.score >= 0.40 and not text_has_hard_indicators and dis < 0.86:
            logger.info(f"  WhatsApp clean floor re-applied: DIS {dis:.3f} → 0.860")
            dis = 0.86
        elif text_result.score >= 0.65 and visual_result.score >= 0.40 and dis < 0.60:
            logger.info(f"  WhatsApp clean floor (hard indicators) re-applied: DIS {dis:.3f} → 0.600")
            dis = 0.60
    if visual_meta_override and dis < 0.45:
        logger.info(f"  Strong metadata floor re-applied: DIS {dis:.3f} → 0.450")
        dis = 0.45

    # Agent convergence bonus/penalty (applied after evidence)
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
    if document.file_size_bytes > 1_048_576:
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
                if available == 0:
                    # No metadata at all — agent has no real signal, just
                    # defaults. Reduce weight to near-zero so it doesn't
                    # pollute the score.
                    weight *= 0.15
                    logger.debug(f"Metadata agent weight heavily reduced: zero metadata fields")
                elif available <= 1:
                    weight *= 0.35
                    logger.debug(f"Metadata agent weight reduced: sparse metadata ({available} fields)")

        adjusted.append((result, float(np.clip(weight, 0.0, 1.0))))

    # === Visual-text disagreement: boost visual when signals conflict ===
    # When visual agent flags anomalies (< 0.65) but text agent sees nothing
    # wrong (> 0.85), the visual signal should carry more weight. Visual
    # anomalies from ManTraNet/ELA are harder to produce as false positives
    # than clean text (which is trivial to achieve with digital editing).
    vis_idx = next((i for i, (r, _) in enumerate(adjusted)
                    if "visual" in r.agent_name.lower()), None)
    txt_idx = next((i for i, (r, _) in enumerate(adjusted)
                    if "text" in r.agent_name.lower()), None)
    if vis_idx is not None and txt_idx is not None:
        vis_r, vis_w = adjusted[vis_idx]
        txt_r, txt_w = adjusted[txt_idx]
        if vis_r.score < 0.70 and txt_r.score > 0.85 and vis_w > 0.1:
            # Stronger boost when metadata also has no provenance
            meta_idx = next((i for i, (r, _) in enumerate(adjusted)
                             if "metadata" in r.agent_name.lower()), None)
            no_provenance = (meta_idx is not None and adjusted[meta_idx][1] < 0.15)
            boost = 1.4 if no_provenance else 1.25
            adjusted[vis_idx] = (vis_r, float(np.clip(vis_w * boost, 0.0, 1.0)))
            logger.debug(f"Visual-text disagreement: visual weight boosted {vis_w:.2f} → "
                         f"{vis_w * boost:.2f} (visual={vis_r.score:.2f} flagging, "
                         f"text={txt_r.score:.2f} clean, provenance={'none' if no_provenance else 'present'})")

    return adjusted


def _apply_evidence_based_adjustments(
    dis: float,
    agent_results: list[AgentResult],
    active_agents: list[AgentResult],
    metadata_strongly_legit: bool = False,
    is_whatsapp: bool = False,
) -> float:
    """
    Scan agent findings for evidence-based signals that should override
    or adjust the weighted DIS.

    Three categories:
    - Hard indicators: instant DIS cap (e.g., editing tool on govt doc, QR mismatch)
    - Soft indicators: incremental penalty (e.g., ManTraNet forgery signal, noise anomaly)
    - Authentic signals: boost DIS toward authentic (e.g., QR validates, govt tool confirmed)
    """
    all_findings_text = []
    for r in agent_results:
        for f in r.findings:
            all_findings_text.append(f.description.lower())

    findings_joined = " ".join(all_findings_text)

    # === HARD INDICATORS → instant DIS cap ===
    hard_cap = None

    # Editing software on government document
    if "strong forgery indicator" in findings_joined or \
       "never issued from editing software" in findings_joined:
        hard_cap = 0.20
        logger.info("  Evidence: editing tool on govt doc → cap 0.20")

    # QR code contradicts OCR
    if "contradicts ocr" in findings_joined:
        cap = 0.15
        if hard_cap is None or cap < hard_cap:
            hard_cap = cap
        logger.info("  Evidence: QR contradicts OCR → cap 0.15")

    # Consumer/mobile tool on government ID
    if "never by consumer software" in findings_joined or \
       "never by consumer" in findings_joined:
        cap = 0.25
        if hard_cap is None or cap < hard_cap:
            hard_cap = cap
        logger.info("  Evidence: consumer tool on govt ID → cap 0.25")

    # ManTraNet strong forgery signal — but NOT when metadata strongly confirms
    # legitimacy (e.g., wkhtmltopdf-generated e-Aadhaar triggers ManTraNet false
    # positives due to PDF rendering artifacts)
    if "strong forgery signal" in findings_joined and "mantranet" in findings_joined:
        if not metadata_strongly_legit:
            cap = 0.25
            if hard_cap is None or cap < hard_cap:
                hard_cap = cap
            logger.info("  Evidence: ManTraNet strong forgery → cap 0.25")
        else:
            logger.info("  Evidence: ManTraNet strong forgery SKIPPED — metadata strongly confirms legitimacy")

    # Confirmed ELA across all quality levels (same exception for strong metadata)
    if "confirmed" in findings_joined and "all 3 quality levels" in findings_joined:
        if not metadata_strongly_legit:
            cap = 0.30
            if hard_cap is None or cap < hard_cap:
                hard_cap = cap
        logger.info("  Evidence: confirmed multi-scale ELA → cap 0.30")

    # Verhoeff checksum failure on non-government tool
    if "fails verhoeff checksum validation" in findings_joined:
        cap = 0.20
        if hard_cap is None or cap < hard_cap:
            hard_cap = cap
        logger.info("  Evidence: Aadhaar fails Verhoeff → cap 0.20")

    # No QR on Aadhaar
    if "no qr code detected on aadhaar" in findings_joined:
        if not is_whatsapp:
            cap = 0.45
            if hard_cap is None or cap < hard_cap:
                hard_cap = cap
            logger.info("  Evidence: No QR on Aadhaar → cap 0.45")
        else:
            logger.info("  Evidence: No QR on Aadhaar SKIPPED — WhatsApp image")

    # Text Splicing / Digital tampering
    # NOTE: These indicators are natural artefacts in multi-script (Hindi+English)
    # government documents and scanned IDs. Skip the cap when strong government
    # metadata (≥0.95) confirms legitimacy — same exemption as ManTraNet.
    has_text_cluster = "low-confidence text cluster" in findings_joined
    has_baseline = "baseline misalignment" in findings_joined
    has_spacing = "character spacing anomaly" in findings_joined

    splicing_score = int(has_text_cluster) + int(has_baseline) + int(has_spacing)
    if splicing_score >= 2:
        if not metadata_strongly_legit:
            cap = 0.38
            if hard_cap is None or cap < hard_cap:
                hard_cap = cap
            logger.info(f"  Evidence: Multiple text splicing indicators ({splicing_score}/3) → cap 0.38")
        else:
            logger.info(
                f"  Evidence: Text splicing indicators ({splicing_score}/3) SKIPPED "
                f"— strong govt metadata confirms legitimacy (multi-script baseline/spacing artefacts expected)"
            )
    elif splicing_score == 1:
        if not metadata_strongly_legit:
            cap = 0.58
            if hard_cap is None or cap < hard_cap:
                hard_cap = cap
            logger.info("  Evidence: Single text splicing indicator → cap 0.58")
        else:
            logger.info("  Evidence: Single text splicing indicator SKIPPED — strong govt metadata")

    if hard_cap is not None and dis > hard_cap:
        dis = hard_cap

    # === SOFT INDICATORS → incremental penalty ===
    soft_penalty = 0.0

    # ManTraNet detects manipulation (moderate)
    if "possible manipulation detected" in findings_joined:
        soft_penalty += 0.05
    if "mantranet" in findings_joined and "elevated anomaly" in findings_joined:
        soft_penalty += 0.04
    if "mantranet" in findings_joined and "minor anomaly" in findings_joined:
        soft_penalty += 0.02

    if soft_penalty > 0:
        dis = max(0.0, dis - soft_penalty)
        logger.info(f"  Evidence soft penalty: -{soft_penalty:.2f}")

    # === AUTHENTIC SIGNALS → boost toward authentic ===
    auth_boost = 0.0

    # QR code validates with OCR
    if "qr code validates" in findings_joined:
        auth_boost += 0.05

    # WhatsApp image correctly identified (not suspicious)
    if "messaging app" in findings_joined and "not a forgery indicator" in findings_joined:
        auth_boost += 0.02
        
    # DigiLocker Screenshot detected
    if "digilocker screenshot" in findings_joined:
        auth_boost += 0.15

    # Government tool confirmed
    if any(r.score >= 0.90 for r in active_agents
           if "metadata" in r.agent_name.lower()):
        # Metadata agent is very confident → slight boost
        auth_boost += 0.02

    if auth_boost > 0 and dis < 1.0:
        dis = min(1.0, dis + auth_boost)
        logger.info(f"  Evidence auth boost: +{auth_boost:.2f}")

    return dis


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
