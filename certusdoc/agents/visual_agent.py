"""
Visual Tamper Detection Agent

Detection methods:
1. Multi-scale ELA (Q90/Q75/Q50) — catches JPEG recompression, splicing
2. Copy-move detection (ORB feature matching) — catches duplicated regions
3. JPEG quantization table analysis — detects double compression
4. Noise consistency analysis — catches spliced regions with different noise

This agent catches: splicing, copy-move, image manipulation, JPEG recompression
"""
import time
import io
from typing import Optional
from collections import Counter

import cv2
import numpy as np
from PIL import Image
from loguru import logger

from certusdoc.agents.base import BaseAgent
from certusdoc.models import Document, AgentResult, AgentFinding
from certusdoc.utils.doc_detector import detect_document_type


class VisualTamperAgent(BaseAgent):
    """Detects visual tampering using ELA and optional deep learning models."""

    def __init__(self, trufor_model_path: Optional[str] = None):
        super().__init__(name="Visual Tamper Agent")
        self.trufor_model_path = trufor_model_path
        self.trufor_model = None

        if trufor_model_path:
            self._load_trufor(trufor_model_path)

    def analyze(self, document: Document) -> AgentResult:
        start_time = time.time()
        all_findings = []
        page_scores = []
        combined_heatmap = None

        # Detect document type for adaptive thresholds
        doc_class = detect_document_type(document.ocr_text)

        for page_idx, page_img in enumerate(document.pages):
            sub_scores = {}

            # --- Multi-Scale ELA Analysis ---
            ela_score, ela_heatmap, ela_findings = self._run_multiscale_ela(page_img, page_idx)
            all_findings.extend(ela_findings)
            sub_scores["ela"] = ela_score

            # --- Copy-Move Detection (ORB) ---
            copymove_score, copymove_findings = self._detect_copy_move(
                page_img, page_idx, doc_class
            )
            all_findings.extend(copymove_findings)
            sub_scores["copymove"] = copymove_score

            # --- JPEG Quantization Analysis ---
            quant_score, quant_findings = self._analyze_jpeg_artifacts(
                page_img, document, page_idx
            )
            all_findings.extend(quant_findings)
            sub_scores["jpeg_quant"] = quant_score

            # --- Noise Consistency Analysis ---
            noise_score, noise_findings = self._analyze_noise_consistency(
                page_img, page_idx, is_structured=(doc_class and doc_class.is_structured)
            )
            all_findings.extend(noise_findings)
            sub_scores["noise"] = noise_score

            # --- TruFor (if available) ---
            if self.trufor_model is not None:
                trufor_score, trufor_heatmap, trufor_findings = self._run_trufor(
                    page_img, page_idx
                )
                all_findings.extend(trufor_findings)
                sub_scores["trufor"] = trufor_score
                if ela_heatmap is not None and trufor_heatmap is not None:
                    ela_heatmap = cv2.addWeighted(ela_heatmap, 0.4, trufor_heatmap, 0.6, 0)

            # === SCORING: severity-driven, not just weighted average ===
            # The page score is capped by the WORST sub-score.
            # A single strong detection signal should dominate.
            worst_sub = min(sub_scores.values())

            if self.trufor_model is not None:
                weighted = (0.15 * ela_score + 0.15 * copymove_score +
                            0.10 * quant_score + 0.10 * noise_score +
                            0.50 * sub_scores["trufor"])
            else:
                weighted = (0.30 * ela_score + 0.30 * copymove_score +
                            0.20 * quant_score + 0.20 * noise_score)

            # The page score is the LESSER of the weighted avg and
            # (worst_sub + 0.15) — so a strong signal from any method
            # pulls the whole score down with tight coupling.
            ceiling_from_worst = worst_sub + 0.15
            page_score = min(weighted, ceiling_from_worst)
            page_score = max(0.0, min(1.0, page_score))

            page_scores.append(page_score)

            if combined_heatmap is None:
                combined_heatmap = ela_heatmap
            else:
                if page_score < page_scores[-2] if len(page_scores) > 1 else True:
                    combined_heatmap = ela_heatmap

        final_score = min(page_scores) if page_scores else 1.0
        reliability = self._compute_reliability(document)
        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(f"Visual agent complete: score={final_score:.3f}, "
                     f"reliability={reliability:.2f}, {len(all_findings)} findings, "
                     f"{elapsed_ms:.0f}ms")

        return AgentResult(
            agent_name=self.name,
            score=final_score,
            reliability_weight=reliability,
            findings=all_findings,
            heatmap=combined_heatmap,
            processing_time_ms=elapsed_ms,
            details={
                "methods_used": ["multi_scale_ELA", "copy_move_ORB",
                                 "JPEG_quantization", "noise_consistency"]
                                + (["TruFor"] if self.trufor_model else []),
                "pages_analyzed": len(document.pages),
                "per_page_scores": page_scores,
            }
        )

    # ================================================================
    # Multi-Scale ELA
    # ================================================================

    def _run_ela_single(
        self, rgb: np.ndarray, quality: int
    ) -> tuple[np.ndarray, float, float, float]:
        """Run ELA at a single JPEG quality level. Returns (ela_map, mean, std, max)."""
        pil_img = Image.fromarray(rgb)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        resaved = np.array(Image.open(buffer))
        diff = cv2.absdiff(rgb, resaved).astype(np.float32)
        ela_map = np.mean(diff, axis=2)
        return ela_map, float(np.mean(ela_map)), float(np.std(ela_map)), float(np.max(ela_map))

    def _run_multiscale_ela(
        self, image: np.ndarray, page_idx: int
    ) -> tuple[float, np.ndarray, list[AgentFinding]]:
        """
        Multi-scale ELA at Q90, Q75, Q50.
        Anomalies persisting across all 3 scales = real tampering (hard penalty).
        Anomaly at only 1 scale = possible false positive (soft penalty).
        """
        findings = []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        qualities = [90, 75, 50]
        ela_maps = []
        anomaly_masks = []
        anomaly_ratios = []

        for q in qualities:
            ela_map, mean_ela, std_ela, max_ela = self._run_ela_single(rgb, q)
            ela_maps.append(ela_map)

            threshold = mean_ela + 2.0 * std_ela
            mask = ela_map > threshold
            ratio = float(np.sum(mask)) / ela_map.size
            anomaly_masks.append(mask)
            anomaly_ratios.append(ratio)

        # Cross-scale analysis
        combined_mask = sum(m.astype(np.float32) for m in anomaly_masks)

        # Regions flagged in ALL 3 scales — confirmed tampering
        persistent_3 = combined_mask >= 3
        persistent_3_ratio = float(np.sum(persistent_3)) / persistent_3.size

        # Regions flagged in 2+ scales
        persistent_2 = combined_mask >= 2
        persistent_2_ratio = float(np.sum(persistent_2)) / persistent_2.size

        # Scoring based on persistence level
        if persistent_3_ratio > 0.02:
            # Confirmed tampering: anomaly survives all quality levels
            score = max(0.0, 0.3 - persistent_3_ratio * 5)
            findings.append(AgentFinding(
                description=(
                    f"CONFIRMED: ELA anomaly persists across ALL 3 quality levels "
                    f"(Q90/Q75/Q50). {persistent_3_ratio*100:.1f}% of image affected. "
                    f"This strongly indicates tampering."
                ),
                severity=min(1.0, 0.7 + persistent_3_ratio * 3),
                page=page_idx
            ))
        elif persistent_2_ratio > 0.03:
            # Likely tampering
            score = max(0.1, 0.5 - persistent_2_ratio * 4)
            findings.append(AgentFinding(
                description=(
                    f"Multi-scale ELA: {persistent_2_ratio*100:.1f}% of image flagged "
                    f"across 2+ quality levels. Per-scale anomaly ratios: "
                    f"{[f'{r*100:.1f}%' for r in anomaly_ratios]}"
                ),
                severity=min(1.0, 0.5 + persistent_2_ratio * 3),
                page=page_idx
            ))
        elif max(anomaly_ratios) > 0.08:
            # Single-scale anomaly — possible but less certain
            worst_q_idx = anomaly_ratios.index(max(anomaly_ratios))
            worst_ratio = anomaly_ratios[worst_q_idx]
            score = max(0.3, 0.7 - worst_ratio * 3)
            findings.append(AgentFinding(
                description=(
                    f"ELA anomaly at Q{qualities[worst_q_idx]}: "
                    f"{worst_ratio*100:.1f}% of image affected. "
                    f"Per-scale: {[f'{r*100:.1f}%' for r in anomaly_ratios]}"
                ),
                severity=min(1.0, worst_ratio * 4),
                page=page_idx
            ))
        elif max(anomaly_ratios) > 0.04:
            score = max(0.5, 0.85 - max(anomaly_ratios) * 3)
        else:
            score = min(1.0, 0.90 + (1.0 - max(anomaly_ratios)) * 0.10)

        # === Block-based ELA variance analysis ===
        # Divides image into blocks and checks for ELA outliers.
        # Forged regions have systematically different ELA than surrounding blocks.
        bh, bw = ela_maps[0].shape
        block_sz = max(32, min(bh, bw) // 8)
        block_means_q90 = []
        for yb in range(0, bh - block_sz, block_sz):
            for xb in range(0, bw - block_sz, block_sz):
                bm = float(np.mean(ela_maps[0][yb:yb+block_sz, xb:xb+block_sz]))
                block_means_q90.append(bm)

        if len(block_means_q90) >= 4:
            bm_arr = np.array(block_means_q90)
            bm_mean = float(np.mean(bm_arr))
            bm_std = float(np.std(bm_arr))
            if bm_mean > 0 and bm_std > 0:
                cv_ela = bm_std / bm_mean  # Coefficient of variation
                outlier_blocks = int(np.sum(bm_arr > bm_mean + 2.0 * bm_std))
                if cv_ela > 0.60 and outlier_blocks >= 2:
                    # High block variance with multiple outliers = localized tampering
                    block_penalty = min(score, max(0.30, 0.55 - cv_ela * 0.25))
                    if block_penalty < score:
                        score = block_penalty
                        findings.append(AgentFinding(
                            description=(
                                f"ELA block variance: CV={cv_ela:.2f}, "
                                f"{outlier_blocks} outlier blocks of "
                                f"{len(block_means_q90)}. Localized ELA anomaly "
                                f"suggests region-level tampering."
                            ),
                            severity=min(0.8, cv_ela),
                            page=page_idx
                        ))
                elif cv_ela > 0.40 and outlier_blocks >= 1:
                    block_penalty = min(score, max(0.50, 0.70 - cv_ela * 0.20))
                    if block_penalty < score:
                        score = block_penalty
                        findings.append(AgentFinding(
                            description=(
                                f"Moderate ELA block variance: CV={cv_ela:.2f}, "
                                f"{outlier_blocks} outlier blocks."
                            ),
                            severity=min(0.5, cv_ela),
                            page=page_idx
                        ))

        # Bounding box for largest persistent anomaly
        report_mask = persistent_3 if persistent_3_ratio > 0.01 else persistent_2
        if float(np.sum(report_mask)) / report_mask.size > 0.005:
            contours, _ = cv2.findContours(
                report_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                findings.append(AgentFinding(
                    description=f"Largest ELA anomaly region at ({x},{y}) size {w}x{h}",
                    severity=min(1.0, persistent_2_ratio * 3),
                    region=(x, y, w, h),
                    page=page_idx
                ))

        # Heatmap
        combined_ela = (ela_maps[0] * 0.5 + ela_maps[1] * 0.3 + ela_maps[2] * 0.2)
        ela_normalized = np.clip(combined_ela * 15, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(ela_normalized, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

        return score, heatmap, findings

    # ================================================================
    # Copy-Move Detection (ORB)
    # ================================================================

    def _detect_copy_move(
        self, image: np.ndarray, page_idx: int, doc_class=None
    ) -> tuple[float, list[AgentFinding]]:
        """
        Detect copy-move forgery using ORB feature matching.

        For structured documents (government IDs, certificates, invoices),
        thresholds are raised because repeated design elements (logos, borders,
        headers/footers) are expected.
        """
        findings = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        scale = 1.0
        if max(h, w) > 2000:
            scale = 2000.0 / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)

        scaled_h, scaled_w = gray.shape

        orb = cv2.ORB_create(nfeatures=3000)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 20:
            return 1.0, findings

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(descriptors, descriptors, k=5)

        # Structured/govt docs have repeated elements → use looser distance
        dist_threshold = 30 if (doc_class and doc_class.is_structured) else 35
        min_displacement = 30 * scale
        max_displacement = max(h, w) * scale * 0.6

        suspicious_pairs = []
        for match_group in matches:
            for m in match_group:
                if m.queryIdx == m.trainIdx:
                    continue
                if m.distance > dist_threshold:
                    continue

                pt1 = keypoints[m.queryIdx].pt
                pt2 = keypoints[m.trainIdx].pt
                dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

                if min_displacement < dist < max_displacement:
                    suspicious_pairs.append((pt1, pt2, m.distance, dist))

        total_suspicious = len(suspicious_pairs)

        if total_suspicious < 5:
            return 1.0, findings

        # Cluster by displacement vector — TRUE copy-move will show a dominant cluster
        bucket_size = 20
        buckets = Counter()
        for pt1, pt2, _, _ in suspicious_pairs:
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            key = (int(dx / bucket_size), int(dy / bucket_size))
            buckets[key] += 1

        if not buckets:
            return 1.0, findings

        most_common_bucket, cluster_count = buckets.most_common(1)[0]
        consistency_ratio = cluster_count / total_suspicious if total_suspicious > 0 else 0

        # === FILTER: Structured document exceptions ===
        # Government IDs, certificates, invoices have repeated design elements
        # (same logo top+bottom, same borders, same header/footer).
        # Matches between top-half and bottom-half are often layout symmetry.
        is_structured = doc_class and doc_class.is_structured
        is_govt = doc_class and doc_class.is_government

        if is_structured:
            # For structured docs, filter out matches between top and bottom halves
            # (front/back of card, header/footer repetition)
            mid_y = scaled_h / 2.0
            margin = scaled_h * 0.15  # 15% margin around midpoint
            cross_half_matches = 0
            for pt1, pt2, _, _ in suspicious_pairs:
                if ((pt1[1] < mid_y - margin and pt2[1] > mid_y + margin) or
                        (pt2[1] < mid_y - margin and pt1[1] > mid_y + margin)):
                    cross_half_matches += 1

            cross_ratio = cross_half_matches / total_suspicious if total_suspicious > 0 else 0
            if cross_ratio > 0.4:
                # Most matches are top-half ↔ bottom-half → likely layout symmetry
                return 0.90, findings

        # === FILTER: Low consistency = scattered noise, not copy-move ===
        min_consistency = 0.20 if is_govt else (0.18 if is_structured else 0.15)
        if consistency_ratio < min_consistency:
            if total_suspicious < 50:
                return 0.95, findings
            else:
                return 0.90 if is_structured else 0.85, findings

        # Use cluster_count (not total matches) as the primary signal
        effective_matches = cluster_count

        # Structured docs need MUCH higher match counts to be suspicious
        if is_govt:
            thresholds = (200, 100, 40, 20)  # Government ID
        elif is_structured:
            thresholds = (150, 70, 30, 15)   # Certificate, invoice, etc.
        else:
            thresholds = (80, 40, 15, 8)     # Unstructured documents

        # === SCORING based on clustered matches ===
        if effective_matches > thresholds[0]:
            score = min(0.20, max(0.05, 0.30 - effective_matches / 400.0))
            severity = 0.95
        elif effective_matches > thresholds[1]:
            score = min(0.40, max(0.15, 0.55 - effective_matches / 150.0))
            severity = 0.8
        elif effective_matches > thresholds[2]:
            score = min(0.60, max(0.30, 0.75 - effective_matches / 80.0))
            severity = 0.6
        elif effective_matches > thresholds[3]:
            score = min(0.75, max(0.50, 0.85 - effective_matches / 60.0))
            severity = 0.4
        else:
            score = 0.85
            severity = 0.25

        # Extra penalty if consistency is very high (strong copy-move signal)
        high_consistency_thresh = 0.40 if is_structured else 0.30
        if consistency_ratio > high_consistency_thresh and effective_matches >= 10:
            cluster_penalty = min(0.2, consistency_ratio * 0.4)
            score = max(0.0, score - cluster_penalty)

        # Build finding with bounding box
        target_dx = most_common_bucket[0] * bucket_size
        target_dy = most_common_bucket[1] * bucket_size
        cluster_pts = []
        for pt1, pt2, _, _ in suspicious_pairs:
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            if (abs(dx - target_dx) < bucket_size * 1.5 and
                    abs(dy - target_dy) < bucket_size * 1.5):
                cluster_pts.append(pt1)

        bbox = None
        if cluster_pts:
            xs = [p[0] / scale for p in cluster_pts]
            ys = [p[1] / scale for p in cluster_pts]
            bbox = (int(min(xs)), int(min(ys)),
                    int(max(xs) - min(xs) + 1), int(max(ys) - min(ys) + 1))

        findings.append(AgentFinding(
            description=(
                f"Copy-move detected: {effective_matches} clustered matches "
                f"(of {total_suspicious} total), displacement "
                f"({target_dx/scale:.0f}, {target_dy/scale:.0f})px, "
                f"consistency: {consistency_ratio*100:.0f}%"
            ),
            severity=severity,
            region=bbox,
            page=page_idx
        ))

        return score, findings

    # ================================================================
    # JPEG Quantization Analysis
    # ================================================================

    def _analyze_jpeg_artifacts(
        self, image: np.ndarray, document: Document, page_idx: int
    ) -> tuple[float, list[AgentFinding]]:
        """Analyze JPEG compression artifacts to detect double compression."""
        findings = []
        score = 1.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h, w = gray.shape
        h8 = (h // 8) * 8
        w8 = (w // 8) * 8
        gray = gray[:h8, :w8]

        # Boundary vs interior discontinuity — safe slicing
        h_a, h_b = gray[7::8, :], gray[8::8, :]
        mh = min(h_a.shape[0], h_b.shape[0])
        h_boundaries = np.abs(h_a[:mh] - h_b[:mh])

        v_a, v_b = gray[:, 7::8], gray[:, 8::8]
        mv = min(v_a.shape[1], v_b.shape[1])
        v_boundaries = np.abs(v_a[:, :mv] - v_b[:, :mv])

        ih_a, ih_b = gray[3::8, :], gray[4::8, :]
        mih = min(ih_a.shape[0], ih_b.shape[0])
        interior_h = np.abs(ih_a[:mih] - ih_b[:mih])

        iv_a, iv_b = gray[:, 3::8], gray[:, 4::8]
        miv = min(iv_a.shape[1], iv_b.shape[1])
        interior_v = np.abs(iv_a[:, :miv] - iv_b[:, :miv])

        mean_h_disc = np.mean(h_boundaries)
        mean_v_disc = np.mean(v_boundaries)
        mean_h_int = np.mean(interior_h)
        mean_v_int = np.mean(interior_v)

        if mean_h_int > 0 and mean_v_int > 0:
            ratio_h = mean_h_disc / mean_h_int
            ratio_v = mean_v_disc / mean_v_int

            if ratio_h < 0.7 or ratio_v < 0.7:
                score = 0.4
                findings.append(AgentFinding(
                    description=(
                        f"Strong double compression signal: JPEG block boundary "
                        f"ratios H={ratio_h:.2f}, V={ratio_v:.2f} (expected ~1.0)"
                    ),
                    severity=0.7,
                    page=page_idx
                ))
            elif ratio_h < 0.85 or ratio_v < 0.85:
                score = 0.6
                findings.append(AgentFinding(
                    description=(
                        f"Possible double compression: block boundary "
                        f"ratios H={ratio_h:.2f}, V={ratio_v:.2f}"
                    ),
                    severity=0.5,
                    page=page_idx
                ))
            elif ratio_h > 1.5 or ratio_v > 1.5:
                score = 0.7
                findings.append(AgentFinding(
                    description=(
                        f"Unusual JPEG block patterns: H={ratio_h:.2f}, V={ratio_v:.2f}"
                    ),
                    severity=0.3,
                    page=page_idx
                ))

        return score, findings

    # ================================================================
    # Noise Consistency
    # ================================================================

    def _analyze_noise_consistency(
        self, image: np.ndarray, page_idx: int, is_structured: bool = False
    ) -> tuple[float, list[AgentFinding]]:
        """Check if noise levels are consistent across the image."""
        findings = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

        h, w = gray.shape
        block_size = 64
        noise_levels = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                laplacian = cv2.Laplacian(block, cv2.CV_64F)
                noise = np.std(laplacian)
                noise_levels.append((x, y, noise))

        if not noise_levels:
            return 1.0, findings

        noise_values = [n[2] for n in noise_levels]
        mean_noise = np.mean(noise_values)
        std_noise = np.std(noise_values)

        # Structured docs (government IDs, certificates) need wider tolerance
        # due to inherent design variation (logos, text, borders, blank areas)
        sigma_thresh = 3.0 if is_structured else 2.5

        anomalous_blocks = []
        for x, y, noise in noise_levels:
            if std_noise > 0 and abs(noise - mean_noise) > sigma_thresh * std_noise:
                anomalous_blocks.append((x, y, noise))

        anomaly_ratio = len(anomalous_blocks) / len(noise_levels)

        if anomaly_ratio > 0.08:
            score = max(0.1, 0.6 - anomaly_ratio * 5)
            findings.append(AgentFinding(
                description=(
                    f"Noise inconsistency: {len(anomalous_blocks)}/{len(noise_levels)} "
                    f"blocks anomalous ({anomaly_ratio*100:.1f}%). "
                    f"Mean noise: {mean_noise:.1f}, std: {std_noise:.1f}"
                ),
                severity=min(1.0, anomaly_ratio * 5),
                page=page_idx
            ))
        elif anomaly_ratio > 0.02:
            score = max(0.4, 0.8 - anomaly_ratio * 5)
            findings.append(AgentFinding(
                description=(
                    f"Minor noise inconsistency: {len(anomalous_blocks)}/{len(noise_levels)} "
                    f"blocks anomalous ({anomaly_ratio*100:.1f}%)"
                ),
                severity=min(0.5, anomaly_ratio * 3),
                page=page_idx
            ))
        else:
            score = 1.0

        return score, findings

    # ================================================================
    # Reliability
    # ================================================================

    def _compute_reliability(self, document: Document) -> float:
        """Compute reliability of visual analysis based on input quality."""
        if not document.pages:
            return 0.0

        page = document.pages[0]
        h, w = page.shape[:2]
        resolution_factor = min(1.0, (h * w) / (2000 * 3000))

        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_factor = min(1.0, laplacian_var / 500.0)

        return max(0.1, 0.5 * resolution_factor + 0.5 * sharpness_factor)

    # ================================================================
    # TruFor (placeholder)
    # ================================================================

    def _load_trufor(self, model_path: str) -> None:
        """Load TruFor model weights. Implement when weights are available."""
        logger.info(f"TruFor model loading from: {model_path}")
        logger.warning("TruFor not yet integrated — using ELA-based detection only")

    def _run_trufor(
        self, image: np.ndarray, page_idx: int
    ) -> tuple[float, Optional[np.ndarray], list[AgentFinding]]:
        """Run TruFor inference. Placeholder until model is integrated."""
        return 1.0, None, []
