"""
Text Forensics Agent

Detection methods:
1. OCR confidence variance — forged text regions have different OCR confidence
2. Font consistency analysis — detect mixed fonts within same document region
3. Baseline alignment — detect misaligned text baselines
4. Character spacing analysis — detect kerning anomalies
5. Text region comparison — compare text style consistency across regions

This agent catches: text splicing, font replacement, character-level forgery
"""
import time
from collections import Counter
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from certusdoc.agents.base import BaseAgent
from certusdoc.models import Document, AgentResult, AgentFinding
from certusdoc.utils.doc_detector import detect_document_type, ScriptType


class TextForensicsAgent(BaseAgent):
    """Detects text-level tampering using OCR forensics."""

    def __init__(self):
        super().__init__(name="Text Forensics Agent")

    def analyze(self, document: Document) -> AgentResult:
        start_time = time.time()
        all_findings = []
        page_scores = []

        # Detect document type and script for adaptive thresholds
        doc_class = detect_document_type(document.ocr_text)

        for page_idx in range(len(document.pages)):
            page_img = document.pages[page_idx]
            word_data = document.ocr_word_data[page_idx]
            avg_conf = document.ocr_confidence[page_idx]

            if not word_data:
                page_scores.append(1.0)
                continue

            # --- OCR Confidence Variance Analysis ---
            conf_score, conf_findings = self._analyze_confidence_variance(
                word_data, page_idx
            )
            all_findings.extend(conf_findings)

            # --- Baseline Alignment Analysis ---
            baseline_score, baseline_findings = self._analyze_baseline_alignment(
                word_data, page_idx, doc_class
            )
            all_findings.extend(baseline_findings)

            # --- Character Spacing Analysis ---
            spacing_score, spacing_findings = self._analyze_character_spacing(
                word_data, page_img, page_idx
            )
            all_findings.extend(spacing_findings)

            # --- Font Size Consistency ---
            fontsize_score, fontsize_findings = self._analyze_font_size_consistency(
                word_data, page_idx
            )
            all_findings.extend(fontsize_findings)

            # --- Text Block Regularity ---
            regularity_score, regularity_findings = self._analyze_text_block_regularity(
                word_data, page_img, page_idx
            )
            all_findings.extend(regularity_findings)

            # --- Regional OCR Confidence Comparison ---
            regional_score, regional_findings = self._analyze_regional_confidence(
                word_data, page_img, page_idx
            )
            all_findings.extend(regional_findings)

            page_score = (
                0.25 * conf_score
                + 0.20 * baseline_score
                + 0.15 * spacing_score
                + 0.10 * fontsize_score
                + 0.10 * regularity_score
                + 0.20 * regional_score
            )
            page_scores.append(page_score)

        final_score = min(page_scores) if page_scores else 1.0
        reliability = self._compute_reliability(document)
        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(f"Text agent complete: score={final_score:.3f}, "
                     f"reliability={reliability:.2f}, {len(all_findings)} findings, "
                     f"{elapsed_ms:.0f}ms")

        return AgentResult(
            agent_name=self.name,
            score=final_score,
            reliability_weight=reliability,
            findings=all_findings,
            processing_time_ms=elapsed_ms,
            details={
                "pages_analyzed": len(document.pages),
                "per_page_scores": page_scores,
                "total_words_analyzed": sum(len(wd) for wd in document.ocr_word_data),
            }
        )

    def _analyze_confidence_variance(
        self, word_data: list[dict], page_idx: int
    ) -> tuple[float, list[AgentFinding]]:
        """
        Detect regions where OCR confidence drops significantly.
        Forged text often produces different OCR confidence than authentic text.
        """
        findings = []
        if len(word_data) < 5:
            return 1.0, findings

        confidences = [w["conf"] for w in word_data]
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)

        # Find words with anomalously low confidence
        low_conf_words = []
        for word in word_data:
            if std_conf > 0 and (mean_conf - word["conf"]) > 2.0 * std_conf:
                low_conf_words.append(word)

        if not low_conf_words:
            return 1.0, findings

        anomaly_ratio = len(low_conf_words) / len(word_data)

        if anomaly_ratio > 0.03:
            # Check if low-confidence words cluster spatially
            clusters = self._find_spatial_clusters(low_conf_words)

            score = max(0.0, 1.0 - anomaly_ratio * 8)

            sample_words = [w["text"] for w in low_conf_words[:5]]
            findings.append(AgentFinding(
                description=(
                    f"OCR confidence anomaly: {len(low_conf_words)} words with "
                    f"significantly lower confidence (mean: {mean_conf:.1f}, "
                    f"anomalous words: {', '.join(sample_words)}). "
                    f"Found {len(clusters)} spatial cluster(s)."
                ),
                severity=min(1.0, anomaly_ratio * 5),
                page=page_idx
            ))

            # Report each cluster location
            for cluster in clusters:
                if len(cluster) >= 2:
                    xs = [w["x"] for w in cluster]
                    ys = [w["y"] for w in cluster]
                    ws = [w["x"] + w["w"] for w in cluster]
                    hs = [w["y"] + w["h"] for w in cluster]
                    bbox = (min(xs), min(ys), max(ws) - min(xs), max(hs) - min(ys))
                    findings.append(AgentFinding(
                        description=f"Low-confidence text cluster ({len(cluster)} words)",
                        severity=0.6,
                        region=bbox,
                        page=page_idx
                    ))
        else:
            score = 0.95

        return score, findings

    def _analyze_baseline_alignment(
        self, word_data: list[dict], page_idx: int, doc_class=None
    ) -> tuple[float, list[AgentFinding]]:
        """
        Check if text baselines are properly aligned within rows.
        Forged text often has slightly different vertical positioning.

        Multi-script documents (Hindi+English, Tamil+English) naturally have
        different baselines, so we use a higher tolerance for those.
        """
        findings = []
        if len(word_data) < 5:
            return 1.0, findings

        # Adaptive baseline tolerance based on script type
        # Multi-script docs (Devanagari + Latin) have inherently different baselines
        if doc_class and doc_class.is_multi_language:
            baseline_tolerance = 0.30  # 30% of char height for multi-script
        elif doc_class and doc_class.script_type not in (ScriptType.LATIN_ONLY,):
            baseline_tolerance = 0.25  # 25% for non-Latin single script
        else:
            baseline_tolerance = 0.12  # 12% for Latin-only

        # Group words into approximate text lines
        lines = self._group_into_lines(word_data)
        misaligned_lines = []

        for line in lines:
            if len(line) < 3:
                continue

            # Check baseline consistency (bottom of bounding box)
            baselines = [w["y"] + w["h"] for w in line]
            baseline_std = np.std(baselines)
            baseline_mean = np.mean(baselines)

            # Average character height for this line
            avg_height = np.mean([w["h"] for w in line])

            # Baseline deviation threshold: adaptive
            if baseline_std > avg_height * baseline_tolerance:
                misaligned_lines.append((line, baseline_std, avg_height))

        if not misaligned_lines:
            return 1.0, findings

        misalign_ratio = len(misaligned_lines) / max(1, len(lines))

        if misalign_ratio > 0.05:
            score = max(0.0, 1.0 - misalign_ratio * 5)
            worst_line, worst_std, worst_height = max(
                misaligned_lines, key=lambda x: x[1]
            )
            findings.append(AgentFinding(
                description=(
                    f"Baseline misalignment detected in {len(misaligned_lines)} of "
                    f"{len(lines)} text lines. Worst deviation: "
                    f"{worst_std:.1f}px (character height: {worst_height:.0f}px)"
                ),
                severity=min(1.0, misalign_ratio * 4),
                page=page_idx
            ))
        else:
            score = 0.95

        return score, findings

    def _analyze_character_spacing(
        self, word_data: list[dict], page_img: np.ndarray, page_idx: int
    ) -> tuple[float, list[AgentFinding]]:
        """
        Analyze inter-word spacing consistency within text lines.
        """
        findings = []
        lines = self._group_into_lines(word_data)
        spacing_anomalies = []

        for line in lines:
            if len(line) < 3:
                continue

            # Sort by x position
            sorted_words = sorted(line, key=lambda w: w["x"])

            # Compute inter-word gaps
            gaps = []
            for i in range(1, len(sorted_words)):
                gap = sorted_words[i]["x"] - (sorted_words[i-1]["x"] + sorted_words[i-1]["w"])
                gaps.append(gap)

            if len(gaps) < 2:
                continue

            gap_mean = np.mean(gaps)
            gap_std = np.std(gaps)

            # Find words with anomalous spacing
            for i, gap in enumerate(gaps):
                if gap_std > 0 and abs(gap - gap_mean) > 2.5 * gap_std:
                    word_before = sorted_words[i]
                    word_after = sorted_words[i + 1]
                    spacing_anomalies.append((word_before, word_after, gap, gap_mean))

        if not spacing_anomalies:
            return 1.0, findings

        total_gaps = sum(max(0, len(line) - 1) for line in lines)
        anomaly_ratio = len(spacing_anomalies) / max(1, total_gaps)

        if anomaly_ratio > 0.02:
            score = max(0.0, 1.0 - anomaly_ratio * 10)
            findings.append(AgentFinding(
                description=(
                    f"Character spacing anomaly: {len(spacing_anomalies)} irregular "
                    f"gaps detected across {len(lines)} text lines"
                ),
                severity=min(1.0, anomaly_ratio * 5),
                page=page_idx
            ))
        else:
            score = 1.0

        return score, findings

    def _analyze_font_size_consistency(
        self, word_data: list[dict], page_idx: int
    ) -> tuple[float, list[AgentFinding]]:
        """
        Check for unexpected font size variations within text blocks.
        """
        findings = []
        if len(word_data) < 10:
            return 1.0, findings

        lines = self._group_into_lines(word_data)

        # Compute typical character height per line
        line_heights = []
        for line in lines:
            heights = [w["h"] for w in line]
            if heights:
                line_heights.append(np.median(heights))

        if len(line_heights) < 3:
            return 1.0, findings

        # Body text lines should have consistent height
        # (Excluding headers which are intentionally larger)
        height_median = np.median(line_heights)
        body_lines = [h for h in line_heights if h < height_median * 1.5]

        if len(body_lines) < 3:
            return 1.0, findings

        body_std = np.std(body_lines)
        body_mean = np.mean(body_lines)

        # Variation threshold — multi-script docs naturally have more variation
        cv = body_std / body_mean if body_mean > 0 else 0
        cv_threshold = 0.25  # Default: 25% (raised from 15% for robustness)

        if cv > cv_threshold:
            score = max(0.0, 1.0 - (cv - cv_threshold) * 5)
            findings.append(AgentFinding(
                description=(
                    f"Font size inconsistency: body text height varies by "
                    f"{cv*100:.1f}% (mean: {body_mean:.0f}px, std: {body_std:.1f}px)"
                ),
                severity=min(1.0, cv * 3),
                page=page_idx
            ))
        else:
            score = 1.0

        return score, findings

    def _analyze_text_block_regularity(
        self, word_data: list[dict], page_img: np.ndarray, page_idx: int
    ) -> tuple[float, list[AgentFinding]]:
        """
        Analyze the regularity of text blocks — forged regions may have
        different background intensity or edge characteristics.
        """
        findings = []
        if len(word_data) < 5:
            return 1.0, findings

        gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)

        # Sample background intensity around each word
        bg_intensities = []
        for word in word_data:
            x, y, w, h = word["x"], word["y"], word["w"], word["h"]
            # Pad slightly to get background
            pad = max(2, h // 4)
            y1 = max(0, y - pad)
            y2 = min(gray.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(gray.shape[1], x + w + pad)

            region = gray[y1:y2, x1:x2]
            if region.size > 0:
                # Background estimate: take the mode/high percentile
                bg = np.percentile(region, 90)
                bg_intensities.append((word, bg))

        if len(bg_intensities) < 5:
            return 1.0, findings

        bg_values = [b[1] for b in bg_intensities]
        bg_mean = np.mean(bg_values)
        bg_std = np.std(bg_values)

        # Find words with different background
        anomalous = []
        for word, bg in bg_intensities:
            if bg_std > 0 and abs(bg - bg_mean) > 3 * bg_std:
                anomalous.append((word, bg))

        if len(anomalous) > 2:
            ratio = len(anomalous) / len(bg_intensities)
            score = max(0.0, 1.0 - ratio * 8)
            findings.append(AgentFinding(
                description=(
                    f"Background inconsistency: {len(anomalous)} text regions "
                    f"have different background intensity (expected ~{bg_mean:.0f}, "
                    f"std: {bg_std:.1f})"
                ),
                severity=min(1.0, ratio * 5),
                page=page_idx
            ))
        else:
            score = 1.0

        return score, findings

    def _analyze_regional_confidence(
        self, word_data: list[dict], page_img: np.ndarray, page_idx: int
    ) -> tuple[float, list[AgentFinding]]:
        """
        Divide the page into a grid and compare OCR confidence per region vs
        the global average. Forged patches tend to have locally depressed or
        elevated confidence compared to their surroundings.
        """
        findings = []
        if len(word_data) < 10:
            return 1.0, findings

        h, w = page_img.shape[:2]
        grid_rows, grid_cols = 4, 3  # 12 regions

        global_confs = [wd["conf"] for wd in word_data]
        global_mean = float(np.mean(global_confs))
        global_std = float(np.std(global_confs)) if len(global_confs) > 1 else 1.0

        cell_h = h / grid_rows
        cell_w = w / grid_cols

        suspicious_regions = []

        for row in range(grid_rows):
            for col in range(grid_cols):
                x1 = col * cell_w
                y1 = row * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h

                region_words = [
                    wd for wd in word_data
                    if x1 <= wd["x"] + wd["w"] / 2 < x2
                    and y1 <= wd["y"] + wd["h"] / 2 < y2
                ]

                if len(region_words) < 3:
                    continue

                region_confs = [wd["conf"] for wd in region_words]
                region_mean = float(np.mean(region_confs))

                # Check if region deviates significantly from global
                deviation = abs(region_mean - global_mean)
                if global_std > 0 and deviation > 2.0 * global_std and deviation > 10:
                    suspicious_regions.append({
                        "row": row, "col": col,
                        "region_mean": region_mean,
                        "deviation": deviation,
                        "word_count": len(region_words),
                        "bbox": (int(x1), int(y1), int(cell_w), int(cell_h)),
                    })

        if not suspicious_regions:
            return 1.0, findings

        # Score based on how many regions are suspicious
        total_regions = sum(
            1 for row in range(grid_rows) for col in range(grid_cols)
            if sum(1 for wd in word_data
                   if col * cell_w <= wd["x"] + wd["w"] / 2 < (col + 1) * cell_w
                   and row * cell_h <= wd["y"] + wd["h"] / 2 < (row + 1) * cell_h) >= 3
        )

        anomaly_ratio = len(suspicious_regions) / max(1, total_regions)
        score = max(0.0, 1.0 - anomaly_ratio * 4)

        for region in suspicious_regions[:3]:
            direction = "lower" if region["region_mean"] < global_mean else "higher"
            findings.append(AgentFinding(
                description=(
                    f"Regional OCR confidence anomaly at grid ({region['row']},{region['col']}): "
                    f"local mean {region['region_mean']:.1f}% vs global {global_mean:.1f}% "
                    f"({direction} by {region['deviation']:.1f}%, "
                    f"{region['word_count']} words)"
                ),
                severity=min(1.0, region["deviation"] / 30.0),
                region=region["bbox"],
                page=page_idx
            ))

        return score, findings

    def _group_into_lines(self, word_data: list[dict]) -> list[list[dict]]:
        """Group words into approximate text lines based on vertical position."""
        if not word_data:
            return []

        sorted_words = sorted(word_data, key=lambda w: (w["y"], w["x"]))
        lines = []
        current_line = [sorted_words[0]]

        for word in sorted_words[1:]:
            prev = current_line[-1]
            # Same line if vertical overlap is significant
            prev_center_y = prev["y"] + prev["h"] / 2
            curr_center_y = word["y"] + word["h"] / 2
            threshold = max(prev["h"], word["h"]) * 0.5

            if abs(curr_center_y - prev_center_y) < threshold:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]

        if current_line:
            lines.append(current_line)

        return lines

    def _find_spatial_clusters(
        self, words: list[dict], distance_threshold: int = 100
    ) -> list[list[dict]]:
        """Cluster words by spatial proximity."""
        if not words:
            return []

        clusters = []
        used = set()

        for i, word in enumerate(words):
            if i in used:
                continue

            cluster = [word]
            used.add(i)

            for j, other in enumerate(words):
                if j in used:
                    continue
                cx1 = word["x"] + word["w"] / 2
                cy1 = word["y"] + word["h"] / 2
                cx2 = other["x"] + other["w"] / 2
                cy2 = other["y"] + other["h"] / 2
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                if dist < distance_threshold:
                    cluster.append(other)
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def _compute_reliability(self, document: Document) -> float:
        """Text agent reliability based on OCR quality."""
        if not document.ocr_confidence:
            return 0.0

        avg_conf = np.mean(document.ocr_confidence)

        # OCR confidence 0-100 → reliability 0-1
        # Below 50% confidence, text agent is unreliable
        return min(1.0, max(0.0, (avg_conf - 30) / 60))
