"""
Visual Tamper Detection Agent

Detection methods:
1. ManTraNet deep learning (pixel-level forgery localization) — primary detector
2. Multi-scale ELA (Q90/Q75/Q50) — catches JPEG recompression, splicing
3. Copy-move detection (ORB feature matching) — catches duplicated regions
4. JPEG quantization table analysis — detects double compression
5. Noise consistency analysis — catches spliced regions with different noise

This agent catches: splicing, copy-move, image manipulation, JPEG recompression
"""
import sys
import time
import io
from pathlib import Path
from typing import Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import cv2
import numpy as np
from PIL import Image
from loguru import logger
import torch

from certusdoc.agents.base import BaseAgent
from certusdoc.models import Document, AgentResult, AgentFinding
from certusdoc.utils.doc_detector import detect_document_type
from certusdoc.agents.print_scan_detector import PrintScanDetector

# ManTraNet model directory
_MANTRANET_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "mantranet"

# ManTraNet on CPU is slow. When enabled, we downscale to 512px max and
# enforce a timeout. Set MANTRANET_CPU_ENABLED = True to enable on CPU.
MANTRANET_CPU_ENABLED = False
MANTRANET_TIMEOUT_SECONDS = 30


class VisualTamperAgent(BaseAgent):
    """Detects visual tampering using ManTraNet deep learning + classical methods."""

    def __init__(self, trufor_model_path: Optional[str] = None):
        super().__init__(name="Visual Tamper Agent")
        self.trufor_model_path = trufor_model_path
        self.trufor_model = None
        self.mantranet_model = None
        self._mantranet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ManTraNet (primary deep learning detector)
        self._load_mantranet()

        if trufor_model_path:
            self._load_trufor(trufor_model_path)

        self.print_scan_detector = PrintScanDetector()

    def analyze(self, document: Document, doc_source_tool: str = None) -> AgentResult:
        start_time = time.time()
        all_findings = []
        page_scores = []
        combined_heatmap = None

        # Detect document type for adaptive thresholds
        doc_class = detect_document_type(document.ocr_text)

        # Run print-scan detection on first page
        print_scan_result = {"is_print_scan": False, "confidence": 0.0, "signals": []}
        if document.pages:
            first_page_gray = cv2.cvtColor(document.pages[0], cv2.COLOR_BGR2GRAY)
            try:
                print_scan_result = self.print_scan_detector.analyze(first_page_gray)
                if print_scan_result["is_print_scan"]:
                    all_findings.append(AgentFinding(
                        description=(
                            f"Print-scan attack suspected — digital artifact analysis "
                            f"may be unreliable. Confidence: {print_scan_result['confidence']:.2f}. "
                            f"Signals: {'; '.join(print_scan_result['signals'])}"
                        ),
                        severity=0.5,
                    ))
            except Exception as e:
                logger.debug(f"Print-scan detection failed: {e}")

        # Large PDF optimization: >3 pages → full analysis on pages 1, 2, last only
        total_pages = len(document.pages)
        if total_pages > 3:
            full_analysis_pages = {0, 1, total_pages - 1}
            all_findings.append(AgentFinding(
                description=(
                    f"Full forensic analysis performed on pages 1, 2, {total_pages}. "
                    f"Remaining pages received partial analysis (JPEG quantization only)."
                ),
                severity=0.0,
            ))
        else:
            full_analysis_pages = set(range(total_pages))

        for page_idx, page_img in enumerate(document.pages):
            sub_scores = {}
            is_full_page = page_idx in full_analysis_pages

            if not is_full_page:
                # Partial analysis: JPEG quantization only (fastest method)
                quant_score, quant_findings = self._analyze_jpeg_artifacts(
                    page_img, document, page_idx
                )
                all_findings.extend(quant_findings)
                sub_scores["jpeg_quant"] = quant_score
                page_scores.append(quant_score)
                continue

            # --- ManTraNet Deep Learning (primary detector) ---
            if self.mantranet_model is not None:
                is_cpu = self._mantranet_device.type == "cpu"
                if is_cpu and not MANTRANET_CPU_ENABLED:
                    logger.debug("ManTraNet skipped on CPU (MANTRANET_CPU_ENABLED=False)")
                else:
                    mtn_score, mtn_heatmap, mtn_findings = self._run_mantranet_with_timeout(
                        page_img, page_idx, timeout=MANTRANET_TIMEOUT_SECONDS
                    )
                    all_findings.extend(mtn_findings)
                    sub_scores["mantranet"] = mtn_score

            # --- Multi-Scale ELA Analysis ---
            ela_score, ela_heatmap, ela_findings = self._run_multiscale_ela(
                page_img, page_idx, doc_source_tool=doc_source_tool
            )
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

            # --- TruFor (if available, takes priority over ManTraNet) ---
            if self.trufor_model is not None:
                trufor_score, trufor_heatmap, trufor_findings = self._run_trufor(
                    page_img, page_idx
                )
                all_findings.extend(trufor_findings)
                sub_scores["trufor"] = trufor_score

            # === SCORING: severity-driven, not just weighted average ===
            worst_sub = min(sub_scores.values())

            # Weight allocation depends on which deep models are available
            has_mantranet = "mantranet" in sub_scores
            has_trufor = "trufor" in sub_scores

            if has_trufor:
                # TruFor takes priority if available
                weighted = (0.10 * ela_score + 0.10 * copymove_score +
                            0.05 * quant_score + 0.05 * noise_score +
                            0.20 * sub_scores.get("mantranet", ela_score) +
                            0.50 * sub_scores["trufor"])
            elif has_mantranet:
                # ManTraNet is primary deep learning signal (50% weight)
                weighted = (0.15 * ela_score + 0.15 * copymove_score +
                            0.05 * quant_score + 0.15 * noise_score +
                            0.50 * sub_scores["mantranet"])
            else:
                # Classical only fallback
                weighted = (0.30 * ela_score + 0.30 * copymove_score +
                            0.20 * quant_score + 0.20 * noise_score)

            # The page score is capped by (worst_sub + 0.20) — strong signal
            # from any method pulls the whole score down.
            ceiling_from_worst = worst_sub + 0.20
            page_score = min(weighted, ceiling_from_worst)
            page_score = max(0.0, min(1.0, page_score))

            page_scores.append(page_score)

            # Combine heatmaps: prefer ManTraNet heatmap if available
            best_heatmap = ela_heatmap
            if has_mantranet and mtn_heatmap is not None:
                if ela_heatmap is not None:
                    # Blend ManTraNet (60%) with ELA (40%)
                    mtn_resized = cv2.resize(
                        mtn_heatmap, (ela_heatmap.shape[1], ela_heatmap.shape[0])
                    )
                    best_heatmap = cv2.addWeighted(
                        ela_heatmap, 0.4, mtn_resized, 0.6, 0
                    )
                else:
                    best_heatmap = mtn_heatmap

            if combined_heatmap is None:
                combined_heatmap = best_heatmap
            else:
                if page_score < page_scores[-2] if len(page_scores) > 1 else True:
                    combined_heatmap = best_heatmap

        final_score = min(page_scores) if page_scores else 1.0
        reliability = self._compute_reliability(document)

        if print_scan_result["is_print_scan"] and print_scan_result["confidence"] > 0.7:
            reliability = max(0.1, reliability - 0.2)
            logger.info(f"  Visual reliability reduced by 0.2 due to print-scan detection")

        elapsed_ms = (time.time() - start_time) * 1000

        methods = ["multi_scale_ELA", "copy_move_ORB",
                    "JPEG_quantization", "noise_consistency"]
        if self.mantranet_model is not None:
            methods.append("ManTraNet")
        if self.trufor_model is not None:
            methods.append("TruFor")

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
                "methods_used": methods,
                "pages_analyzed": len(document.pages),
                "per_page_scores": page_scores,
            }
        )

    # ================================================================
    # ManTraNet Deep Learning
    # ================================================================

    def _load_mantranet(self) -> None:
        """Load ManTraNet pretrained model for pixel-level forgery detection."""
        weight_path = _MANTRANET_DIR / "MantraNetv4.pt"
        if not weight_path.exists():
            logger.warning(f"ManTraNet weights not found at {weight_path} — skipping")
            return

        try:
            # Add ManTraNet source dir to path for imports
            mantranet_src = str(_MANTRANET_DIR)
            if mantranet_src not in sys.path:
                sys.path.insert(0, mantranet_src)

            import os
            # ManTraNet's IMTFE __init__ loads 'IMTFEv4.pt' from CWD —
            # temporarily chdir so it finds the weights
            orig_cwd = os.getcwd()
            os.chdir(str(_MANTRANET_DIR))
            try:
                from mantranet import MantraNet
                model = MantraNet(device=self._mantranet_device)
                model.load_state_dict(
                    torch.load(str(weight_path), map_location=self._mantranet_device,
                               weights_only=False)
                )
            finally:
                os.chdir(orig_cwd)

            model.to(self._mantranet_device)
            model.eval()
            self.mantranet_model = model
            logger.info(f"ManTraNet loaded on {self._mantranet_device}")
        except Exception as e:
            logger.error(f"Failed to load ManTraNet: {e}")
            self.mantranet_model = None

    def _run_mantranet_with_timeout(
        self, image: np.ndarray, page_idx: int, timeout: int = 30
    ) -> tuple[float, Optional[np.ndarray], list[AgentFinding]]:
        """Run ManTraNet in a thread with timeout. On CPU, downscales to 512px."""
        is_cpu = self._mantranet_device.type == "cpu"
        if is_cpu:
            h, w = image.shape[:2]
            if max(h, w) > 512:
                scale = 512 / max(h, w)
                image = cv2.resize(image, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
                logger.debug(f"ManTraNet CPU: downscaled to {image.shape[1]}x{image.shape[0]}")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_mantranet, image, page_idx)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                logger.warning(f"ManTraNet timed out after {timeout}s on page {page_idx}")
                future.cancel()
                return 1.0, None, [AgentFinding(
                    description=f"ManTraNet analysis timed out after {timeout}s",
                    severity=0.0, page=page_idx,
                )]
            except Exception as e:
                logger.error(f"ManTraNet thread error: {e}")
                return 1.0, None, []

    def _run_mantranet(
        self, image: np.ndarray, page_idx: int
    ) -> tuple[float, Optional[np.ndarray], list[AgentFinding]]:
        """
        Run ManTraNet inference on a page image.

        ManTraNet outputs a pixel-level forgery probability map (0=authentic, 1=forged).
        We convert this to an integrity score (1=authentic, 0=forged) and extract findings.
        """
        findings = []

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # ManTraNet works best at reasonable resolution; resize if too large.
            # 768px gives a good accuracy/speed trade-off (~44% fewer pixels
            # than 1024px, roughly 2× faster inference with similar detection).
            h, w = rgb.shape[:2]
            max_dim = 768
            scale = 1.0
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                rgb = cv2.resize(rgb, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)

            # Prepare tensor: (1, C, H, W) float32
            tensor = torch.from_numpy(rgb).float()
            tensor = tensor.unsqueeze(0)  # (1, H, W, C)
            tensor = tensor.permute(0, 3, 1, 2)  # (1, C, H, W)
            tensor = tensor.to(self._mantranet_device)

            with torch.no_grad():
                forgery_map = self.mantranet_model(tensor)  # (1, 1, H, W)

            # Convert to numpy: values in [0, 1] where higher = more forged
            fmap = forgery_map[0, 0].cpu().numpy()

            # Compute score from forgery map statistics
            mean_forgery = float(np.mean(fmap))
            max_forgery = float(np.max(fmap))
            # Ratio of pixels with forgery probability > 0.3
            forged_ratio = float(np.sum(fmap > 0.3)) / fmap.size
            # Ratio of pixels with strong forgery signal > 0.5
            strong_ratio = float(np.sum(fmap > 0.5)) / fmap.size

            # Convert forgery probability to integrity score
            # Higher mean_forgery and forged_ratio → lower integrity
            #
            # Coloured government IDs (Aadhaar, PAN, DL) have vibrant sections
            # (orange/teal headers, photos, QR codes) that produce localised
            # hotspots in ManTraNet's forgery map even on authentic documents.
            # These appear as high strong_ratio but LOW mean_forgery (<0.12).
            # Real tampered images typically show high mean_forgery (>0.15)
            # because the manipulation affects broad contiguous regions.
            if strong_ratio > 0.05 and (strong_ratio > 0.12 or mean_forgery > 0.15):
                # Widespread strong signal → genuine forgery indicator
                score = max(0.05, 0.25 - strong_ratio * 2)
                findings.append(AgentFinding(
                    description=(
                        f"ManTraNet: {strong_ratio*100:.1f}% of image has strong "
                        f"forgery signal (>0.5). Mean forgery: {mean_forgery:.3f}, "
                        f"max: {max_forgery:.3f}. Deep learning model detects "
                        f"pixel-level manipulation traces."
                    ),
                    severity=min(1.0, 0.7 + strong_ratio * 2),
                    page=page_idx
                ))
            elif strong_ratio > 0.05:
                # Elevated strong_ratio but low mean → scattered hotspots in
                # coloured document regions (headers, photos, QR), not splicing.
                score = max(0.35, 0.65 - strong_ratio * 2)
                findings.append(AgentFinding(
                    description=(
                        f"ManTraNet: {strong_ratio*100:.1f}% of image has localised "
                        f"strong signal (>0.5) but mean forgery is low ({mean_forgery:.3f}). "
                        f"Likely coloured document content (header/photo/QR), not tampering."
                    ),
                    severity=min(0.5, 0.3 + strong_ratio * 2),
                    page=page_idx
                ))
            elif forged_ratio > 0.05:
                # Moderate forgery signal
                score = max(0.15, 0.45 - forged_ratio * 3)
                findings.append(AgentFinding(
                    description=(
                        f"ManTraNet: {forged_ratio*100:.1f}% of image flagged "
                        f"(>0.3 threshold). Mean: {mean_forgery:.3f}, "
                        f"max: {max_forgery:.3f}. Possible manipulation detected."
                    ),
                    severity=min(0.9, 0.5 + forged_ratio * 3),
                    page=page_idx
                ))
            elif forged_ratio > 0.035:
                # Moderate-mild signal — above natural noise but below clear forgery.
                # This range (3.5-5%) catches subtle edits with localized anomalies.
                score = max(0.30, 0.55 - forged_ratio * 4)
                findings.append(AgentFinding(
                    description=(
                        f"ManTraNet: elevated anomaly — {forged_ratio*100:.1f}% flagged "
                        f"(>0.3 threshold). Mean: {mean_forgery:.3f}, max: {max_forgery:.3f}. "
                        f"Possible localized manipulation."
                    ),
                    severity=min(0.6, forged_ratio * 6),
                    page=page_idx
                ))
            elif forged_ratio > 0.02:
                # Mild signal — could be compression artifacts
                score = max(0.45, 0.70 - forged_ratio * 5)
                findings.append(AgentFinding(
                    description=(
                        f"ManTraNet: minor anomaly — {forged_ratio*100:.1f}% flagged. "
                        f"Mean: {mean_forgery:.3f}, max: {max_forgery:.3f}."
                    ),
                    severity=min(0.5, forged_ratio * 5),
                    page=page_idx
                ))
            elif mean_forgery > 0.15:
                # Elevated mean but diffuse — often JPEG artifacts
                score = max(0.55, 0.85 - mean_forgery * 2)
            else:
                # Clean image
                score = min(1.0, 0.90 + (1.0 - mean_forgery) * 0.10)

            # Generate heatmap from forgery map
            fmap_uint8 = np.clip(fmap * 255, 0, 255).astype(np.uint8)
            # Resize back to original image size if we downscaled
            if scale != 1.0:
                fmap_uint8 = cv2.resize(
                    fmap_uint8, (w, h), interpolation=cv2.INTER_LINEAR
                )
            heatmap = cv2.applyColorMap(fmap_uint8, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

            logger.info(
                f"  ManTraNet p{page_idx}: score={score:.3f}, "
                f"mean_forgery={mean_forgery:.3f}, max={max_forgery:.3f}, "
                f"forged_ratio={forged_ratio*100:.1f}%, "
                f"strong_ratio={strong_ratio*100:.1f}%"
            )

            return score, heatmap, findings

        except Exception as e:
            logger.error(f"ManTraNet inference failed: {e}")
            return 1.0, None, []

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
        self, image: np.ndarray, page_idx: int, doc_source_tool: str = None
    ) -> tuple[float, np.ndarray, list[AgentFinding]]:
        """
        Multi-scale ELA at Q90, Q75, Q50.
        Anomalies persisting across all 3 scales = real tampering (hard penalty).
        Anomaly at only 1 scale = possible false positive (soft penalty).
        """
        findings = []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Downsample for ELA — full 300 DPI images (2400×3100+) are slow to
        # process three times. 1024px preserves enough detail for detecting
        # recompression artefacts while cutting compute by ~6×.
        ela_max_dim = 1024
        h_ela, w_ela = rgb.shape[:2]
        if max(h_ela, w_ela) > ela_max_dim:
            ela_scale = ela_max_dim / max(h_ela, w_ela)
            rgb = cv2.resize(rgb, None, fx=ela_scale, fy=ela_scale,
                             interpolation=cv2.INTER_AREA)

        qualities = [90, 75, 50]
        ela_maps = []
        anomaly_masks = []
        anomaly_ratios = []

        for q in qualities:
            ela_map, mean_ela, std_ela, max_ela = self._run_ela_single(rgb, q)
            ela_maps.append(ela_map)

            threshold = mean_ela + 2.0 * std_ela
            # Raise ELA threshold for known PDF generators that produce
            # JPEG artifacts mimicking tampering (wkhtmltopdf, fpdf, reportlab)
            if doc_source_tool:
                tool_lower = doc_source_tool.lower()
                if any(t in tool_lower for t in ("wkhtmltopdf", "fpdf", "reportlab",
                                                   "weasyprint", "prince", "puppeteer")):
                    threshold *= 1.5
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
            # Check if the signal is uniform across all quality levels (compression
            # artefact) vs concentrated (localised tampering).
            # Uniform JPEG re-encoding produces similar anomaly ratios at every
            # quality level; splicing produces uneven ratios because the spliced
            # region was saved at a different quality than the background.
            ratio_cv = (float(np.std(anomaly_ratios)) /
                        (float(np.mean(anomaly_ratios)) + 1e-6))
            is_uniform_compression = ratio_cv < 0.20 and persistent_3_ratio < 0.015

            if is_uniform_compression:
                # Uniform signal = whole-image JPEG recompression (e.g. scan, WhatsApp).
                # This is expected for authentic printed/scanned govt IDs.
                score = max(0.55, 0.85 - persistent_2_ratio * 3)
                findings.append(AgentFinding(
                    description=(
                        f"Multi-scale ELA: {persistent_2_ratio*100:.1f}% of image flagged "
                        f"across 2+ quality levels. Per-scale anomaly ratios: "
                        f"{[f'{r*100:.1f}%' for r in anomaly_ratios]}. "
                        f"Uniform ratios (CV={ratio_cv:.2f}) indicate whole-image JPEG "
                        f"recompression, not localised tampering."
                    ),
                    severity=min(0.35, persistent_2_ratio * 4),
                    page=page_idx
                ))
            else:
                # Non-uniform signal = localised recompression → likely tampering
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
        elif max(anomaly_ratios) > 0.06:
            # Single-scale anomaly — possible but less certain
            worst_q_idx = anomaly_ratios.index(max(anomaly_ratios))
            worst_ratio = anomaly_ratios[worst_q_idx]
            score = max(0.2, 0.6 - worst_ratio * 4)
            findings.append(AgentFinding(
                description=(
                    f"ELA anomaly at Q{qualities[worst_q_idx]}: "
                    f"{worst_ratio*100:.1f}% of image affected. "
                    f"Per-scale: {[f'{r*100:.1f}%' for r in anomaly_ratios]}"
                ),
                severity=min(1.0, worst_ratio * 5),
                page=page_idx
            ))
        elif max(anomaly_ratios) > 0.03:
            score = max(0.45, 0.80 - max(anomaly_ratios) * 4)
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
                # Use 2.5 sigma for outlier detection — 2.0 is too sensitive
                # for documents with high contrast (text on blank background)
                outlier_blocks = int(np.sum(bm_arr > bm_mean + 2.5 * bm_std))
                # Also require minimum absolute ELA level — blocks with very low
                # ELA mean the image is clean, high CV is just noise
                min_ela_for_concern = 3.0
                if bm_mean < min_ela_for_concern:
                    outlier_blocks = 0  # Low ELA overall = clean image
                if cv_ela > 0.50 and outlier_blocks >= 2:
                    # High block variance with multiple outliers = localized tampering
                    block_penalty = min(score, max(0.20, 0.45 - cv_ela * 0.30))
                    if block_penalty < score:
                        score = block_penalty
                        findings.append(AgentFinding(
                            description=(
                                f"ELA block variance: CV={cv_ela:.2f}, "
                                f"{outlier_blocks} outlier blocks of "
                                f"{len(block_means_q90)}. Localized ELA anomaly "
                                f"suggests region-level tampering."
                            ),
                            severity=min(0.9, cv_ela),
                            page=page_idx
                        ))
                elif cv_ela > 0.35 and outlier_blocks >= 1:
                    block_penalty = min(score, max(0.40, 0.60 - cv_ela * 0.25))
                    if block_penalty < score:
                        score = block_penalty
                        findings.append(AgentFinding(
                            description=(
                                f"Moderate ELA block variance: CV={cv_ela:.2f}, "
                                f"{outlier_blocks} outlier blocks."
                            ),
                            severity=min(0.6, cv_ela),
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

        if total_suspicious < 4:
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

        # Only meaningful for JPEG images — PNGs don't have block artifacts
        fmt = document.original_format.lower()
        if fmt in ("png", "bmp", "tiff"):
            return 1.0, findings

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
            min_ratio = min(ratio_h, ratio_v)
            max_ratio = max(ratio_h, ratio_v)

            # Double compression: both axes show low ratios
            # OR one axis is very low AND the other isn't wildly high (< 2.0)
            both_low = ratio_h < 0.7 and ratio_v < 0.7
            one_very_low = min_ratio < 0.5 and max_ratio < 2.0

            if both_low and min_ratio < 0.4:
                score = 0.3
                findings.append(AgentFinding(
                    description=(
                        f"Strong double compression signal: JPEG block boundary "
                        f"ratios H={ratio_h:.2f}, V={ratio_v:.2f} (expected ~1.0)"
                    ),
                    severity=0.8,
                    page=page_idx
                ))
            elif one_very_low or (both_low and min_ratio < 0.7):
                score = 0.50
                findings.append(AgentFinding(
                    description=(
                        f"Possible double compression: block boundary "
                        f"ratios H={ratio_h:.2f}, V={ratio_v:.2f}"
                    ),
                    severity=0.5,
                    page=page_idx
                ))
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
        """
        Check if noise levels are consistent across the image using two methods:
        1. Global Laplacian noise outlier detection (original)
        2. Local median-filter residual analysis (catches subtle edits where a
           region's noise texture differs from its surroundings)
        """
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

        noise_values = np.array([n[2] for n in noise_levels])

        # Use median + MAD instead of mean + std to avoid being skewed by
        # blank/white regions. On documents with large blank areas, mean/std
        # are dominated by zero-noise blocks, making text blocks look anomalous.
        median_noise = float(np.median(noise_values))
        mad = float(np.median(np.abs(noise_values - median_noise)))

        # When median noise is very low (< 1.0), the image is mostly blank.
        # On mostly-blank images, any content blocks (text, logos) will appear
        # as noise outliers relative to the blank baseline — but this is normal
        # document structure, not forgery. Skip outlier detection in this case.
        if median_noise < 1.0:
            anomalous_blocks = []
        elif mad < 0.5:
            # Very uniform noise — use std-based fallback with lenient threshold
            mean_noise = float(np.mean(noise_values))
            std_noise = float(np.std(noise_values))
            sigma_thresh = 3.5 if is_structured else 3.0
            anomalous_blocks = []
            for x, y, noise in noise_levels:
                if std_noise > 0 and abs(noise - mean_noise) > sigma_thresh * std_noise:
                    anomalous_blocks.append((x, y, noise))
        else:
            # MAD-based outlier detection (robust to blank/content dichotomy)
            sigma_thresh = 4.0 if is_structured else 3.5
            mad_std = mad * 1.4826  # MAD to std conversion
            anomalous_blocks = []
            for x, y, noise in noise_levels:
                if abs(noise - median_noise) > sigma_thresh * mad_std:
                    anomalous_blocks.append((x, y, noise))

        anomaly_ratio = len(anomalous_blocks) / len(noise_levels)

        if anomaly_ratio > 0.05:
            score = max(0.1, 0.5 - anomaly_ratio * 5)
            findings.append(AgentFinding(
                description=(
                    f"Noise inconsistency: {len(anomalous_blocks)}/{len(noise_levels)} "
                    f"blocks anomalous ({anomaly_ratio*100:.1f}%). "
                    f"Mean noise: {mean_noise:.1f}, std: {std_noise:.1f}"
                ),
                severity=min(1.0, anomaly_ratio * 6),
                page=page_idx
            ))
        elif anomaly_ratio > 0.015:
            score = max(0.35, 0.7 - anomaly_ratio * 6)
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

        # === Local median-filter residual analysis ===
        # Estimates per-block noise as std(pixel - median_filtered), then
        # checks for blocks whose noise texture differs from local neighbors.
        # Catches subtle edits where forged region noise differs from scan noise.
        local_score = self._analyze_local_noise_texture(gray, page_idx, findings)
        if local_score < score:
            score = local_score

        return score, findings

    def _analyze_local_noise_texture(
        self, gray: np.ndarray, page_idx: int,
        findings: list[AgentFinding],
    ) -> float:
        """
        Compare each block's noise texture to its local neighborhood.
        Edited regions have different noise from surrounding original content.
        Uses median-filter residual as a robust noise estimator.
        """
        h, w = gray.shape
        bs = 32
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

        rows = (h - bs) // bs
        cols = (w - bs) // bs
        if rows < 4 or cols < 4:
            return 1.0

        # Build noise grid and edge density grid
        noise_grid = np.zeros((rows, cols), dtype=np.float64)
        edge_grid = np.zeros((rows, cols), dtype=np.float64)
        for r in range(rows):
            for c in range(cols):
                y, x = r * bs, c * bs
                block = gray_u8[y:y+bs, x:x+bs]
                med = cv2.medianBlur(block, 3)
                noise_grid[r, c] = float(np.std(block.astype(np.float64) - med.astype(np.float64)))
                edge_grid[r, c] = float(np.mean(cv2.Canny(block, 50, 150) > 0))

        # Only analyze content blocks (edge density > 1%)
        content_mask = edge_grid > 0.01
        content_count = int(np.sum(content_mask))
        # Need sufficient content blocks for meaningful analysis.
        # Also skip if content is very sparse (< 5% of image) — the noise
        # comparison is unreliable with few content blocks (e.g., synthetic
        # images with just a few lines of text on white background).
        total_blocks = rows * cols
        if content_count < 16 or content_count < total_blocks * 0.05:
            return 1.0

        # For each content block, compare its noise to its local 5x5 neighborhood
        pad = 2
        local_anomaly_count = 0
        local_anomalies = []

        for r in range(rows):
            for c in range(cols):
                if not content_mask[r, c]:
                    continue
                # Get neighborhood content blocks
                r1, r2 = max(0, r - pad), min(rows, r + pad + 1)
                c1, c2 = max(0, c - pad), min(cols, c + pad + 1)
                nbr_mask = content_mask[r1:r2, c1:c2]
                if np.sum(nbr_mask) < 3:
                    continue
                nbr_noise = noise_grid[r1:r2, c1:c2][nbr_mask]
                nbr_median = float(np.median(nbr_noise))
                nbr_mad = float(np.median(np.abs(nbr_noise - nbr_median)))
                if nbr_mad < 0.5:
                    continue  # Very uniform neighborhood, skip

                # Check if this block deviates from local neighborhood
                deviation = abs(noise_grid[r, c] - nbr_median)
                threshold = max(3.0 * nbr_mad * 1.4826, 2.0)  # MAD-based + min
                if deviation > threshold:
                    local_anomaly_count += 1
                    local_anomalies.append((r * bs, c * bs, float(noise_grid[r, c]),
                                            nbr_median, deviation))

        local_ratio = local_anomaly_count / content_count

        # Need minimum absolute count of anomalous blocks (not just ratio)
        # to avoid false positives on sparse images with few content blocks
        if local_ratio > 0.08 and local_anomaly_count >= 5:
            score = max(0.15, 0.45 - local_ratio * 3)
            findings.append(AgentFinding(
                description=(
                    f"Local noise texture anomaly: {local_anomaly_count}/{content_count} "
                    f"content blocks differ from their local neighborhood "
                    f"({local_ratio*100:.1f}%). Indicates region-level editing."
                ),
                severity=min(0.9, local_ratio * 5),
                page=page_idx,
            ))
        elif local_ratio > 0.04 and local_anomaly_count >= 3:
            score = max(0.45, 0.65 - local_ratio * 3)
            findings.append(AgentFinding(
                description=(
                    f"Moderate noise texture variation: {local_anomaly_count}/{content_count} "
                    f"blocks with local noise deviation ({local_ratio*100:.1f}%)."
                ),
                severity=min(0.6, local_ratio * 4),
                page=page_idx,
            ))
        else:
            score = 1.0

        return score

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
        logger.warning("TruFor not yet integrated — using ManTraNet as deep learning backbone")

    def _run_trufor(
        self, image: np.ndarray, page_idx: int
    ) -> tuple[float, Optional[np.ndarray], list[AgentFinding]]:
        """Run TruFor inference. Placeholder until model is integrated."""
        return 1.0, None, []
