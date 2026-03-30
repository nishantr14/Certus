"""
Print-Scan Detection Module

Detects documents that have been printed on paper and then re-scanned.
This is a common attack to bypass digital forensic analysis because:
  - ELA (Error Level Analysis) becomes unreliable after scanning
  - Metadata is stripped or replaced by the scanner
  - Copy-move detection is defeated by the rasterisation process

Detection relies on four physical artefacts introduced by the print-scan cycle:
  1. Halftone dot patterns  — laser/inkjet printers render continuous tones as
     periodic dot screens at 45°/75°/90°/105° angles (CMYK screen angles).
  2. Ink bleed / spread     — ink soaks into paper fibres, widening edges and
     softening high-frequency transitions.
  3. Scan line banding      — CCD/CIS sensor irregularities produce faint
     horizontal intensity bands along the scan direction.
  4. Moiré interference     — when the scanner sampling grid beats against the
     printer halftone grid, low-frequency moiré fringes appear.
"""
import cv2
import numpy as np
from loguru import logger


class PrintScanDetector:
    """
    Utility class (not a BaseAgent) that scores a single grayscale image
    for evidence of having been printed and re-scanned.

    Usage
    -----
    detector = PrintScanDetector()
    result = detector.analyze(gray_image)   # gray_image: np.ndarray (H, W) uint8
    # result == {"is_print_scan": bool, "confidence": float, "signals": [str, ...]}
    """

    # Weighted combination of the four sub-detectors
    _WEIGHTS = {
        "halftone":  0.40,
        "ink_bleed": 0.20,
        "scan_lines": 0.20,
        "moire":     0.20,
    }

    # Decision threshold
    _THRESHOLD = 0.7

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def analyze(self, image: np.ndarray) -> dict:
        """
        Analyse a grayscale (or colour) image for print-scan artefacts.

        Parameters
        ----------
        image : np.ndarray
            Either a grayscale (H, W) or BGR colour (H, W, 3) image,
            dtype uint8.

        Returns
        -------
        dict with keys:
            "is_print_scan" : bool   — True when confidence > 0.70
            "confidence"    : float  — weighted combination in [0.0, 1.0]
            "signals"       : list[str] — human-readable signal descriptions
        """
        # Ensure grayscale
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Sub-detector results: (score, signal_str)
        results = {}
        try:
            results["halftone"] = self._detect_halftone(gray)
        except Exception as exc:
            logger.debug(f"PrintScanDetector._detect_halftone failed: {exc}")
            results["halftone"] = (0.0, "")

        try:
            results["ink_bleed"] = self._detect_ink_bleed(gray)
        except Exception as exc:
            logger.debug(f"PrintScanDetector._detect_ink_bleed failed: {exc}")
            results["ink_bleed"] = (0.0, "")

        try:
            results["scan_lines"] = self._detect_scan_lines(gray)
        except Exception as exc:
            logger.debug(f"PrintScanDetector._detect_scan_lines failed: {exc}")
            results["scan_lines"] = (0.0, "")

        try:
            results["moire"] = self._detect_moire(gray)
        except Exception as exc:
            logger.debug(f"PrintScanDetector._detect_moire failed: {exc}")
            results["moire"] = (0.0, "")

        # Weighted confidence
        confidence = sum(
            self._WEIGHTS[k] * results[k][0]
            for k in self._WEIGHTS
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # Collect non-empty signal strings from detectors that fired
        signals = [
            results[k][1]
            for k in self._WEIGHTS
            if results[k][0] > 0.0 and results[k][1]
        ]

        is_print_scan = confidence > self._THRESHOLD

        logger.debug(
            f"PrintScanDetector: confidence={confidence:.3f} "
            f"is_print_scan={is_print_scan} signals={signals}"
        )

        return {
            "is_print_scan": is_print_scan,
            "confidence": confidence,
            "signals": signals,
        }

    # ------------------------------------------------------------------ #
    # Sub-detectors
    # ------------------------------------------------------------------ #

    def _detect_halftone(self, gray: np.ndarray) -> tuple:
        """
        Detect halftone dot patterns using FFT frequency-domain analysis.

        Forensic rationale
        ------------------
        Inkjet and laser printers render continuous-tone images as grids of
        dots screened at standard angles (typically 45° for black, with CMYK
        at 75°/90°/105°).  These periodic structures produce sharp spectral
        peaks in the FFT magnitude spectrum at spatial frequencies
        corresponding to the dot pitch (commonly 60–120 lpi → peaks at
        specific radial distances from DC).  A genuine digital document has
        no such periodic physical structure.

        Parameters
        ----------
        gray : np.ndarray  (H, W) uint8

        Returns
        -------
        (score: float, signal: str)
            score is 0.0 if image is too small or featureless.
        """
        h, w = gray.shape
        if h < 64 or w < 64:
            return 0.0, ""

        # Work on a centre crop to avoid border effects
        crop_h, crop_w = min(h, 512), min(w, 512)
        y0, x0 = (h - crop_h) // 2, (w - crop_w) // 2
        crop = gray[y0:y0 + crop_h, x0:x0 + crop_w].astype(np.float32)

        # Zero-mean + window to reduce spectral leakage
        crop -= crop.mean()
        window = np.outer(
            np.hanning(crop_h).astype(np.float32),
            np.hanning(crop_w).astype(np.float32),
        )
        crop *= window

        # FFT magnitude (log-scaled, DC suppressed)
        fft = np.fft.fft2(crop)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift))

        cy, cx = crop_h // 2, crop_w // 2

        # Suppress DC and very low frequencies (inner 5% radius)
        dc_radius = int(min(crop_h, crop_w) * 0.05)
        yy, xx = np.ogrid[:crop_h, :crop_w]
        dist_from_dc = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        magnitude[dist_from_dc < dc_radius] = 0.0

        # Look for spectral peaks in the mid-frequency band typical of
        # halftone screens (radial distance 8–25% of Nyquist)
        nyquist = min(crop_h, crop_w) / 2.0
        inner = nyquist * 0.08
        outer = nyquist * 0.35
        band_mask = (dist_from_dc >= inner) & (dist_from_dc <= outer)
        band = magnitude[band_mask]

        if band.size == 0 or band.std() < 1e-6:
            return 0.0, ""

        # Normalise so the metric is independent of overall image brightness
        band_norm = (band - band.mean()) / (band.std() + 1e-8)

        # Count "sharp peaks" well above the band mean (z-score > 4)
        peak_count = int((band_norm > 4.0).sum())
        total_band = band.size

        # Score: fraction of peak pixels, scaled to expected range
        raw_score = peak_count / max(total_band * 0.001, 1.0)
        score = float(np.clip(raw_score / 10.0, 0.0, 1.0))

        if score > 0.1:
            signal = (
                f"Halftone dot screen detected in FFT "
                f"(peak_count={peak_count}, score={score:.2f})"
            )
        else:
            signal = ""

        return score, signal

    def _detect_ink_bleed(self, gray: np.ndarray) -> tuple:
        """
        Detect ink bleed / spread by comparing edge sharpness to background.

        Forensic rationale
        ------------------
        When a document is printed, ink soaks into paper fibres and spreads
        laterally.  On re-scanning the text edges appear blurry compared to
        a genuine digital document rendered at screen resolution.  We measure
        this as the ratio of Laplacian variance at detected edges vs. the
        global Laplacian variance: a low ratio indicates soft edges consistent
        with ink bleed, while a high ratio (sharp edges) indicates a native
        digital document.

        Parameters
        ----------
        gray : np.ndarray  (H, W) uint8

        Returns
        -------
        (score: float, signal: str)
        """
        h, w = gray.shape
        if h < 32 or w < 32:
            return 0.0, ""

        # Laplacian gives a measure of local edge strength
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_abs = np.abs(lap)

        global_var = float(lap_abs.var())
        if global_var < 1e-6:
            # Blank or nearly blank image — no edges to analyse
            return 0.0, ""

        # Detect strong edge pixels via Canny
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = lap_abs[edges > 0]

        if edge_pixels.size < 10:
            return 0.0, ""

        edge_var = float(edge_pixels.var())

        # Ratio: edge sharpness relative to global variance
        # A low ratio means edges are not much sharper than background → bleed
        ratio = edge_var / (global_var + 1e-8)

        # In a clean digital document, edges dominate the Laplacian variance
        # so ratio >> 1.  After print-scan, ink bleed smooths edges → ratio
        # approaches 1 or falls below it.  We score inversely.
        # Empirically: ratio < 2 → strong signal, ratio > 8 → clean.
        score = float(np.clip(1.0 - (ratio - 1.0) / 7.0, 0.0, 1.0))

        if score > 0.3:
            signal = (
                f"Ink bleed / edge softening detected "
                f"(edge/global Laplacian ratio={ratio:.2f}, score={score:.2f})"
            )
        else:
            signal = ""

        return score, signal

    def _detect_scan_lines(self, gray: np.ndarray) -> tuple:
        """
        Detect horizontal scan-line banding from CCD/CIS sensor irregularities.

        Forensic rationale
        ------------------
        Flatbed scanners use a linear CCD or CIS sensor array that moves down
        the page.  Variations in individual pixel sensitivity or illumination
        produce faint horizontal bands — rows of slightly higher or lower
        intensity than their neighbours.  We detect this by computing the
        row-wise mean intensity and measuring the variance of those row means:
        a high row-mean variance relative to the overall image variance is a
        strong indicator of scanner banding.

        Parameters
        ----------
        gray : np.ndarray  (H, W) uint8

        Returns
        -------
        (score: float, signal: str)
        """
        h, w = gray.shape
        if h < 64 or w < 64:
            return 0.0, ""

        img_float = gray.astype(np.float32)
        overall_var = float(img_float.var())
        if overall_var < 1e-6:
            return 0.0, ""

        # Row-wise mean intensity profile
        row_means = img_float.mean(axis=1)  # shape (H,)

        # Remove low-frequency global gradient (detrend linearly)
        x = np.arange(h, dtype=np.float32)
        coeffs = np.polyfit(x, row_means, 1)
        trend = np.polyval(coeffs, x)
        detrended = row_means - trend

        row_var = float(detrended.var())

        # Ratio of row-mean variance to overall pixel variance
        # Scan lines raise this ratio; natural content has low row_var/overall_var
        ratio = row_var / (overall_var + 1e-8)

        # Threshold: ratio > 0.02 begins to suggest banding
        score = float(np.clip(ratio / 0.10, 0.0, 1.0))

        if score > 0.2:
            signal = (
                f"Horizontal scan-line banding detected "
                f"(row-mean variance ratio={ratio:.4f}, score={score:.2f})"
            )
        else:
            signal = ""

        return score, signal

    def _detect_moire(self, gray: np.ndarray) -> tuple:
        """
        Detect moiré interference patterns using bandpass FFT filtering.

        Forensic rationale
        ------------------
        When a scanner's sampling grid (typically 300–600 dpi) interacts with
        a printer's halftone grid (typically 1200+ dpi rendered dots), the
        two periodic structures beat against each other and produce low-
        frequency moiré fringes that are visible as wavy bands across the
        document.  We detect this by bandpass-filtering the FFT magnitude to
        the low-mid spatial frequencies (3–15% of Nyquist) and measuring
        whether there are anomalously high-energy periodic components in that
        band that would not normally be present in a native digital document.

        Parameters
        ----------
        gray : np.ndarray  (H, W) uint8

        Returns
        -------
        (score: float, signal: str)
        """
        h, w = gray.shape
        if h < 64 or w < 64:
            return 0.0, ""

        crop_h, crop_w = min(h, 512), min(w, 512)
        y0, x0 = (h - crop_h) // 2, (w - crop_w) // 2
        crop = gray[y0:y0 + crop_h, x0:x0 + crop_w].astype(np.float32)
        crop -= crop.mean()

        fft = np.fft.fft2(crop)
        fft_shift = np.fft.fftshift(fft)
        power = np.abs(fft_shift) ** 2  # power spectrum

        cy, cx = crop_h // 2, crop_w // 2
        yy, xx = np.ogrid[:crop_h, :crop_w]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        nyquist = min(crop_h, crop_w) / 2.0

        # Bandpass: moiré lives at low-mid frequencies (3–15% Nyquist)
        bp_mask = (dist >= nyquist * 0.03) & (dist <= nyquist * 0.15)
        # Reference: the full mid band (3–50% Nyquist)
        full_mask = (dist >= nyquist * 0.03) & (dist <= nyquist * 0.50)

        bp_energy = float(power[bp_mask].sum())
        full_energy = float(power[full_mask].sum())

        if full_energy < 1e-6:
            return 0.0, ""

        # Fraction of total mid-band energy concentrated in the low-mid band
        energy_fraction = bp_energy / (full_energy + 1e-8)

        # In a native digital document the energy is spread across all
        # frequencies; moiré concentrates energy at its characteristic
        # frequency.  A fraction above ~0.40 is suspicious.
        score = float(np.clip((energy_fraction - 0.40) / 0.40, 0.0, 1.0))

        if score > 0.1:
            signal = (
                f"Moiré interference pattern detected "
                f"(low-mid band energy fraction={energy_fraction:.3f}, "
                f"score={score:.2f})"
            )
        else:
            signal = ""

        return score, signal
