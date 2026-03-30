"""
Print-Scan Attack Detector

Forensic rationale: A forger can bypass ELA and metadata checks by:
  1. Editing a document digitally (Photoshop etc.)
  2. Printing the edited document
  3. Scanning the printout
  4. Submitting the scan

The print→scan cycle:
  - Erases ALL digital compression history (ELA becomes useless)
  - Produces scanner EXIF (looks legitimate)
  - Introduces characteristic physical artifacts detectable in frequency domain

This module detects those physical print-scan artifacts using:
  a. Halftone pattern detection — FFT reveals dot patterns at 45°/75°/90°/105°
  b. Ink bleed detection — text edges blur in a characteristic way on paper
  c. Scan line artifacts — horizontal banding from CCD/CIS scanner sensors
  d. Moiré pattern detection — interference patterns from halftone + scanner aliasing

A confidence > 0.7 means "this is very likely a physical scan, not a digital original."
This is NOT a forgery indicator by itself — it means ELA analysis is unreliable.
"""
import numpy as np
import cv2
from loguru import logger


def analyze(image: np.ndarray) -> dict:
    """
    Analyze an image for print-scan artifacts.

    Args:
        image: BGR image as numpy array (from OpenCV).

    Returns:
        dict with keys:
            is_print_scan (bool): True if strong evidence of physical print+scan.
            confidence (float): 0.0-1.0 confidence in print-scan classification.
            signals (list[str]): Human-readable list of detected signals.
            details (dict): Per-method scores and parameters.
    """
    if image is None or image.size == 0:
        return {"is_print_scan": False, "confidence": 0.0, "signals": [], "details": {}}

    try:
        # Convert to grayscale for frequency-domain analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = gray.astype(np.float32)

        signals = []
        scores = {}

        # --- Method A: Halftone Pattern Detection ---
        halftone_score, halftone_signals = _detect_halftone_patterns(gray)
        scores["halftone"] = halftone_score
        signals.extend(halftone_signals)

        # --- Method B: Ink Bleed Detection ---
        bleed_score, bleed_signals = _detect_ink_bleed(gray)
        scores["ink_bleed"] = bleed_score
        signals.extend(bleed_signals)

        # --- Method C: Scan Line Artifacts ---
        scanline_score, scanline_signals = _detect_scan_lines(gray)
        scores["scan_lines"] = scanline_score
        signals.extend(scanline_signals)

        # --- Method D: Moiré Pattern Detection ---
        moire_score, moire_signals = _detect_moire(gray)
        scores["moire"] = moire_score
        signals.extend(moire_signals)

        # Combined confidence — weighted average
        # Halftone is the most reliable indicator; moiré is a strong secondary.
        confidence = (
            0.35 * halftone_score
            + 0.25 * bleed_score
            + 0.20 * scanline_score
            + 0.20 * moire_score
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

        is_print_scan = confidence > 0.55

        logger.debug(
            f"PrintScanDetector: confidence={confidence:.3f}, "
            f"halftone={halftone_score:.2f}, bleed={bleed_score:.2f}, "
            f"scanlines={scanline_score:.2f}, moire={moire_score:.2f}"
        )

        return {
            "is_print_scan": is_print_scan,
            "confidence": confidence,
            "signals": signals,
            "details": scores,
        }

    except Exception as e:
        logger.warning(f"PrintScanDetector failed: {e}")
        return {"is_print_scan": False, "confidence": 0.0, "signals": [], "details": {}}


def _detect_halftone_patterns(gray: np.ndarray) -> tuple[float, list[str]]:
    """
    Detect halftone dot patterns using FFT frequency domain analysis.

    Printed documents use halftone screens to reproduce continuous tones.
    When scanned, these dot patterns produce periodic peaks in the frequency
    spectrum at specific angles: 45° (black), 75° (cyan), 90° (yellow), 105° (magenta).

    A digital-only document has no periodic dot structure — its FFT shows
    continuous spectra, not isolated angular peaks.
    """
    signals = []

    # Resize for consistent FFT analysis (performance + normalization)
    h, w = gray.shape
    target_size = 1024
    if max(h, w) > target_size:
        scale = target_size / max(h, w)
        gray_small = cv2.resize(gray, (int(w * scale), int(h * scale)))
    else:
        gray_small = gray

    # 2D FFT and shift zero-frequency to center
    fft = np.fft.fft2(gray_small)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)

    # Log magnitude for better dynamic range
    magnitude_log = np.log1p(magnitude)

    # Normalize
    mag_norm = magnitude_log / (magnitude_log.max() + 1e-8)

    fh, fw = mag_norm.shape
    cy, cx = fh // 2, fw // 2

    # Halftone screen frequencies typically appear at 20-80% of Nyquist.
    # Define an annular region (ring) to search for angular peaks.
    y_idx, x_idx = np.mgrid[0:fh, 0:fw]
    dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    max_dist = min(cy, cx)

    # Look in 15-50% of max radius (typical halftone frequency range)
    ring_mask = (dist > 0.15 * max_dist) & (dist < 0.50 * max_dist)
    ring_region = mag_norm * ring_mask

    # Compute angular histogram to find periodic peaks
    angles = np.arctan2(y_idx - cy, x_idx - cx)
    angles_deg = np.degrees(angles) % 180  # Fold to 0-180° (symmetric)

    n_bins = 36  # 5° per bin
    angle_bins = np.linspace(0, 180, n_bins + 1)
    bin_sums = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (angles_deg >= angle_bins[i]) & (angles_deg < angle_bins[i + 1])
        bin_sums[i] = ring_region[mask & ring_mask].sum()

    # Normalize bins
    bin_sums /= (bin_sums.max() + 1e-8)

    # Halftone screens produce peaks at 45°, 75°, 90°, 105° (±5°)
    halftone_angles = [45, 75, 90, 105]
    peak_count = 0
    for target_angle in halftone_angles:
        bin_idx = int(target_angle / 180 * n_bins) % n_bins
        # Check the bin and its neighbors
        neighborhood = [bin_idx - 1, bin_idx, bin_idx + 1]
        peak_val = max(bin_sums[i % n_bins] for i in neighborhood)
        if peak_val > 0.65:  # Strong angular peak
            peak_count += 1

    # Additionally: check overall "peakiness" of the angular spectrum
    # A printed document has sharp peaks; digital has a flatter distribution.
    peak_ratio = bin_sums.max() / (bin_sums.mean() + 1e-8)

    halftone_score = 0.0
    if peak_count >= 2:
        halftone_score = min(1.0, 0.5 + 0.25 * (peak_count - 1))
        signals.append(
            f"Halftone screen detected: {peak_count} angular peaks in FFT "
            f"(typical of printed+scanned documents)"
        )
    elif peak_ratio > 4.0:
        halftone_score = 0.35
        signals.append(
            f"Periodic frequency peaks detected (peak ratio {peak_ratio:.1f}x), "
            "possible halftone pattern"
        )

    return halftone_score, signals


def _detect_ink_bleed(gray: np.ndarray) -> tuple[float, list[str]]:
    """
    Detect ink bleed patterns characteristic of printed+scanned documents.

    When ink is printed on paper, it slightly bleeds beyond the intended boundary
    (capillary action). Under scanning, this creates a characteristic gradient
    profile at text edges: sharp on one side, blurred on the other.

    A digital document has mathematically sharp edges (no physical bleeding).
    We measure the asymmetry ratio: bleed σ on dark→light vs light→dark transitions.
    """
    signals = []

    # Binarize with Otsu to get text/background separation
    _, binary = cv2.threshold(
        gray.astype(np.uint8), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Find edge pixels using Canny
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    if edges.sum() == 0:
        return 0.0, []

    # For each edge pixel, measure gradient profiles in 4 directions
    # Compare the slope leading into vs out of the edge
    kernel_size = 7
    laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F, ksize=kernel_size)

    # Ink bleed creates characteristic Laplacian signature:
    # asymmetric response at dark→light edges vs light→dark
    edge_coords = np.where(edges > 0)
    if len(edge_coords[0]) < 10:
        return 0.0, []

    # Sample up to 500 edge points for performance
    sample_size = min(500, len(edge_coords[0]))
    sample_idx = np.random.choice(len(edge_coords[0]), sample_size, replace=False)
    ys = edge_coords[0][sample_idx]
    xs = edge_coords[1][sample_idx]

    # Measure Laplacian values at edges — ink bleed creates asymmetric profiles
    lap_vals = np.abs(laplacian[ys, xs])
    lap_mean = float(np.mean(lap_vals))
    lap_std = float(np.std(lap_vals))

    # Compute edge sharpness variance
    # Uniform sharpness = digital; high variance with right-skew = print-scan
    if lap_mean > 0:
        variation_coeff = lap_std / lap_mean
    else:
        return 0.0, []

    # Ink bleed also creates a specific dilation pattern around dark regions.
    # Check if dark regions (text) are slightly over-expanded vs the binary reference.
    kernel = np.ones((3, 3), np.uint8)
    dilated_binary = cv2.dilate(binary, kernel, iterations=1)
    dilation_ratio = float(np.sum(dilated_binary < 128)) / (np.sum(binary < 128) + 1e-8)

    bleed_score = 0.0
    if variation_coeff > 1.5 and dilation_ratio > 1.15:
        bleed_score = min(1.0, 0.3 + 0.3 * (variation_coeff - 1.5) + 0.2 * (dilation_ratio - 1.0))
        signals.append(
            f"Ink bleed pattern detected: edge sharpness CV={variation_coeff:.2f}, "
            f"dark region expansion={dilation_ratio:.2f}x (scan characteristic)"
        )
    elif variation_coeff > 1.2:
        bleed_score = 0.25
        signals.append(
            f"Mild edge sharpness variance (CV={variation_coeff:.2f}), possible print artifact"
        )

    return float(np.clip(bleed_score, 0.0, 1.0)), signals


def _detect_scan_lines(gray: np.ndarray) -> tuple[float, list[str]]:
    """
    Detect horizontal scan-line banding from CCD/CIS scanner sensors.

    Physical scanners move a linear CCD or CIS (Contact Image Sensor) array
    across the document. Slight variations in sensor gain or illumination across
    the scan head produce faint horizontal banding — rows that are systematically
    slightly brighter or darker than their neighbors.

    This banding is periodic and strictly horizontal (row-wise), unlike random noise.
    We detect it by analyzing row-wise intensity variance in a smoothed image.
    """
    signals = []

    h, w = gray.shape

    # Smooth slightly to reduce random noise
    smoothed = cv2.GaussianBlur(gray, (1, 5), 0)  # 1D vertical blur

    # Compute row-mean intensity profile
    row_means = smoothed.mean(axis=1)  # Shape: (h,)

    # Detrend: remove global illumination gradient
    x = np.arange(h, dtype=np.float64)
    # Linear fit to global trend
    coeffs = np.polyfit(x, row_means, 1)
    trend = np.polyval(coeffs, x)
    detrended = row_means - trend

    # FFT of the row profile to find periodic banding
    row_fft = np.abs(np.fft.rfft(detrended))
    freqs = np.fft.rfftfreq(h)

    if len(row_fft) < 5:
        return 0.0, []

    # Look for peaks in scanner-typical frequency range (5-50 cycles across doc height)
    scan_freq_min = 5 / h
    scan_freq_max = 50 / h
    freq_mask = (freqs > scan_freq_min) & (freqs < scan_freq_max)

    if freq_mask.sum() == 0:
        return 0.0, []

    fft_in_range = row_fft[freq_mask]
    fft_baseline = row_fft[~freq_mask & (freqs > 0)]

    peak_in_range = float(fft_in_range.max()) if len(fft_in_range) > 0 else 0.0
    baseline_mean = float(fft_baseline.mean()) if len(fft_baseline) > 0 else 1.0

    peak_ratio = peak_in_range / (baseline_mean + 1e-8)

    # Also check: variance of row means should be unusually high for scan bands
    row_variance = float(np.var(detrended))
    overall_variance = float(np.var(gray))
    variance_ratio = row_variance / (overall_variance + 1e-8)

    scanline_score = 0.0
    if peak_ratio > 5.0 and variance_ratio > 0.01:
        scanline_score = min(1.0, 0.4 + 0.15 * np.log10(peak_ratio))
        signals.append(
            f"Horizontal scan-line banding detected: peak ratio {peak_ratio:.1f}x "
            f"above baseline (scanner CCD/CIS artifact)"
        )
    elif peak_ratio > 3.0:
        scanline_score = 0.25
        signals.append(f"Mild horizontal banding (peak ratio {peak_ratio:.1f}x)")

    return float(np.clip(scanline_score, 0.0, 1.0)), signals


def _detect_moire(gray: np.ndarray) -> tuple[float, list[str]]:
    """
    Detect Moiré interference patterns from halftone + scanner aliasing.

    When a halftone pattern (printed dots) is scanned, the scanner's sampling
    grid interferes with the dot pattern grid, producing Moiré interference.
    This shows up as a low-frequency wave-like pattern overlaid on the image.

    Detection: bandpass filter in the spatial domain to isolate the Moiré frequency
    range (typically 2-15 cycles per image width for typical document DPI/scanning combos).
    Then check if the filtered signal has significant energy that is spatially coherent.
    """
    signals = []

    # Resize for consistent analysis
    h, w = gray.shape
    target = 512
    if max(h, w) > target:
        scale = target / max(h, w)
        gray_small = cv2.resize(gray, (int(w * scale), int(h * scale)))
    else:
        gray_small = gray.copy()

    # Bandpass filter: keep only Moiré frequency range
    # Moiré typically appears at 5-30 cycles per image dimension
    gray_blur_low = cv2.GaussianBlur(gray_small, (0, 0), sigmaX=3.0)
    gray_blur_high = cv2.GaussianBlur(gray_small, (0, 0), sigmaX=12.0)

    # Bandpassed signal = low_sigma − high_sigma
    bandpassed = gray_blur_low - gray_blur_high

    # Measure energy and spatial coherence of the bandpassed signal
    bp_variance = float(np.var(bandpassed))
    bp_mean_abs = float(np.mean(np.abs(bandpassed)))

    # Check for spatial coherence using 2D autocorrelation proxy:
    # compute local variance in blocks — Moiré has high regularity
    block_size = 32
    bh, bw = gray_small.shape
    block_vars = []
    for i in range(0, bh - block_size, block_size):
        for j in range(0, bw - block_size, block_size):
            block = bandpassed[i:i+block_size, j:j+block_size]
            block_vars.append(float(np.var(block)))

    if not block_vars:
        return 0.0, []

    block_var_cv = float(np.std(block_vars)) / (float(np.mean(block_vars)) + 1e-8)

    # Low CV means variance is uniform across blocks → coherent Moiré pattern
    # High CV means variance is concentrated in patches → not Moiré
    is_coherent = block_var_cv < 0.8

    moire_score = 0.0
    if bp_variance > 50.0 and is_coherent:
        moire_score = min(1.0, 0.35 + bp_mean_abs / 20.0)
        signals.append(
            f"Moiré interference pattern detected: bandpass energy={bp_variance:.1f}, "
            f"spatial coherence CV={block_var_cv:.2f} (halftone+scanner aliasing)"
        )
    elif bp_variance > 30.0:
        moire_score = 0.20
        signals.append(f"Mild interference pattern in bandpass signal (energy={bp_variance:.1f})")

    return float(np.clip(moire_score, 0.0, 1.0)), signals
