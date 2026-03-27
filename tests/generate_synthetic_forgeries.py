"""
Synthetic Forgery Generator for CertusDoc Evaluation

Generates 10 types of document forgeries from clean source images:
1. Text replacement
2. Copy-move
3. Splicing
4. Double compression
5. Triple compression
6. Noise injection
7. Brightness manipulation
8. Blur attack
9. Metadata stripping
10. Resolution mismatch

Usage:
    python tests/generate_synthetic_forgeries.py
    python tests/generate_synthetic_forgeries.py --source data/authentic
    python tests/generate_synthetic_forgeries.py --count 20  # per forgery type
"""
import os
import sys
import csv
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

OUTPUT_DIR = Path("data/synthetic_forged")
GROUND_TRUTH_CSV = OUTPUT_DIR / "ground_truth.csv"


def load_source_images(source_dir: str, limit: int = 10) -> list[tuple[str, np.ndarray]]:
    """Load source images from a directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = []
    source_path = Path(source_dir)

    candidates = []
    for f in sorted(source_path.rglob("*")):
        if f.is_file() and f.suffix.lower() in extensions:
            candidates.append(f)

    # Sample evenly if too many
    if len(candidates) > limit:
        candidates = random.sample(candidates, limit)

    for f in candidates:
        img = cv2.imread(str(f))
        if img is not None and img.shape[0] >= 200 and img.shape[1] >= 200:
            images.append((f.name, img))

    return images


# ================================================================
# Forgery generators
# ================================================================

def forge_text_replacement(img: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Replace a text-like region with white + new text overlay."""
    h, w = img.shape[:2]
    # Find a plausible text region (upper-middle of document)
    rx = random.randint(w // 6, w // 2)
    ry = random.randint(h // 6, h // 3)
    rw = random.randint(100, min(250, w - rx))
    rh = random.randint(25, 50)

    out = img.copy()
    # White out the region
    out[ry:ry+rh, rx:rx+rw] = 255
    # Draw fake text
    cv2.putText(out, "FORGED TEXT", (rx + 5, ry + rh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return out, (rx, ry, rw, rh)


def forge_copy_move(img: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Copy a region and paste it elsewhere."""
    h, w = img.shape[:2]
    size = min(100, h // 4, w // 4)
    sx = random.randint(10, w - size - 220)
    sy = random.randint(10, h - size - 10)

    out = img.copy()
    region = img[sy:sy+size, sx:sx+size].copy()
    # Paste 200px to the right (or left if no room)
    dx = min(200, w - sx - size - 5)
    if dx < 50:
        dx = -min(200, sx - 5)
    out[sy:sy+size, sx+dx:sx+dx+size] = region
    return out, (sx+dx, sy, size, size)


def forge_splicing(img: np.ndarray, donor: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Splice a region from one image into another."""
    h, w = img.shape[:2]
    dh, dw = donor.shape[:2]
    size = min(120, h // 3, w // 3, dh // 3, dw // 3)

    # Extract region from donor
    dx = random.randint(0, dw - size)
    dy = random.randint(0, dh - size)
    patch = donor[dy:dy+size, dx:dx+size].copy()

    # Paste into target
    tx = random.randint(w // 4, w - size - 10)
    ty = random.randint(h // 4, h - size - 10)

    out = img.copy()
    # Resize patch to target size if needed
    patch = cv2.resize(patch, (size, size))
    out[ty:ty+size, tx:tx+size] = patch
    return out, (tx, ty, size, size)


def forge_double_compression(img: np.ndarray) -> tuple[np.ndarray, None]:
    """Save at JPEG q40, reload, save at q90."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    import io
    buf1 = io.BytesIO()
    pil.save(buf1, format="JPEG", quality=40)
    buf1.seek(0)
    img2 = np.array(Image.open(buf1))

    buf2 = io.BytesIO()
    Image.fromarray(img2).save(buf2, format="JPEG", quality=90)
    buf2.seek(0)
    result = cv2.cvtColor(np.array(Image.open(buf2)), cv2.COLOR_RGB2BGR)
    return result, None


def forge_triple_compression(img: np.ndarray) -> tuple[np.ndarray, None]:
    """Save at q30, q60, q90."""
    import io
    current = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for q in [30, 60, 90]:
        buf = io.BytesIO()
        current.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        current = Image.open(buf)
    result = cv2.cvtColor(np.array(current), cv2.COLOR_RGB2BGR)
    return result, None


def forge_noise_injection(img: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Add Gaussian noise to a random 200x200 region."""
    h, w = img.shape[:2]
    size = min(200, h // 2, w // 2)
    rx = random.randint(10, w - size - 10)
    ry = random.randint(10, h - size - 10)

    out = img.copy().astype(np.float32)
    noise = np.random.normal(0, 15, (size, size, 3))
    out[ry:ry+size, rx:rx+size] += noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, (rx, ry, size, size)


def forge_brightness(img: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Increase brightness by 20% in one region."""
    h, w = img.shape[:2]
    rw = random.randint(100, min(250, w // 2))
    rh = random.randint(30, min(80, h // 4))
    rx = random.randint(10, w - rw - 10)
    ry = random.randint(10, h - rh - 10)

    out = img.copy().astype(np.float32)
    out[ry:ry+rh, rx:rx+rw] *= 1.20
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, (rx, ry, rw, rh)


def forge_blur_attack(img: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Apply Gaussian blur (kernel=5) to a specific region."""
    h, w = img.shape[:2]
    size = min(150, h // 3, w // 3)
    rx = random.randint(10, w - size - 10)
    ry = random.randint(10, h - size - 10)

    out = img.copy()
    region = out[ry:ry+size, rx:rx+size]
    blurred = cv2.GaussianBlur(region, (5, 5), 0)
    out[ry:ry+size, rx:rx+size] = blurred
    return out, (rx, ry, size, size)


def forge_metadata_strip(img: np.ndarray) -> tuple[np.ndarray, None]:
    """Remove all metadata by re-encoding as raw pixel data."""
    # Just return a clean copy — metadata is stripped during save
    return img.copy(), None


def forge_resolution_mismatch(img: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Downscale one region to 72 DPI then upscale back to 300 DPI."""
    h, w = img.shape[:2]
    size = min(200, h // 3, w // 3)
    rx = random.randint(10, w - size - 10)
    ry = random.randint(10, h - size - 10)

    out = img.copy()
    region = out[ry:ry+size, rx:rx+size]
    # Downscale to 24% (72/300) then upscale back
    small = cv2.resize(region, None, fx=0.24, fy=0.24, interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR)
    out[ry:ry+size, rx:rx+size] = restored
    return out, (rx, ry, size, size)


# ================================================================
# Main generator
# ================================================================

FORGERY_TYPES = {
    "text_replacement": forge_text_replacement,
    "copy_move": forge_copy_move,
    "splicing": None,  # needs donor image
    "double_compression": forge_double_compression,
    "triple_compression": forge_triple_compression,
    "noise_injection": forge_noise_injection,
    "brightness_manipulation": forge_brightness,
    "blur_attack": forge_blur_attack,
    "metadata_strip": forge_metadata_strip,
    "resolution_mismatch": forge_resolution_mismatch,
}


def generate_all(source_images: list[tuple[str, np.ndarray]], count_per_type: int):
    """Generate all forgery types and save to output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create forged/ subdirectory
    forged_dir = OUTPUT_DIR / "forged"
    forged_dir.mkdir(exist_ok=True)

    records = []
    total = 0

    for forgery_name, forge_fn in FORGERY_TYPES.items():
        print(f"Generating {forgery_name}...")
        generated = 0

        for i in range(min(count_per_type, len(source_images))):
            src_name, src_img = source_images[i % len(source_images)]
            base = Path(src_name).stem

            try:
                if forgery_name == "splicing":
                    # Need a different donor image
                    donor_idx = (i + 1) % len(source_images)
                    _, donor_img = source_images[donor_idx]
                    forged_img, region = forge_splicing(src_img, donor_img)
                else:
                    forged_img, region = forge_fn(src_img)

                out_name = f"{forgery_name}_{base}_{i:03d}.jpg"
                out_path = forged_dir / out_name

                cv2.imwrite(str(out_path), forged_img,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])

                region_str = f"{region[0]},{region[1]},{region[2]},{region[3]}" if region else ""
                records.append({
                    "filename": f"forged/{out_name}",
                    "forgery_type": forgery_name,
                    "forged_region": region_str,
                    "source": src_name,
                    "label": "forged",
                })
                generated += 1
                total += 1

            except Exception as e:
                print(f"  Error on {src_name}: {e}")

        print(f"  Generated {generated} {forgery_name} images")

    # Write ground truth CSV
    with open(GROUND_TRUTH_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "forgery_type",
                                                "forged_region", "source", "label"])
        writer.writeheader()
        writer.writerows(records)

    print(f"\nTotal: {total} synthetic forgeries saved to {OUTPUT_DIR}")
    print(f"Ground truth: {GROUND_TRUTH_CSV}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic document forgeries")
    parser.add_argument("--source", default="data/authentic",
                        help="Source directory for clean images")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of forgeries per type (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load source images
    source_dir = args.source
    if not Path(source_dir).exists():
        # Fallback to roboflow images as source
        alt_dirs = ["data/roboflow/train", "data/roboflow"]
        for alt in alt_dirs:
            if Path(alt).exists():
                source_dir = alt
                break

    print(f"Loading source images from: {source_dir}")
    images = load_source_images(source_dir, limit=args.count)

    if not images:
        print(f"No images found in {source_dir}")
        sys.exit(1)

    print(f"Loaded {len(images)} source images")
    generate_all(images, args.count)


if __name__ == "__main__":
    main()
