"""
Document Ingestion Pipeline (Stage 1)
Handles: PDF/image intake → 300 DPI extraction → Tesseract OCR → metadata extraction
"""
import os
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image
from loguru import logger

from certusdoc.models import Document


TARGET_DPI = 300

# Configure Tesseract path on Windows if not in PATH
import shutil
if not shutil.which("tesseract"):
    import pytesseract
    _tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for _path in _tesseract_paths:
        if os.path.isfile(_path):
            pytesseract.pytesseract.tesseract_cmd = _path
            break


def ingest_document(file_path: Union[str, Path]) -> Document:
    """
    Ingest a document file and prepare it for analysis.
    
    Supports: PDF, PNG, JPG, JPEG, TIFF, BMP
    
    Args:
        file_path: Path to the document file.
        
    Returns:
        Document object ready for agent analysis.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    logger.info(f"Ingesting document: {file_path.name}")
    start_time = time.time()

    file_size = file_path.stat().st_size
    suffix = file_path.suffix.lower()

    # Extract pages as images
    if suffix == ".pdf":
        pages = _extract_pdf_pages(file_path)
        metadata = _extract_pdf_metadata(file_path)
    elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
        pages = _extract_image_pages(file_path)
        metadata = _extract_image_metadata(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    # Run OCR on all pages
    ocr_text = []
    ocr_confidence = []
    ocr_word_data = []

    for i, page_img in enumerate(pages):
        logger.info(f"Running OCR on page {i + 1}/{len(pages)}")
        text, confidence, word_data = _run_ocr(page_img)
        ocr_text.append(text)
        ocr_confidence.append(confidence)
        ocr_word_data.append(word_data)

    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"Ingestion complete in {elapsed_ms:.0f}ms — {len(pages)} page(s)")

    return Document(
        file_path=str(file_path),
        file_name=file_path.name,
        file_size_bytes=file_size,
        pages=pages,
        ocr_text=ocr_text,
        ocr_confidence=ocr_confidence,
        ocr_word_data=ocr_word_data,
        metadata=metadata,
        original_format=suffix.lstrip(".")
    )


def _extract_pdf_pages(file_path: Path) -> list[np.ndarray]:
    """Extract pages from PDF as 300 DPI images."""
    try:
        from pdf2image import convert_from_path

        # Find poppler on Windows
        poppler_path = None
        _poppler_dirs = [
            r"C:\Users\nisha\AppData\Local\poppler\poppler-24.08.0\Library\bin",
        ]
        for _dir in _poppler_dirs:
            if os.path.isdir(_dir):
                poppler_path = _dir
                break

        pil_images = convert_from_path(
            str(file_path), dpi=TARGET_DPI,
            poppler_path=poppler_path,
        )
        pages = []
        for img in pil_images:
            # Convert PIL Image to BGR numpy array (OpenCV format)
            rgb = np.array(img)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pages.append(bgr)
        return pages
    except Exception as e:
        logger.error(f"PDF page extraction failed: {e}")
        raise


def _extract_image_pages(file_path: Path) -> list[np.ndarray]:
    """Load a single image file as a one-page document."""
    img = cv2.imread(str(file_path))
    if img is None:
        raise ValueError(f"Could not read image: {file_path}")
    return [img]


def _extract_pdf_metadata(file_path: Path) -> dict:
    """Extract PDF metadata using pikepdf."""
    metadata = {
        "source": "pdf",
        "creation_tool": None,
        "creator": None,
        "producer": None,
        "creation_date": None,
        "modification_date": None,
        "page_count": 0,
        "embedded_fonts": [],
        "has_javascript": False,
        "has_forms": False,
        "raw": {}
    }

    try:
        import pikepdf
        with pikepdf.open(str(file_path)) as pdf:
            metadata["page_count"] = len(pdf.pages)

            info = pdf.docinfo
            if info:
                raw = {}
                for key in info.keys():
                    try:
                        raw[str(key)] = str(info[key])
                    except Exception:
                        pass
                metadata["raw"] = raw
                metadata["creator"] = raw.get("/Creator")
                metadata["producer"] = raw.get("/Producer")
                metadata["creation_date"] = raw.get("/CreationDate")
                metadata["modification_date"] = raw.get("/ModDate")
                metadata["creation_tool"] = raw.get("/Creator") or raw.get("/Producer")

            # Check for JavaScript
            if "/Names" in pdf.Root:
                names = pdf.Root["/Names"]
                if "/JavaScript" in names:
                    metadata["has_javascript"] = True

            # Check for forms
            if "/AcroForm" in pdf.Root:
                metadata["has_forms"] = True

            # Extract embedded font names
            fonts = set()
            for page in pdf.pages:
                if "/Resources" in page and "/Font" in page["/Resources"]:
                    font_dict = page["/Resources"]["/Font"]
                    for font_key in font_dict.keys():
                        try:
                            font_obj = font_dict[font_key]
                            if "/BaseFont" in font_obj:
                                fonts.add(str(font_obj["/BaseFont"]))
                        except Exception:
                            pass
            metadata["embedded_fonts"] = list(fonts)

    except Exception as e:
        logger.warning(f"PDF metadata extraction partial failure: {e}")

    return metadata


def _extract_image_metadata(file_path: Path) -> dict:
    """Extract EXIF and image metadata."""
    metadata = {
        "source": "image",
        "creation_tool": None,
        "width": 0,
        "height": 0,
        "format": None,
        "mode": None,
        "exif": {},
        "raw": {}
    }

    try:
        img = Image.open(str(file_path))
        metadata["width"] = img.width
        metadata["height"] = img.height
        metadata["format"] = img.format
        metadata["mode"] = img.mode

        # Extract EXIF
        exif_data = img.getexif()
        if exif_data:
            from PIL.ExifTags import TAGS
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                try:
                    metadata["exif"][tag_name] = str(value)
                except Exception:
                    pass

            # Software used
            if "Software" in metadata["exif"]:
                metadata["creation_tool"] = metadata["exif"]["Software"]

    except Exception as e:
        logger.warning(f"Image metadata extraction failed: {e}")

    return metadata


def _run_ocr(image: np.ndarray) -> tuple[str, float, list[dict]]:
    """
    Run Tesseract OCR on an image.
    
    Returns:
        Tuple of (full_text, average_confidence, word_data_list)
        where word_data_list contains dicts with keys: text, x, y, w, h, conf
    """
    try:
        import pytesseract

        # Convert to RGB for Tesseract
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get full text
        full_text = pytesseract.image_to_string(rgb)

        # Get word-level data with confidence
        data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)

        word_data = []
        confidences = []

        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])

            if text and conf > 0:  # Skip empty entries and -1 confidence
                word_data.append({
                    "text": text,
                    "x": data["left"][i],
                    "y": data["top"][i],
                    "w": data["width"][i],
                    "h": data["height"][i],
                    "conf": conf
                })
                confidences.append(conf)

        avg_confidence = np.mean(confidences) if confidences else 0.0

        return full_text, avg_confidence, word_data

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return "", 0.0, []
