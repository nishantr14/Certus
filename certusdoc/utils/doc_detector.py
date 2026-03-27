"""
Document Type Detector

Classifies documents based on OCR text keywords and Unicode range analysis.
Used by agents to adjust detection thresholds for structured documents
(government IDs, invoices, certificates) that have repeated design elements.
"""
import re
from enum import Enum
from dataclasses import dataclass


class DocType(Enum):
    GOVERNMENT_ID = "government_id"       # Aadhaar, PAN, DL, Passport
    CERTIFICATE = "certificate"           # Degree, marksheet, completion cert
    INVOICE = "invoice"                   # Tax invoice, receipt, bill
    BANK_DOCUMENT = "bank_document"       # Statement, cheque, passbook
    SALARY_SLIP = "salary_slip"           # Payslip, salary certificate
    LETTER = "letter"                     # Official letter, notice
    UNKNOWN = "unknown"


class ScriptType(Enum):
    LATIN_ONLY = "latin_only"
    DEVANAGARI = "devanagari"             # Hindi, Marathi, Sanskrit
    TAMIL = "tamil"
    TELUGU = "telugu"
    BENGALI = "bengali"
    KANNADA = "kannada"
    MALAYALAM = "malayalam"
    GUJARATI = "gujarati"
    MULTI_SCRIPT = "multi_script"         # Mixed scripts detected


@dataclass
class DocClassification:
    doc_type: DocType
    script_type: ScriptType
    is_government: bool
    is_structured: bool           # Has repeated design elements (headers, borders, logos)
    is_multi_language: bool
    confidence: float             # 0-1 how confident we are in classification
    detected_keywords: list[str]


# Keyword patterns for document type detection
_DOC_TYPE_PATTERNS = {
    DocType.GOVERNMENT_ID: {
        "keywords": [
            "aadhaar", "unique identification", "uidai", "government of india",
            "permanent account number", "income tax", "pan card",
            "driving licence", "driving license", "motor vehicle", "rto",
            "passport", "republic of india", "nationality indian",
            "voter id", "election commission", "ration card",
            "enrolment", "vid ", "dob", "male", "female",
        ],
        "min_hits": 2,
    },
    DocType.CERTIFICATE: {
        "keywords": [
            "certificate", "certify", "certifies", "hereby", "awarded",
            "degree", "marksheet", "marks statement", "semester",
            "university", "institute", "college", "board of examination",
            "grade", "cgpa", "sgpa", "distinction", "first class",
            "completion", "training", "course",
        ],
        "min_hits": 2,
    },
    DocType.INVOICE: {
        "keywords": [
            "invoice", "tax invoice", "bill", "receipt", "gst",
            "gstin", "total amount", "subtotal", "grand total",
            "payment", "due date", "bill to", "ship to",
            "hsn", "sac code", "cgst", "sgst", "igst",
        ],
        "min_hits": 2,
    },
    DocType.BANK_DOCUMENT: {
        "keywords": [
            "bank", "account statement", "account number", "ifsc",
            "branch", "balance", "debit", "credit", "transaction",
            "savings account", "current account", "cheque",
            "passbook", "neft", "rtgs", "imps",
        ],
        "min_hits": 2,
    },
    DocType.SALARY_SLIP: {
        "keywords": [
            "salary", "payslip", "pay slip", "earnings", "deductions",
            "basic pay", "hra", "da", "pf", "esi", "gross salary",
            "net salary", "net pay", "employee", "designation",
            "ctc", "take home", "professional tax",
        ],
        "min_hits": 2,
    },
}

# Unicode ranges for Indian scripts
_SCRIPT_RANGES = {
    ScriptType.DEVANAGARI: (0x0900, 0x097F),
    ScriptType.BENGALI: (0x0980, 0x09FF),
    ScriptType.GUJARATI: (0x0A80, 0x0AFF),
    ScriptType.TAMIL: (0x0B80, 0x0BFF),
    ScriptType.TELUGU: (0x0C00, 0x0C7F),
    ScriptType.KANNADA: (0x0C80, 0x0CFF),
    ScriptType.MALAYALAM: (0x0D00, 0x0D7F),
}


def detect_document_type(ocr_text: list[str]) -> DocClassification:
    """
    Classify a document based on its OCR text content.

    Args:
        ocr_text: List of OCR text strings (one per page).

    Returns:
        DocClassification with type, script, and metadata.
    """
    full_text = " ".join(ocr_text).lower()
    full_text_raw = " ".join(ocr_text)

    # Detect document type
    best_type = DocType.UNKNOWN
    best_hits = 0
    best_keywords = []

    for doc_type, patterns in _DOC_TYPE_PATTERNS.items():
        hits = []
        for kw in patterns["keywords"]:
            if kw in full_text:
                hits.append(kw)
        if len(hits) >= patterns["min_hits"] and len(hits) > best_hits:
            best_type = doc_type
            best_hits = len(hits)
            best_keywords = hits

    # Detect scripts
    scripts_found = set()
    has_latin = False

    for ch in full_text_raw:
        cp = ord(ch)
        if 0x0041 <= cp <= 0x007A:  # Basic Latin letters
            has_latin = True
        for script_type, (start, end) in _SCRIPT_RANGES.items():
            if start <= cp <= end:
                scripts_found.add(script_type)

    if len(scripts_found) > 1:
        script_type = ScriptType.MULTI_SCRIPT
    elif len(scripts_found) == 1:
        script_type = list(scripts_found)[0]
    elif has_latin:
        script_type = ScriptType.LATIN_ONLY
    else:
        script_type = ScriptType.LATIN_ONLY

    is_multi_language = has_latin and len(scripts_found) >= 1
    is_government = best_type == DocType.GOVERNMENT_ID
    is_structured = best_type in (
        DocType.GOVERNMENT_ID, DocType.CERTIFICATE,
        DocType.INVOICE, DocType.BANK_DOCUMENT, DocType.SALARY_SLIP,
    )

    confidence = min(1.0, best_hits / 4.0) if best_hits > 0 else 0.0

    return DocClassification(
        doc_type=best_type,
        script_type=script_type,
        is_government=is_government,
        is_structured=is_structured,
        is_multi_language=is_multi_language,
        confidence=confidence,
        detected_keywords=best_keywords,
    )
