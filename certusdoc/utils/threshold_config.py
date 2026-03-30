"""
Per-Document-Type Threshold Configuration

Forensic rationale: different document types have different visual and textual
characteristics. A government ID has a rigid layout; a salary slip is more variable.
Applying uniform thresholds across all types causes both false positives (structured
documents with repetitive patterns trigger ELA) and false negatives (informal documents
with large acceptable variance pass text checks too easily).
"""
from dataclasses import dataclass
from enum import Enum


class ThresholdDocType(Enum):
    """Document type for threshold selection. Maps from DocType in doc_detector."""
    AADHAAR = "aadhaar"
    PAN = "pan"
    DRIVING_LICENSE = "driving_license"
    PASSPORT = "passport"
    VOTER_ID = "voter_id"
    BANK_DOCUMENT = "bank_document"
    SALARY_SLIP = "salary_slip"
    CERTIFICATE = "certificate"
    INVOICE = "invoice"
    GENERIC = "generic"


@dataclass
class DocTypeThresholds:
    """
    Per-document-type threshold configuration for DetectionAgents and FusionEngine.

    Fields:
        ela_anomaly_multiplier: Multiply ELA anomaly threshold by this factor.
            >1.0 = more lenient (for tools like wkhtmltopdf that produce ELA artifacts).
            <1.0 = stricter.
        ocr_confidence_min: Minimum acceptable OCR confidence for text agent to be
            fully reliable. Below this, text agent reliability weight is penalized.
        metadata_weight: Base weight for metadata agent in fusion (before dynamic
            adjustments).
        text_weight: Base weight for text agent.
        visual_weight: Base weight for visual agent.
        whatsapp_cap: Maximum DIS for a document of this type shared via WhatsApp.
            Government IDs shared via WhatsApp should be capped lower than generic docs.
        consumer_tool_cap: Maximum DIS when created with a consumer/mobile tool.
        verhoeff_required: Whether Verhoeff checksum validation is expected.
        mrz_validation: Whether MRZ checksum validation is expected.
        expected_metadata_fields: Minimum number of metadata fields expected.
        notes: Human-readable explanation of why these thresholds differ from generic.
    """
    ela_anomaly_multiplier: float = 1.0
    ocr_confidence_min: float = 50.0
    metadata_weight: float = 0.22
    text_weight: float = 0.40
    visual_weight: float = 0.38
    whatsapp_cap: float = 0.35
    consumer_tool_cap: float = 0.25
    verhoeff_required: bool = False
    mrz_validation: bool = False
    expected_metadata_fields: int = 2
    notes: str = ""


# === THRESHOLD PRESETS ===
# Each preset is tuned to the typical forensic characteristics of that doc type.

THRESHOLD_PRESETS: dict[ThresholdDocType, DocTypeThresholds] = {

    ThresholdDocType.AADHAAR: DocTypeThresholds(
        # wkhtmltopdf-rendered e-Aadhaar PDFs have strong JPEG artifacts from the
        # HTML→PDF→image pipeline. Raise ELA threshold to avoid false positives.
        ela_anomaly_multiplier=1.5,
        # Aadhaar cards mix Hindi+English; OCR confidence can be lower legitimately.
        ocr_confidence_min=45.0,
        # Metadata is the STRONGEST signal for Aadhaar — wkhtmltopdf vs iOS Quartz
        # is a near-perfect discriminator. Give it heavy weight.
        metadata_weight=0.35,
        text_weight=0.35,
        visual_weight=0.30,
        # Genuine Aadhaar should NEVER be shared via WhatsApp — hard cap.
        whatsapp_cap=0.30,
        consumer_tool_cap=0.20,
        verhoeff_required=True,
        expected_metadata_fields=3,
        notes="Strict: wkhtmltopdf/DigiLocker expected. Verhoeff+QR required. WhatsApp = broken provenance.",
    ),

    ThresholdDocType.PAN: DocTypeThresholds(
        # PAN cards are typically scanned; ELA is slightly more tolerant of scan noise.
        ela_anomaly_multiplier=1.2,
        ocr_confidence_min=50.0,
        metadata_weight=0.30,
        text_weight=0.40,
        visual_weight=0.30,
        whatsapp_cap=0.35,
        consumer_tool_cap=0.22,
        verhoeff_required=False,
        expected_metadata_fields=2,
        notes="Moderate: font consistency (PAN uses Helvetica-style) is important. NSDL/UTI tools expected.",
    ),

    ThresholdDocType.DRIVING_LICENSE: DocTypeThresholds(
        ela_anomaly_multiplier=1.2,
        ocr_confidence_min=45.0,
        metadata_weight=0.28,
        text_weight=0.38,
        visual_weight=0.34,
        whatsapp_cap=0.35,
        consumer_tool_cap=0.22,
        expected_metadata_fields=2,
        notes="Moderate: RTO/mParivahan/DigiLocker tools expected. Format varies by state.",
    ),

    ThresholdDocType.PASSPORT: DocTypeThresholds(
        # Passports use MRZ (Machine Readable Zone) — 2×44 char lines with checksums.
        # ELA should be strict since passports use special paper/printing.
        ela_anomaly_multiplier=0.9,
        ocr_confidence_min=55.0,
        metadata_weight=0.25,
        text_weight=0.45,
        visual_weight=0.30,
        whatsapp_cap=0.25,
        consumer_tool_cap=0.15,
        mrz_validation=True,
        expected_metadata_fields=3,
        notes="Strict: MRZ checksum validation. No legitimate WhatsApp sharing of passports.",
    ),

    ThresholdDocType.VOTER_ID: DocTypeThresholds(
        ela_anomaly_multiplier=1.3,
        ocr_confidence_min=40.0,  # Voter IDs vary widely in print quality
        metadata_weight=0.25,
        text_weight=0.38,
        visual_weight=0.37,
        whatsapp_cap=0.35,
        consumer_tool_cap=0.22,
        expected_metadata_fields=1,
        notes="Lenient ELA: voter ID physical cards have high scan variability across states.",
    ),

    ThresholdDocType.BANK_DOCUMENT: DocTypeThresholds(
        ela_anomaly_multiplier=1.1,
        ocr_confidence_min=55.0,
        metadata_weight=0.20,
        text_weight=0.45,  # Text (IFSC, account numbers) is primary signal
        visual_weight=0.35,
        whatsapp_cap=0.45,
        consumer_tool_cap=0.40,  # Banks legitimately use consumer PDF tools
        expected_metadata_fields=2,
        notes="Text-heavy: IFSC/account format validation is primary. PDF from bank portal is normal.",
    ),

    ThresholdDocType.SALARY_SLIP: DocTypeThresholds(
        ela_anomaly_multiplier=1.0,
        ocr_confidence_min=50.0,
        metadata_weight=0.18,
        text_weight=0.45,
        visual_weight=0.37,
        whatsapp_cap=0.50,
        consumer_tool_cap=0.45,  # Salary slips are often generated by HR software/Excel
        expected_metadata_fields=2,
        notes="Moderate: salary slips are often generated by Excel/HR platforms, so consumer tools are acceptable.",
    ),

    ThresholdDocType.CERTIFICATE: DocTypeThresholds(
        ela_anomaly_multiplier=1.0,
        ocr_confidence_min=55.0,
        metadata_weight=0.20,
        text_weight=0.42,
        visual_weight=0.38,
        whatsapp_cap=0.45,
        consumer_tool_cap=0.35,
        expected_metadata_fields=2,
        notes="Moderate: universities use varied software. Emphasis on text content consistency.",
    ),

    ThresholdDocType.INVOICE: DocTypeThresholds(
        ela_anomaly_multiplier=1.0,
        ocr_confidence_min=55.0,
        metadata_weight=0.18,
        text_weight=0.48,  # GST/GSTIN format validation drives most detection
        visual_weight=0.34,
        whatsapp_cap=0.50,
        consumer_tool_cap=0.50,  # Many small businesses use Tally/Excel/Word
        expected_metadata_fields=1,
        notes="Lenient: small businesses use consumer tools. GSTIN format validation is primary.",
    ),

    ThresholdDocType.GENERIC: DocTypeThresholds(
        ela_anomaly_multiplier=1.0,
        ocr_confidence_min=50.0,
        metadata_weight=0.22,
        text_weight=0.40,
        visual_weight=0.38,
        whatsapp_cap=0.50,
        consumer_tool_cap=0.40,
        expected_metadata_fields=2,
        notes="Default thresholds. Applied when doc type cannot be determined.",
    ),
}


def get_thresholds(ocr_text: list[str], metadata: dict) -> DocTypeThresholds:
    """
    Determine which threshold preset to use based on document content and metadata.

    Uses keyword matching and metadata fields to identify the specific document
    sub-type (e.g., AADHAAR vs generic GOVERNMENT_ID).

    Args:
        ocr_text: OCR text strings from all pages.
        metadata: Document metadata dict from ingestion.

    Returns:
        DocTypeThresholds preset for the detected document type.
    """
    full_text = " ".join(ocr_text).lower()

    # === Specific sub-type detection (order matters — most-specific first) ===

    # Aadhaar: UIDAI-specific keywords
    if any(kw in full_text for kw in ["aadhaar", "uidai", "unique identification authority",
                                       "enrolment no", "vid "]):
        return THRESHOLD_PRESETS[ThresholdDocType.AADHAAR]

    # Passport: MRZ indicators or passport-specific text
    if any(kw in full_text for kw in ["republic of india", "passport no", "nationality",
                                       "place of birth", "date of issue", "date of expiry",
                                       "machine readable"]):
        return THRESHOLD_PRESETS[ThresholdDocType.PASSPORT]

    # PAN: Income tax specific
    if any(kw in full_text for kw in ["permanent account number", "income tax department",
                                       "pan card", "nsdl", "utiitsl"]):
        return THRESHOLD_PRESETS[ThresholdDocType.PAN]

    # Driving License: RTO/transport
    if any(kw in full_text for kw in ["driving licence", "driving license",
                                       "motor vehicle", "transport", "rto"]):
        return THRESHOLD_PRESETS[ThresholdDocType.DRIVING_LICENSE]

    # Voter ID: Election Commission
    if any(kw in full_text for kw in ["election commission", "voter", "epic",
                                       "electors photo identity"]):
        return THRESHOLD_PRESETS[ThresholdDocType.VOTER_ID]

    # Bank document
    if any(kw in full_text for kw in ["account statement", "ifsc", "account no",
                                       "savings account", "current account",
                                       "transaction", "neft", "rtgs"]):
        return THRESHOLD_PRESETS[ThresholdDocType.BANK_DOCUMENT]

    # Salary slip
    if any(kw in full_text for kw in ["salary", "payslip", "pay slip", "basic pay",
                                       "hra", "gross salary", "net salary", "net pay"]):
        return THRESHOLD_PRESETS[ThresholdDocType.SALARY_SLIP]

    # Certificate
    if any(kw in full_text for kw in ["certificate", "certify", "hereby", "awarded",
                                       "degree", "marksheet", "university"]):
        return THRESHOLD_PRESETS[ThresholdDocType.CERTIFICATE]

    # Invoice/GST
    if any(kw in full_text for kw in ["tax invoice", "gstin", "gst no", "cgst", "sgst"]):
        return THRESHOLD_PRESETS[ThresholdDocType.INVOICE]

    # Default
    return THRESHOLD_PRESETS[ThresholdDocType.GENERIC]
