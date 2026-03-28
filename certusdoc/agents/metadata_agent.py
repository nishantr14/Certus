"""
Metadata & Provenance Agent

Detection methods:
1. Creation tool analysis — check if creation software is consistent with document appearance
2. Timestamp analysis — detect suspicious modification patterns
3. EXIF consistency — detect missing/inconsistent EXIF data
4. Font embedding analysis — detect unusual font patterns
5. Metadata completeness scoring — flag documents with suspicious metadata gaps

This agent catches: metadata spoofing, tool signature inconsistencies, timestamp anomalies
"""
import time
import re
from datetime import datetime
from typing import Optional

import numpy as np
from loguru import logger

from certusdoc.agents.base import BaseAgent
from certusdoc.models import Document, AgentResult, AgentFinding


# Known suspicious tool signatures
EDITING_TOOLS = {
    "photoshop", "gimp", "paint.net", "pixlr", "affinity photo",
    "adobe photoshop", "adobe illustrator", "corel", "inkscape",
    "sketch", "figma", "canva",
}

# Consumer/mobile PDF and image tools — NOT government software
CONSUMER_TOOLS = {
    "quartz pdfcontext", "ios quartz", "apple preview", "preview",
    "apple pdfkit", "cgpdfcontext", "mac os x quartz",
    "microsoft print to pdf", "cutepdf", "bullzip",
    "foxit", "nitro", "smallpdf", "ilovepdf", "sejda",
    "pdfcreator", "pdf-xchange", "camscanner", "adobe scan",
    "genius scan", "office lens", "microsoft lens",
    "google drive", "onedrive",
}

# Mobile device indicators in metadata
MOBILE_DEVICE_INDICATORS = {
    "iphone", "ipad", "android", "samsung", "pixel", "oneplus",
    "xiaomi", "huawei", "oppo", "vivo", "realme", "motorola",
}

LEGITIMATE_SCAN_TOOLS = {
    "scanner", "epson", "canon", "hp", "brother", "fujitsu",
    "twain", "wia", "scansnap", "ricoh", "xerox", "konica",
}

LEGITIMATE_PDF_CREATORS = {
    "microsoft", "word", "excel", "powerpoint", "libreoffice",
    "openoffice", "google docs", "latex", "pdflatex", "xelatex",
    "lualatex", "pdftex", "xetex", "luatex",
    "wkhtmltopdf", "prince", "weasyprint", "reportlab",
    "itext", "fpdf", "tcpdf", "mpdf", "dompdf", "jspdf",
    "pdfkit", "puppeteer", "chromium", "chrome print",
    "sap", "workday", "oracle", "tally",
    "digilocker", "umang",
}

# Tools known to be used by Indian government systems for official document generation
GOVERNMENT_TOOLS = {
    "wkhtmltopdf", "government", "uidai", "nsdl", "utiitsl",
    "digilocker", "umang", "mparivahan", "vahan",
    "itext", "reportlab", "fpdf", "tcpdf",
}

# Government/official document indicators
OFFICIAL_DOC_INDICATORS = [
    "aadhaar", "pan card", "passport", "driving license",
    "birth certificate", "marksheet", "degree", "salary slip",
    "payslip", "invoice", "receipt",
]

# Indian document validation patterns
INDIAN_DOC_PATTERNS = {
    "aadhaar": {
        # Matches full (1234 5678 9012) and masked (XXXX XXXX 2860) Aadhaar numbers
        "regex": r"\b(?:\d{4}|[Xx]{4})\s?(?:\d{4}|[Xx]{4})\s?\d{4}\b",
        "keywords": ["aadhaar", "unique identification", "uidai", "government of india",
                      "enrolment", "vid"],
        "expected_tools": ["scanner", "epson", "canon", "hp", "brother", "fujitsu",
                           "government", "uidai", "wkhtmltopdf", "itext", "reportlab",
                           "fpdf", "digilocker", "nsdl"],
    },
    "pan_card": {
        "regex": r"\b[A-Z]{5}\d{4}[A-Z]\b",
        "keywords": ["permanent account number", "income tax", "pan", "govt. of india",
                      "govt of india"],
        "expected_tools": ["scanner", "epson", "canon", "hp", "brother", "fujitsu",
                           "government", "nsdl", "utiitsl", "wkhtmltopdf", "itext",
                           "digilocker"],
    },
    "driving_license": {
        "regex": r"\b[A-Z]{2}\d{2}\s?\d{4}\s?\d{7}\b",
        "keywords": ["driving licence", "driving license", "motor vehicle",
                      "transport", "rto", "valid from", "valid till"],
        "expected_tools": ["scanner", "government", "transport", "wkhtmltopdf",
                           "itext", "mparivahan", "vahan", "digilocker"],
    },
}


class MetadataAgent(BaseAgent):
    """Detects metadata-level anomalies and provenance inconsistencies."""

    def __init__(self):
        super().__init__(name="Metadata & Provenance Agent")

    def analyze(self, document: Document) -> AgentResult:
        start_time = time.time()
        all_findings = []
        scores = []

        # --- Creation Tool Analysis ---
        tool_score, tool_findings = self._analyze_creation_tool(document)
        all_findings.extend(tool_findings)
        scores.append(("tool", tool_score, 0.22))

        # --- Timestamp Analysis ---
        timestamp_score, timestamp_findings = self._analyze_timestamps(document)
        all_findings.extend(timestamp_findings)
        scores.append(("timestamp", timestamp_score, 0.18))

        # --- EXIF Consistency ---
        exif_score, exif_findings = self._analyze_exif_consistency(document)
        all_findings.extend(exif_findings)
        scores.append(("exif", exif_score, 0.18))

        # --- Font Embedding Analysis ---
        font_score, font_findings = self._analyze_fonts(document)
        all_findings.extend(font_findings)
        scores.append(("fonts", font_score, 0.12))

        # --- Metadata Completeness ---
        completeness_score, completeness_findings = self._analyze_completeness(document)
        all_findings.extend(completeness_findings)
        scores.append(("completeness", completeness_score, 0.15))

        # --- Indian Document Validation ---
        indian_score, indian_findings = self._analyze_indian_documents(document)
        all_findings.extend(indian_findings)
        scores.append(("indian_doc", indian_score, 0.15))

        # Weighted combination
        final_score = sum(s * w for _, s, w in scores)

        # === CEILING LOGIC ===
        # When critical checks (tool analysis, Indian document validation) detect
        # severe issues, they should dominate the final score.
        # A fake Aadhaar created with iOS Quartz cannot be redeemed by good timestamps.
        critical_scores = {name: s for name, s, _ in scores
                          if name in ("tool", "indian_doc")}
        worst_critical = min(critical_scores.values()) if critical_scores else 1.0

        if worst_critical <= 0.15:
            # Severe flag from tool or Indian doc check → hard cap
            final_score = min(final_score, 0.20)
        elif worst_critical <= 0.30:
            final_score = min(final_score, 0.35)
        elif worst_critical <= 0.50:
            final_score = min(final_score, 0.50)

        reliability = self._compute_reliability(document)
        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(f"Metadata agent complete: score={final_score:.3f}, "
                     f"reliability={reliability:.2f}, {len(all_findings)} findings, "
                     f"{elapsed_ms:.0f}ms")

        return AgentResult(
            agent_name=self.name,
            score=final_score,
            reliability_weight=reliability,
            findings=all_findings,
            processing_time_ms=elapsed_ms,
            details={
                "sub_scores": {name: score for name, score, _ in scores},
                "creation_tool": document.metadata.get("creation_tool"),
                "has_exif": bool(document.metadata.get("exif")),
            }
        )

    def _analyze_creation_tool(
        self, document: Document
    ) -> tuple[float, list[AgentFinding]]:
        """
        Check if the creation tool makes sense for this document type.
        E.g., an Aadhaar card created with Photoshop or iOS Quartz is highly suspicious.
        """
        findings = []
        tool = document.metadata.get("creation_tool", "") or ""
        tool_lower = tool.lower()
        producer = (document.metadata.get("producer") or "").lower()
        # Combine tool + producer for broader matching
        combined = f"{tool_lower} {producer}"

        if not tool:
            # Missing tool is mildly suspicious for PDFs, neutral for images
            is_image = document.metadata.get("source") == "image"
            return (0.75 if is_image else 0.70), findings

        # Classify the tool
        is_editing_tool = any(t in combined for t in EDITING_TOOLS)
        is_consumer_tool = any(t in combined for t in CONSUMER_TOOLS)
        is_mobile_device = any(t in combined for t in MOBILE_DEVICE_INDICATORS)
        is_scan_tool = any(t in combined for t in LEGITIMATE_SCAN_TOOLS)
        is_legitimate_pdf = any(t in combined for t in LEGITIMATE_PDF_CREATORS)
        is_govt_tool = any(t in combined for t in GOVERNMENT_TOOLS)

        # Check if document content suggests official/government document
        full_text = " ".join(document.ocr_text).lower()
        is_official = any(ind in full_text for ind in OFFICIAL_DOC_INDICATORS)

        # === Priority 1: Known government tools → always legitimate ===
        if is_govt_tool:
            return 0.95, findings

        # === Priority 2: Known legitimate tools (office, scanner) ===
        if is_scan_tool or is_legitimate_pdf:
            return 0.95, findings

        # === Priority 3: EDITING TOOL → suspicious, severity depends on context ===
        if is_editing_tool:
            if is_official:
                findings.append(AgentFinding(
                    description=(
                        f"Official/government document created with editing software "
                        f"'{tool}' — government documents are never issued from "
                        f"image editing tools. Strong forgery indicator."
                    ),
                    severity=0.9,
                ))
                return 0.1, findings
            else:
                findings.append(AgentFinding(
                    description=(
                        f"Document created with image editing software: '{tool}'. "
                        f"This may indicate document manipulation."
                    ),
                    severity=0.5,
                ))
                return 0.30, findings

        # === Priority 4: Consumer/mobile tool on official doc → suspicious ===
        if is_consumer_tool or is_mobile_device:
            if is_official:
                tool_type = "mobile device" if is_mobile_device else "consumer PDF tool"
                findings.append(AgentFinding(
                    description=(
                        f"Official/government document created with {tool_type} "
                        f"'{tool}'. Government IDs (Aadhaar, PAN, DL) are issued "
                        f"by specialized government systems, never by consumer "
                        f"software or mobile devices."
                    ),
                    severity=0.85,
                ))
                return 0.15, findings
            else:
                # Consumer tool on non-official doc → mildly suspicious but not damning
                findings.append(AgentFinding(
                    description=(
                        f"Document created with consumer tool: '{tool}'."
                    ),
                    severity=0.2,
                ))
                return 0.75, findings

        # === Priority 5: Unknown tool → NEUTRAL, not suspicious ===
        # Evidence-based: an unknown tool with consistent visual/text/metadata
        # should not be penalized. Only flag when OTHER agents find issues.
        if is_official:
            findings.append(AgentFinding(
                description=(
                    f"Official document created with unrecognized tool: '{tool}'. "
                    f"Not a known government tool, but not a known editing tool either."
                ),
                severity=0.25,
            ))
            return 0.65, findings

        # Non-official doc with unknown tool → neutral
        return 0.75, findings

    def _analyze_timestamps(
        self, document: Document
    ) -> tuple[float, list[AgentFinding]]:
        """
        Analyze creation and modification timestamps for anomalies.
        """
        findings = []
        creation = document.metadata.get("creation_date")
        modification = document.metadata.get("modification_date")

        if not creation and not modification:
            return 0.7, findings  # Missing timestamps is mildly suspicious

        # Parse dates
        creation_dt = self._parse_pdf_date(creation) if creation else None
        modification_dt = self._parse_pdf_date(modification) if modification else None

        if creation_dt and modification_dt:
            # Modification before creation is physically impossible
            if modification_dt < creation_dt:
                findings.append(AgentFinding(
                    description=(
                        f"Modification date ({modification_dt}) is before creation date "
                        f"({creation_dt}) — impossible without tampering"
                    ),
                    severity=0.9,
                ))
                return 0.1, findings

            # Very rapid modification after creation (seconds) is suspicious
            # for scanned documents
            time_diff = (modification_dt - creation_dt).total_seconds()
            if time_diff > 0 and time_diff < 60:
                # This might be normal for programmatically created docs
                pass
            elif time_diff > 3600 * 12:
                # Modified 12+ hours after creation
                findings.append(AgentFinding(
                    description=(
                        f"Document modified {time_diff/3600:.0f} hours after creation. "
                        f"Created: {creation_dt}, Modified: {modification_dt}"
                    ),
                    severity=0.4,
                ))
                return 0.7, findings

        # Check for future dates
        now = datetime.now()
        if creation_dt and creation_dt > now:
            findings.append(AgentFinding(
                description=f"Creation date is in the future: {creation_dt}",
                severity=0.9,
            ))
            return 0.1, findings

        return 1.0, findings

    def _analyze_exif_consistency(
        self, document: Document
    ) -> tuple[float, list[AgentFinding]]:
        """
        Analyze EXIF data for inconsistencies.
        Scanned documents that claim to have camera EXIF are suspicious.
        Documents that look like scans but have no EXIF are also checked.
        WhatsApp/messaging app images (no EXIF, JPEG, small, mobile dims) are
        handled gracefully — missing metadata is expected, not suspicious.
        """
        findings = []
        exif = document.metadata.get("exif", {})
        source = document.metadata.get("source", "")

        if source == "image":
            if not exif:
                # Check if this looks like a WhatsApp/messaging app image
                if self._is_messaging_app_image(document):
                    findings.append(AgentFinding(
                        description=(
                            "Image appears to be shared via WhatsApp or similar "
                            "messaging app (no EXIF, JPEG, small file size, mobile "
                            "dimensions). Missing metadata is expected — not a "
                            "forgery indicator."
                        ),
                        severity=0.05,
                    ))
                    return 0.95, findings

                # Image with no EXIF — might be stripped (common in forgery)
                tool = document.metadata.get("creation_tool", "")
                if tool:
                    findings.append(AgentFinding(
                        description=(
                            f"Image has creation tool '{tool}' but no EXIF data. "
                            f"EXIF may have been stripped."
                        ),
                        severity=0.3,
                    ))
                    return 0.70, findings
                # No EXIF and no tool — suspicious for non-WhatsApp images
                findings.append(AgentFinding(
                    description="Image has no EXIF data and no creation tool metadata.",
                    severity=0.2,
                ))
                return 0.75, findings

            # Check for inconsistent EXIF
            if "Make" in exif and "Model" in exif:
                # Has camera info — check if it's consistent with a scan
                make = exif.get("Make", "").lower()
                model = exif.get("Model", "").lower()
                tool = (document.metadata.get("creation_tool") or "").lower()

                if any(t in tool for t in EDITING_TOOLS):
                    findings.append(AgentFinding(
                        description=(
                            f"EXIF shows camera '{make} {model}' but document was "
                            f"created/edited with '{tool}'. Possible metadata mismatch."
                        ),
                        severity=0.5,
                    ))
                    return 0.6, findings

        elif source == "pdf":
            # For PDFs, check if metadata suggests scan but has no camera data
            tool = (document.metadata.get("creation_tool") or "").lower()
            producer = (document.metadata.get("producer") or "").lower()

            # PDF claims to be from scanner but no scan-related metadata
            if any(t in tool for t in LEGITIMATE_SCAN_TOOLS):
                # This is expected — scanner-created PDFs are fine
                return 1.0, findings

        return 1.0, findings

    def _analyze_fonts(
        self, document: Document
    ) -> tuple[float, list[AgentFinding]]:
        """
        Analyze embedded fonts for suspicious patterns.
        """
        findings = []
        fonts = document.metadata.get("embedded_fonts", [])

        if not fonts:
            return 1.0, findings  # No fonts to analyze (image-based doc)

        # Clean font names
        clean_fonts = [f.lstrip("/").split("+")[-1] if "+" in f else f.lstrip("/")
                       for f in fonts]

        # Check for unusual font mixing
        if len(clean_fonts) > 5:
            findings.append(AgentFinding(
                description=(
                    f"Document uses {len(clean_fonts)} different fonts: "
                    f"{', '.join(clean_fonts[:5])}{'...' if len(clean_fonts) > 5 else ''}. "
                    f"Unusually high font variety may indicate content from multiple sources."
                ),
                severity=0.3,
            ))
            return 0.8, findings

        # Check for subset fonts (font name contains '+') — very common in legitimate PDFs
        # but having mixed subset and non-subset is suspicious
        subset_fonts = [f for f in fonts if "+" in f]
        non_subset = [f for f in fonts if "+" not in f]

        if subset_fonts and non_subset and len(fonts) > 2:
            findings.append(AgentFinding(
                description=(
                    f"Mix of subset ({len(subset_fonts)}) and full ({len(non_subset)}) "
                    f"font embeddings detected. May indicate merged documents."
                ),
                severity=0.2,
            ))
            return 0.85, findings

        return 1.0, findings

    def _analyze_completeness(
        self, document: Document
    ) -> tuple[float, list[AgentFinding]]:
        """
        Score metadata completeness. Legitimate documents typically have
        consistent, complete metadata. Forged documents often have gaps.
        """
        findings = []
        expected_fields = [
            "creation_tool", "creator", "creation_date",
            "modification_date", "producer",
        ]

        if document.metadata.get("source") == "pdf":
            expected_fields.extend(["page_count", "embedded_fonts"])

        present = 0
        total = len(expected_fields)

        for field in expected_fields:
            value = document.metadata.get(field)
            if value and (not isinstance(value, list) or len(value) > 0):
                present += 1

        completeness = present / total if total > 0 else 0

        if completeness < 0.4:
            findings.append(AgentFinding(
                description=(
                    f"Metadata completeness: {present}/{total} expected fields present "
                    f"({completeness*100:.0f}%). Low metadata completeness is suspicious."
                ),
                severity=0.4,
            ))
            return 0.6, findings
        elif completeness < 0.7:
            findings.append(AgentFinding(
                description=(
                    f"Partial metadata: {present}/{total} expected fields present "
                    f"({completeness*100:.0f}%)"
                ),
                severity=0.2,
            ))
            return 0.8, findings

        return 1.0, findings

    def _analyze_indian_documents(
        self, document: Document
    ) -> tuple[float, list[AgentFinding]]:
        """
        Validate Indian government documents (Aadhaar, PAN, Driving License).
        Cross-checks detected document type with creation tool and ID format.
        """
        findings = []
        full_text = " ".join(document.ocr_text).lower()
        full_text_raw = " ".join(document.ocr_text)
        tool = (document.metadata.get("creation_tool") or "").lower()

        detected_type = None
        for doc_type, patterns in INDIAN_DOC_PATTERNS.items():
            keyword_hits = sum(1 for kw in patterns["keywords"] if kw in full_text)
            if keyword_hits >= 2:
                detected_type = doc_type
                break

        if detected_type is None:
            return 1.0, findings

        pattern_info = INDIAN_DOC_PATTERNS[detected_type]
        score = 1.0

        # Validate ID number format
        id_matches = re.findall(pattern_info["regex"], full_text_raw)
        if not id_matches:
            findings.append(AgentFinding(
                description=(
                    f"Detected {detected_type.replace('_', ' ').title()} document but "
                    f"no valid ID number pattern found"
                ),
                severity=0.6,
            ))
            score = min(score, 0.5)

        # Validate Aadhaar checksum (Verhoeff algorithm) if found
        # Note: OCR errors on real e-Aadhaars can cause false Verhoeff failures.
        # When the tool is a known government tool, the penalty should be mild.
        # Masked Aadhaar numbers (XXXX XXXX 2860) skip checksum entirely.
        if detected_type == "aadhaar" and id_matches:
            aadhaar_num = id_matches[0].replace(" ", "")
            is_masked = any(c in aadhaar_num.upper() for c in "X")
            if is_masked:
                pass  # Masked Aadhaar (standard e-Aadhaar format) — no checksum possible
            elif not self._validate_aadhaar_checksum(aadhaar_num):
                producer = (document.metadata.get("producer") or "").lower()
                combined_tool = f"{tool} {producer}"
                tool_is_govt = any(t in combined_tool
                                   for t in pattern_info["expected_tools"])
                tool_is_govt = tool_is_govt or any(
                    t in combined_tool for t in GOVERNMENT_TOOLS)
                if tool_is_govt:
                    # Government tool + Verhoeff fail = likely OCR error, mild penalty
                    findings.append(AgentFinding(
                        description=(
                            f"Aadhaar number {aadhaar_num[:4]}XXXX{aadhaar_num[-4:]} "
                            f"fails Verhoeff checksum — likely OCR error on "
                            f"government-issued document"
                        ),
                        severity=0.2,
                    ))
                    score = min(score, 0.85)
                else:
                    # Non-government tool + Verhoeff fail = strong forgery signal
                    findings.append(AgentFinding(
                        description=(
                            f"Aadhaar number {aadhaar_num[:4]}XXXX{aadhaar_num[-4:]} "
                            f"fails Verhoeff checksum validation"
                        ),
                        severity=0.8,
                    ))
                    score = min(score, 0.3)

        # Cross-check creation tool — Indian govt docs should NOT be from editing/consumer software
        if tool:
            producer = (document.metadata.get("producer") or "").lower()
            combined = f"{tool} {producer}"
            is_editing = any(t in combined for t in EDITING_TOOLS)
            is_consumer = any(t in combined for t in CONSUMER_TOOLS)
            is_mobile = any(t in combined for t in MOBILE_DEVICE_INDICATORS)
            is_govt = any(t in combined for t in GOVERNMENT_TOOLS)
            is_expected = any(t in combined for t in pattern_info["expected_tools"])

            if is_govt or is_expected:
                # Known government tool or expected tool → legitimate
                pass  # score stays at current (high) value
            elif is_editing:
                findings.append(AgentFinding(
                    description=(
                        f"{detected_type.replace('_', ' ').title()} created with "
                        f"editing software '{tool}' — Indian government documents "
                        f"are never issued from editing software. STRONG FORGERY INDICATOR."
                    ),
                    severity=0.95,
                ))
                score = min(score, 0.10)
            elif is_consumer or is_mobile:
                tool_kind = "mobile device" if is_mobile else "consumer PDF/image tool"
                findings.append(AgentFinding(
                    description=(
                        f"{detected_type.replace('_', ' ').title()} created with "
                        f"{tool_kind} '{tool}' — Indian government IDs are issued by "
                        f"UIDAI/NSDL/RTO systems, never by consumer software."
                    ),
                    severity=0.9,
                ))
                score = min(score, 0.12)
            else:
                # Unknown tool on govt doc — mildly suspicious, not damning
                findings.append(AgentFinding(
                    description=(
                        f"{detected_type.replace('_', ' ').title()} created with "
                        f"unrecognized tool: '{tool}'. Not a known government tool."
                    ),
                    severity=0.3,
                ))
                score = min(score, 0.7)

        return score, findings

    def _is_messaging_app_image(self, document: Document) -> bool:
        """
        Detect if an image was likely shared via WhatsApp or similar messaging apps.
        WhatsApp strips all EXIF, re-encodes as JPEG with aggressive compression,
        and resizes to mobile-friendly dimensions.
        """
        meta = document.metadata
        if meta.get("source") != "image":
            return False

        fmt = (meta.get("format") or "").upper()
        original = document.original_format.lower()
        if fmt not in ("JPEG", "JPG") and original not in ("jpg", "jpeg"):
            return False

        # No EXIF data
        exif = meta.get("exif", {})
        if exif:
            return False

        # No creation tool
        if meta.get("creation_tool"):
            return False

        # Small file size (WhatsApp aggressive compression)
        if document.file_size_bytes > 500_000:
            return False

        # Mobile-typical dimensions
        w = meta.get("width", 0)
        h = meta.get("height", 0)
        if w > 0 and h > 0:
            if max(w, h) > 1920:
                return False
            if min(w, h) < 200:
                return False

        return True

    def _validate_aadhaar_checksum(self, number: str) -> bool:
        """Validate Aadhaar number using Verhoeff algorithm."""
        if len(number) != 12 or not number.isdigit():
            return False

        verhoeff_table_d = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
            [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
            [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
            [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
            [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
            [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
            [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
            [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ]
        verhoeff_table_p = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
            [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
            [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
            [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
            [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
            [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
            [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
        ]

        c = 0
        reversed_digits = [int(d) for d in reversed(number)]
        for i, digit in enumerate(reversed_digits):
            c = verhoeff_table_d[c][verhoeff_table_p[i % 8][digit]]
        return c == 0

    def _parse_pdf_date(self, date_str: str) -> Optional[datetime]:
        """Parse PDF date format (D:YYYYMMDDHHmmSS) to datetime."""
        if not date_str:
            return None

        # Remove D: prefix
        date_str = date_str.replace("D:", "").strip("'\"")

        # Try various formats
        formats = [
            "%Y%m%d%H%M%S",
            "%Y%m%d%H%M",
            "%Y%m%d",
            "%Y-%m-%dT%H:%M:%S",
        ]

        # Remove timezone info for parsing
        date_str = re.sub(r"[+-]\d{2}'?\d{2}'?$", "", date_str)
        date_str = date_str.rstrip("Z")

        for fmt in formats:
            try:
                return datetime.strptime(date_str[:len(fmt.replace("%", ""))], fmt)
            except (ValueError, IndexError):
                continue

        return None

    def _compute_reliability(self, document: Document) -> float:
        """
        Metadata agent reliability depends on how much metadata is available.
        """
        meta = document.metadata

        # Count available metadata sources
        sources = 0
        if meta.get("creation_tool"):
            sources += 1
        if meta.get("creation_date"):
            sources += 1
        if meta.get("modification_date"):
            sources += 1
        if meta.get("exif"):
            sources += 1
        if meta.get("embedded_fonts"):
            sources += 1
        if meta.get("producer"):
            sources += 1

        # More metadata sources = more reliable analysis
        # But even with limited metadata, absence itself is informative
        return min(1.0, 0.4 + sources * 0.1)
