# CertusDoc — System Report
**Team ByteMe · Livana Datta, Nishant Rajpathak · v6.0-Sentinel**

---

## 1. What the System Does — End to End

CertusDoc is a multi-agent document forgery detection pipeline. When you upload a file, it runs through **four sequential stages**:

```
File Upload → Stage 1: Ingestion → Stage 2: Detection (3 agents, parallel)
→ Stage 3: Fusion → Stage 4: Report
```

---

## 2. Stage-by-Stage Breakdown

### Stage 1 — Ingestion (`certusdoc/ingestion/ingest.py`)

| Step | What happens |
|---|---|
| File type detection | Detects PDF vs image (JPEG/PNG/TIFF/BMP/WEBP) |
| PDF rendering | **Poppler 24.08** converts each PDF page to a 300 DPI image |
| OCR | **Tesseract v5.4** runs on each page image, producing text + word-level bounding boxes + per-word confidence scores |
| Metadata extraction | EXIF, PDF metadata (creator, producer, dates), file size, dimensions, format |
| Output | A `Document` object: pages (images), OCR text, word data, confidence scores, metadata dict |

**What can go wrong here:** Low-quality scans or very small images produce poor OCR confidence, which reduces the Text Agent's reliability weight, making its findings less impactful.

---

### Stage 2 — Detection (3 Agents in parallel)

#### Agent 1: Visual Tamper Agent (`certusdoc/agents/visual_agent.py`)

Runs **5 independent visual forensics methods** per page:

| Method | What it detects | Weight (no GPU) |
|---|---|---|
| **Multi-Scale ELA** (Q90 / Q75 / Q50) | JPEG recompression artifacts — areas that were edited then re-saved show different compression signatures | 30% |
| **ORB Copy-Move Detection** | Finds duplicate pixel regions — used when someone copies a section and pastes it elsewhere | 30% |
| **JPEG Quantization Analysis** | Inspects the JPEG quantization table — double-compressed images (edit → save → WhatsApp etc.) have telltale patterns | 20% |
| **Noise Consistency Maps** | Camera sensor noise should be uniform across an authentic image — localized noise inconsistencies indicate inserted regions | 20% |
| **ManTraNet v4** (GPU only) | Deep learning pixel-level forgery localization (DISABLED on CPU — takes 4+ min) | 50% when active |

**Score = min(weighted_avg, worst_sub_score + 0.20)**
The `+0.20 ceiling` means: if any single method scores 0.10, the page score can't exceed 0.30 regardless of how clean the other methods look.

**Reliability** is computed from image resolution and sharpness — blurry/low-res images reduce the agent's weight in the final score.

---

#### Agent 2: Text Forensics Agent (`certusdoc/agents/text_agent.py`)

Runs **7 statistical OCR-based checks** per page:

| Method | What it detects | Weight |
|---|---|---|
| OCR Confidence Variance | Forged text regions (pasted from different source) often have different OCR confidence than surrounding text | 22% |
| Regional Confidence Comparison | Splits page into regions, compares confidence across regions | 17% |
| Baseline Alignment | Text baselines should be horizontally consistent — cut-and-paste words sit slightly off | 18% |
| Character Spacing | Kerning anomalies from mixing text of different origins | 13% |
| Font Size Consistency | Mixed font sizes in same line/paragraph suggest pasted content | 10% |
| Text Block Regularity | Text block spacing and layout consistency | 10% |
| Text Sharpness Consistency | Sharpness should be uniform — sharper inserted text on a blurry scan is visible | 10% |

**Reliability** = f(average OCR confidence). If Tesseract is struggling (< 60% confidence), this agent's findings are down-weighted in fusion.

---

#### Agent 3: Metadata & Provenance Agent (`certusdoc/agents/metadata_agent.py`)

Runs **7 rule-based and heuristic checks**:

| Check | What it catches | Weight |
|---|---|---|
| Creation Tool Analysis | Photoshop/GIMP/Canva on an Aadhaar → flagged. wkhtmltopdf/DigiLocker → confirmed legit | 22% |
| EXIF Consistency | WhatsApp images: no EXIF + JPEG + small file → unverifiable (0.30 for govt docs, 0.50 for others). EXIF-stripped non-WhatsApp images flagged. | 18% |
| Timestamp Analysis | Modification before creation → impossible. Future creation dates → flagged. | 18% |
| Indian Doc Validation | Regex for Aadhaar, PAN, DL numbers. Verhoeff checksum on Aadhaar UID. Tool cross-check against expected govt tools. | 13% |
| Metadata Completeness | Missing creation tool + dates + EXIF = suspicious gap profile | 15% |
| Font Embedding | >5 fonts or mixed subset/non-subset embeddings | 12% |
| QR Code Verification | pyzbar scans for QR codes. Aadhaar QR XML validated against OCR text (name, DOB, gender, address) | 2% |

**Hard ceiling logic:** If the tool check or Indian doc check scores ≤ 0.15 (e.g., Canva/Photoshop on an Aadhaar), the total metadata score is capped at 0.20 regardless of how clean other metadata looks.

**Reliability** = f(number of metadata fields present). Image-only docs have very little metadata, so reliability is low (~0.4).

---

### Stage 3 — Evidence-Based Fusion (`certusdoc/fusion/engine.py`)

**Formula:** `DIS = Σ(Rᵢ × Sᵢ) / ΣRᵢ`

But this is not a simple average — extensive rules are applied on top:

| Rule | Effect |
|---|---|
| **Hard DIS ceiling** | If ≥1 agent scores < 0.3 (severe), DIS capped at 0.35. If ≥2 agents score < 0.6, DIS capped at 0.35. |
| **Convergence penalty** | If ≥2 agents independently flag issues: additional −0.10 penalty |
| **Cross-agent trust** | If metadata confirms legit govt provenance (≥ 0.90), visual ELA artifacts are softened (wkhtmltopdf PDFs trigger ELA false positives) |
| **WhatsApp cap** | WhatsApp-detected images get DIS capped at 0.55 regardless of how clean they look visually |
| **Evidence-based boosts** | QR verified + wkhtmltopdf tool → small authenticity boost (+0.02 to +0.05) |
| **Consumer tool on govt doc** | DIS hard-capped at 0.25 |
| **MantraNet strong forgery signal** | DIS capped at 0.25 (unless metadata strongly confirms legitimacy) |

**DIS Output Bands:**

| DIS Range | Risk Level | Action |
|---|---|---|
| 0.00–0.20 | HIGH RISK | Flag immediately, escalate |
| 0.21–0.45 | ELEVATED RISK | Human review required |
| 0.46–0.70 | UNCERTAIN | Cross-reference with issuing authority |
| 0.71–0.89 | LOW RISK | Likely authentic |
| 0.90–1.00 | TRUSTED | High confidence, cleared |

---

### Stage 4 — Report Generation (`certusdoc/report/generator.py`)

- Forensic PDF report with per-agent findings and ELA heatmap
- JSON response to frontend with all scores, findings, processing time
- Frontend renders interactive results with animated gauge, agent cards, heatmap

---

## 3. Estimated Accuracy (Honest Assessment)

> **Important caveat:** No formal test set has been run. These estimates are based on observed behavior and system design analysis.

### What the system is good at (High confidence ~85-95%):

| Scenario | Why it works well |
|---|---|
| **Canva/Photoshop Aadhaar fakes** | Creation tool is in EXIF — instantly flagged. DIS < 0.20. |
| **Digitally generated Aadhaar** (no real print) | No QR code, wrong tool, font inconsistencies. DIS ≈ 0.10–0.30 |
| **Metadata-stripped fakes** (EXIF scrubbed) | Missing EXIF + no tool + govt doc → score drops. |
| **PDF timestamp manipulation** | Modification before creation is impossible. Immediately caught. |
| **Documents with invalid Aadhaar UID** | Verhoeff checksum fails → flagged |

### What the system struggles with (~60-70% confidence):

| Scenario | Why it's hard |
|---|---|
| **Genuine scanned WhatsApp Aadhaar** (authentic doc, forwarded) | WhatsApp strips all EXIF. No QR scannable at low res. Could score 0.30 (false positive). This is by design — WhatsApp = broken provenance. |
| **High-quality print + re-scan fakes** | Printing and scanning erases digital artifacts. ELA may not detect. Needs Verhoeff + QR for distinction. |
| **AI-generated photo-realistic documents** (Stable Diffusion, etc.) | No ELA artifacts, no metadata trail. MantraNet (GPU) would help most here. Without it, only text/metadata can flag. |
| **Documents in non-standard formats** | Hindi-only Aadhaar, regional language documents — OCR confidence drops, text agent becomes unreliable. |
| **Legitimate documents with heavy lossy compression** | Over-compressed JPEGs trigger ELA false positives — the system tries to correct with cross-agent trust but isn't perfect. |

### Overall estimated accuracy (no formal benchmark):

| Mode | Estimated Detection Rate | False Positive Rate |
|---|---|---|
| CPU-only (current) | ~75–80% | ~15–20% |
| With MantraNet GPU enabled | ~85–90% | ~10–15% |
| With QR scanning (pyzbar working) | +3-5% on Aadhaar cards | — |

> These are rough estimates. Industry-grade systems like UIDAI's own verifier achieve >95% but have access to ground-truth databases. CertusDoc operates entirely offline without any database lookup.

---

## 4. What Should Be Improved (Priority Order)

### 🔴 Critical — High impact, achievable now

#### 4.1 Fix pyzbar on Windows (QR code scanning is disabled)
The QR scan is currently disabled because `libzbar.dll` is missing. Installing it would enable:
- Aadhaar QR XML cross-validation against visible OCR text
- Estimated +5-10% accuracy improvement on Aadhaar cards specifically
- **Fix:** Install `zbar` binary for Windows (`choco install zbar` or download `libzbar-64.dll`)

#### 4.2 The `_is_messaging_app_image` size threshold is too loose (500KB)
Many real WhatsApp images are larger — documents compressed on newer phones can be 400-600KB. The threshold should be raised to 1MB.

#### 4.3 ELA false positives on wkhtmltopdf PDFs
Legitimate e-Aadhaar PDFs (downloaded from UIDAI) trigger ELA artifacts because wkhtmltopdf renders with specific JPEG settings. The cross-agent trust rule partially handles this, but a document-type-specific ELA threshold would be cleaner.

---

### 🟡 Important — Significant improvement, moderate effort

#### 4.4 Add a Print-Scan Detection Module
Currently there is no dedicated print-scan attack detector. A document can be:
1. Stolen (authentic digital Aadhaar)
2. Edited in Photoshop
3. Printed, then scanned
4. → Defeats ELA (print erases compression history) and metadata (scanner EXIF looks legit)

**Solution:** Detect halftone patterns (characteristic of printed documents), ink bleed artifacts, and scan-specific frequency domain signatures.

#### 4.5 GPU / Async MantraNet for faster deep-learning inference
MantraNet on CPU takes 4+ minutes per image. On a GPU (even a laptop GPU), it runs in ~2-3 seconds. This is the single biggest accuracy improvement available.

#### 4.6 Calibrate confidence thresholds per document type
The system uses global thresholds. An Aadhaar card should have much tighter thresholds than a general business letter. A `DocType`-aware threshold table would reduce false positives significantly.

---

### 🟢 Nice to Have — Long-term improvements

#### 4.7 UIDAI Offline Verification API
UIDAI has an authorized offline XML verification mechanism. If the document contains a QR with signed XML, the signature can be validated cryptographically without any database lookup. This would be **ground truth** for Aadhaar verification.

#### 4.8 Expand Indian document patterns
Currently only Aadhaar, PAN and Driving License are specifically validated. Adding:
- Passport (MRZ checksum validation)
- Voter ID (EPIC number format)
- Bank statements (IFSC + account number pattern checks)

#### 4.9 Formal benchmark dataset
Run the system against a labeled dataset (e.g., CDIP, MIDV-2020, or internally collected authentic/forged Aadhaar pairs) to get real accuracy numbers. Right now all accuracy figures are estimates.

#### 4.10 Add a Provenance Confidence field to the output
Currently the report shows `DIS score` and `risk level`. A separate `Provenance Confidence` field (derived from how much verifiable metadata exists) would help users understand whether a high score comes from strong evidence or just absence of red flags.

---

## 5. Known Bugs / Current Limitations

| Issue | Status | Impact |
|---|---|---|
| pyzbar `libzbar.dll` missing on Windows | Open | QR verification completely disabled |
| MantraNet disabled on CPU | By design (performance) | ~10% accuracy reduction on complex fakes |
| WhatsApp images ≤ 500KB only detected as WhatsApp | Minor | Some large WhatsApp images not handled |
| ELA false positives on wkhtmltopdf PDFs | Partially mitigated | Can produce uncertain scores on real e-Aadhaar |
| No Hindi/regional-script OCR tuning | Open | Lower accuracy on Hindi-only documents |
| Visual agent 70s+ analysis time on large PDFs | Open | Slow on some PDFs even without MantraNet |

---

## 6. Architecture Summary

```
certusdoc/
├── ingestion/ingest.py       # Poppler PDF rendering + Tesseract OCR
├── agents/
│   ├── visual_agent.py       # ELA + ORB + JPEG quant + noise + MantraNet
│   ├── text_agent.py         # OCR forensics (7 checks)
│   └── metadata_agent.py     # EXIF + tool + QR + Verhoeff checksum
├── fusion/engine.py          # Evidence-based DIS computation
├── report/generator.py       # PDF + JSON output
├── utils/doc_detector.py     # Document type classification
└── models/mantranet/         # MantraNet v4 weights (GPU)
```

**Dependencies:** Python 3.13 · FastAPI · Uvicorn · OpenCV · pytesseract · Tesseract 5.4 · Poppler 24.08 · PyTorch · reportlab · loguru

