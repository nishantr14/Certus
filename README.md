# CertusDoc 🔍

**Multi-Agent Document Forgery Detection System**

Team ByteMe — Secure AI Hackathon 2026 (IIT Madras × BITS Pilani Goa)

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd certusdoc
pip install -r requirements.txt

# 2. Install system dependencies
sudo apt-get install tesseract-ocr poppler-utils

# 3. Test the pipeline
python -m tests.test_pipeline path/to/document.pdf

# 4. Launch dashboard
streamlit run app.py
```

## Architecture

```
Input Document → Ingestion (OCR + metadata)
                      ↓
    ┌─────────────────┼─────────────────┐
    │                 │                 │
Visual Agent    Text Agent     Metadata Agent
 (ELA, TruFor)  (OCR forensics)  (EXIF, PDF meta)
    │                 │                 │
    └─────────────────┼─────────────────┘
                      ↓
           Weighted Trust Fusion
           DIS = Σ(Rᵢ × Sᵢ) / ΣRᵢ
                      ↓
        Forensic Report + Dashboard
```

## Detection Capabilities

| Attack Type | Detection Method | Agent |
|---|---|---|
| JPEG recompression | ELA + quantization analysis | Visual |
| Copy-move | Noise consistency analysis | Visual |
| Image splicing | TruFor Noiseprint++ | Visual |
| Font substitution | Font size/spacing analysis | Text |
| Text replacement | OCR confidence variance | Text |
| Baseline drift | Alignment analysis | Text |
| Metadata spoofing | Tool signature + timestamp | Metadata |
| EXIF stripping | EXIF completeness check | Metadata |

## Project Structure

```
certusdoc/
├── app.py                    # Streamlit dashboard
├── certusdoc/
│   ├── pipeline.py           # Main orchestrator
│   ├── models.py             # Data models
│   ├── ingestion/ingest.py   # Document intake
│   ├── agents/
│   │   ├── visual_agent.py   # ELA + TruFor
│   │   ├── text_agent.py     # OCR forensics
│   │   └── metadata_agent.py # Metadata analysis
│   └── fusion/engine.py      # DIS computation
└── tests/
```
