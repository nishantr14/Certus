# CertusDoc — Multi-Agent Document Forgery Detection System

## Project Overview
CertusDoc is a multi-agent document forgery detection pipeline for the Secure AI Hackathon 2026 (IIT Madras × BITS Pilani Goa). We are Blue Team. Red Team will attempt to forge documents to beat our system.

## Architecture (4-Stage Pipeline)
1. **Ingestion**: PDF/image → 300 DPI page extraction → Tesseract OCR → metadata extraction
2. **Detection Agents** (parallel execution):
   - Visual Tamper Agent: ELA + TruFor (pretrained) → anomaly heatmap + score
   - Text Forensics Agent: OCR confidence variance + font consistency + baseline alignment
   - Metadata Agent: PDF metadata rules + EXIF analysis + Isolation Forest anomaly scoring
3. **Weighted Trust Fusion**: DIS = Σ(Rᵢ × Sᵢ) / ΣRᵢ with dynamic reliability weights
4. **Output**: Forensic PDF report + Streamlit dashboard + FastAPI REST endpoint

## Tech Stack
- Python 3.10+
- PyTorch (inference only, no training)
- OpenCV, Pillow, scikit-image
- Tesseract OCR (pytesseract)
- scikit-learn (Isolation Forest)
- Streamlit (dashboard)
- FastAPI (REST API)
- reportlab (PDF reports)
- pikepdf / PyPDF2 (PDF metadata)

## Code Conventions
- Use readable, named-class patterns (no anonymous/lambda-heavy code)
- Type hints on all function signatures
- Docstrings on all public functions
- Each detection agent is a self-contained module with a common interface:
  ```python
  class Agent:
      def analyze(self, document: Document) -> AgentResult:
          ...
  ```
- AgentResult always includes: score (0-1, lower=more forged), reliability_weight (0-1), findings (list of strings), heatmap (optional numpy array)

## Directory Structure
```
certusdoc/
├── CLAUDE.md
├── requirements.txt
├── app.py                    # Streamlit dashboard
├── api.py                    # FastAPI REST endpoint
├── certusdoc/
│   ├── __init__.py
│   ├── pipeline.py           # Main orchestrator
│   ├── models.py             # Data models (Document, AgentResult, ForensicReport)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── ingest.py         # PDF/image intake, OCR, metadata extraction
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py           # Base agent interface
│   │   ├── visual_agent.py   # ELA + TruFor integration
│   │   ├── text_agent.py     # OCR + font forensics
│   │   └── metadata_agent.py # PDF metadata + EXIF analysis
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── engine.py         # DIS weighted fusion
│   └── report/
│       ├── __init__.py
│       └── generator.py      # Forensic PDF report + heatmap overlay
├── models/                   # Pretrained model weights (gitignored)
│   └── trufor/
├── tests/
│   ├── test_pipeline.py
│   └── test_agents.py
└── data/                     # Test documents (gitignored)
    ├── authentic/
    └── forged/
```

## Testing
- Run `pytest tests/` before any commit
- Test each agent independently against known forged/authentic pairs
- Always test the full pipeline end-to-end after changes

## Important Notes
- DO NOT train models from scratch — use pretrained weights for inference
- ELA is the fast baseline (milliseconds) — always run it
- TruFor is the heavy hitter — run via pretrained weights
- Metadata agent is pure Python, no ML dependencies
- Keep the Streamlit dashboard matching the mockup in CertusDoc_final.pdf page 4
