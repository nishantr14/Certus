"""
CertusDoc FastAPI REST Endpoint

POST /analyze — accepts a file upload, returns JSON with DIS score,
risk level, and per-agent results.
"""
import os
import time
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from certusdoc.pipeline import CertusDocPipeline
from certusdoc.report.generator import generate_report

app = FastAPI(
    title="CertusDoc API",
    description="Multi-Agent Document Forgery Detection",
    version="1.0.0",
)

# Initialize pipeline once at startup
pipeline: CertusDocPipeline | None = None


@app.on_event("startup")
def startup():
    global pipeline
    pipeline = CertusDocPipeline()


@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Analyze a document for forgery.

    Accepts: PDF, PNG, JPG, JPEG, TIFF, BMP
    Returns: JSON with DIS score, risk level, per-agent results.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    suffix = Path(file.filename or "upload").suffix.lower()
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {suffix}. Allowed: {', '.join(allowed)}",
        )

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        start = time.time()
        report = pipeline.analyze(tmp_path)
        elapsed = time.time() - start

        result = {
            "file_name": report.document.file_name,
            "file_size_bytes": report.document.file_size_bytes,
            "pages": len(report.document.pages),
            "format": report.document.original_format,
            "dis_score": round(report.dis_score, 4),
            "risk_level": report.risk_level.value,
            "forgery_type": report.primary_forgery_type.value,
            "recommended_action": report.recommended_action,
            "processing_time_ms": round(report.processing_time_ms, 0),
            "agents": [
                {
                    "name": r.agent_name,
                    "score": round(r.score, 4),
                    "reliability_weight": round(r.reliability_weight, 4),
                    "findings": [
                        {
                            "description": f.description,
                            "severity": round(f.severity, 3),
                            "page": f.page,
                            "region": f.region,
                        }
                        for f in r.findings
                    ],
                    "processing_time_ms": round(r.processing_time_ms, 0),
                }
                for r in report.agent_results
            ],
        }

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/analyze/report")
async def analyze_and_report(file: UploadFile = File(...)):
    """
    Analyze a document and return a forensic PDF report.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    suffix = Path(file.filename or "upload").suffix.lower()
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        report = pipeline.analyze(tmp_path)
        pdf_bytes = generate_report(report)

        import io
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=certusdoc_report_{report.document.file_name}.pdf"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": pipeline is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
