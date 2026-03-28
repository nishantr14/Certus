"""
CertusDoc FastAPI REST Endpoint

POST /analyze — accepts a file upload, returns JSON with DIS score,
risk level, and per-agent results.
"""
import os
import io
import base64
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from certusdoc.pipeline import CertusDocPipeline
from certusdoc.report.generator import generate_report

app = FastAPI(
    title="CertusDoc API",
    description="Multi-Agent Document Forgery Detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline once at startup
pipeline: CertusDocPipeline | None = None
last_report: "ForensicReport | None" = None


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
        global last_report
        start = time.time()
        report = pipeline.analyze(tmp_path)
        last_report = report
        elapsed = time.time() - start

        # Encode heatmap as base64 PNG if available
        heatmap_b64 = None
        if report.fused_heatmap is not None:
            heatmap_img = report.fused_heatmap
            if heatmap_img.dtype != np.uint8:
                heatmap_img = (np.clip(heatmap_img, 0, 1) * 255).astype(np.uint8)
            colored = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
            _, buf = cv2.imencode('.png', colored)
            heatmap_b64 = base64.b64encode(buf).decode('utf-8')

        # Encode original page thumbnail as base64
        original_b64 = None
        if report.document.pages:
            page0 = report.document.pages[0]
            h, w = page0.shape[:2]
            scale = min(600 / w, 600 / h, 1.0)
            thumb = cv2.resize(page0, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            original_b64 = base64.b64encode(buf).decode('utf-8')

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
            "heatmap_base64": heatmap_b64,
            "original_base64": original_b64,
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
        global last_report
        report = pipeline.analyze(tmp_path)
        last_report = report
        pdf_bytes = generate_report(report)

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


@app.get("/report/last")
async def get_last_report():
    """
    Generate a PDF report from the most recent analysis (no re-processing).
    """
    if last_report is None:
        raise HTTPException(status_code=404, detail="No analysis has been run yet")

    try:
        pdf_bytes = generate_report(last_report)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=certusdoc_report_{last_report.document.file_name}.pdf"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": pipeline is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
