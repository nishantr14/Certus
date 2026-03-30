import traceback
try:
    from certusdoc.ingestion.ingest import _extract_pdf_pages
    from pathlib import Path
    _extract_pdf_pages(Path(r"c:\Users\LIVANA\Certus\test_report.pdf"))
    print("Success")
except Exception as e:
    import sys
    traceback.print_exc(file=sys.stdout)
