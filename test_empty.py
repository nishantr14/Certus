import traceback
import tempfile
import pdfplumber

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    pass  # Create empty file

try:
    with pdfplumber.open(tmp.name) as pdf:
        pages = pdf.pages
    print("Success")
except Exception as e:
    import sys
    print(f"Exception repr: {repr(e)}")
    print(f"Exception str: {str(e)}")
    traceback.print_exc(file=sys.stdout)
