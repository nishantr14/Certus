import traceback
import tempfile
import os
import shutil
from certusdoc.pipeline import CertusDocPipeline

pipeline = CertusDocPipeline()

def test_api():
    src = r"c:\Users\LIVANA\Certus\test_report.pdf"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        with open(src, "rb") as f:
            content = f.read()
            tmp.write(content)
        tmp_path = tmp.name

    try:
        report = pipeline.analyze(tmp_path)
        print("Success")
    except Exception as e:
        import sys
        print(f"Exception repr: {repr(e)}")
        print(f"Exception str: {str(e)}")
        print(f"Exception class: {e.__class__}")
        traceback.print_exc(file=sys.stdout)
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    test_api()
