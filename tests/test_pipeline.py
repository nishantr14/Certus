"""
Quick smoke test for the CertusDoc pipeline.
Usage: python -m tests.test_pipeline <path_to_document>
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from certusdoc.pipeline import CertusDocPipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m tests.test_pipeline <path_to_document>")
        print("Example: python -m tests.test_pipeline data/sample.pdf")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"\n{'='*60}")
    print(f"CertusDoc Pipeline Test")
    print(f"Document: {file_path}")
    print(f"{'='*60}\n")

    pipeline = CertusDocPipeline()
    report = pipeline.analyze(file_path)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Document Integrity Score: {report.dis_score:.4f}")
    print(f"Risk Level:              {report.risk_level.value}")
    print(f"Forgery Type:            {report.primary_forgery_type.value}")
    print(f"Processing Time:         {report.processing_time_ms:.0f}ms")
    print(f"Recommended Action:      {report.recommended_action}")
    print()

    print("Per-Agent Results:")
    for result in report.agent_results:
        print(f"  {result.agent_name}:")
        print(f"    Score:       {result.score:.3f}")
        print(f"    Reliability: {result.reliability_weight:.2f}")
        print(f"    Findings:    {len(result.findings)}")
        for finding in result.findings[:3]:
            print(f"      - {finding.description[:100]}...")
        print()


if __name__ == "__main__":
    main()
