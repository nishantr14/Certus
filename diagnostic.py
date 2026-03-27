"""Diagnostic: run all test images through the pipeline and produce detailed table."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from certusdoc.pipeline import CertusDocPipeline
from loguru import logger

# Reduce log noise for clean table output
logger.remove()
logger.add(sys.stderr, level="WARNING")

pipeline = CertusDocPipeline()

data_dir = Path(__file__).parent / "data"
results = []

for label_dir, label in [("forged", "forged"), ("authentic", "authentic")]:
    d = data_dir / label_dir
    if not d.is_dir():
        continue
    for f in sorted(d.iterdir()):
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf", ".tiff", ".bmp"):
            try:
                report = pipeline.analyze(str(f))
                agent_scores = {}
                agent_findings = {}
                for ar in report.agent_results:
                    agent_scores[ar.agent_name] = ar.score
                    agent_findings[ar.agent_name] = [
                        {"desc": finding.description, "severity": finding.severity}
                        for finding in ar.findings
                    ]
                results.append({
                    "filename": f.name,
                    "label": label,
                    "dis_score": round(report.dis_score, 4),
                    "risk_level": report.risk_level.value,
                    "visual_score": round(agent_scores.get("Visual Tamper Agent", -1), 4),
                    "text_score": round(agent_scores.get("Text Forensics Agent", -1), 4),
                    "metadata_score": round(agent_scores.get("Metadata & Provenance Agent", -1), 4),
                    "forgery_type": report.primary_forgery_type.value,
                    "findings": agent_findings,
                })
            except Exception as e:
                results.append({
                    "filename": f.name,
                    "label": label,
                    "dis_score": -1,
                    "risk_level": "ERROR",
                    "visual_score": -1,
                    "text_score": -1,
                    "metadata_score": -1,
                    "forgery_type": "error",
                    "findings": {},
                    "error": str(e),
                })

# Print table
print("\n" + "=" * 120)
print(f"{'FILENAME':<35} {'LABEL':<10} {'DIS':>6} {'RISK LEVEL':<15} {'VISUAL':>8} {'TEXT':>8} {'META':>8} {'FORGERY TYPE':<20}")
print("-" * 120)
for r in results:
    flag = " *** MISS" if r["label"] == "forged" and r["dis_score"] >= 0.50 else ""
    flag2 = " *** FP" if r["label"] == "authentic" and r["dis_score"] < 0.60 else ""
    print(f"{r['filename']:<35} {r['label']:<10} {r['dis_score']:>6.4f} {r['risk_level']:<15} "
          f"{r['visual_score']:>8.4f} {r['text_score']:>8.4f} {r['metadata_score']:>8.4f} "
          f"{r['forgery_type']:<20}{flag}{flag2}")
print("=" * 120)

# Save full results with findings to JSON for analysis
with open(str(Path(__file__).parent / "diagnostic_results.json"), "w") as fp:
    json.dump(results, fp, indent=2)

# Print missed forgeries detail
print("\n\n=== MISSED FORGERIES (forged docs with DIS >= 0.50) ===\n")
missed = [r for r in results if r["label"] == "forged" and r["dis_score"] >= 0.50]
if not missed:
    print("None! All forgeries detected correctly.")
else:
    for r in missed:
        print(f"\n--- {r['filename']} (DIS={r['dis_score']}, Risk={r['risk_level']}) ---")
        for agent_name, findings in r["findings"].items():
            print(f"  [{agent_name}] score={r.get('visual_score' if 'Visual' in agent_name else 'text_score' if 'Text' in agent_name else 'metadata_score', '?')}")
            if findings:
                for f in findings:
                    print(f"    - [{f['severity']:.2f}] {f['desc']}")
            else:
                print(f"    (no findings - agent saw nothing suspicious)")

print("\n\n=== FALSE POSITIVES (authentic docs with DIS < 0.60) ===\n")
fps = [r for r in results if r["label"] == "authentic" and r["dis_score"] < 0.60]
if not fps:
    print("None! All authentic docs scored correctly.")
else:
    for r in fps:
        print(f"\n--- {r['filename']} (DIS={r['dis_score']}, Risk={r['risk_level']}) ---")
        for agent_name, findings in r["findings"].items():
            if findings:
                for f in findings:
                    print(f"    - [{f['severity']:.2f}] {f['desc']}")
