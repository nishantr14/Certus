"""
CertusDoc Evaluation Script

Evaluates the pipeline against known authentic/forged document datasets.
Supports: local data dirs, CASIA-style datasets, custom CSV manifests.

Usage:
    python eval.py                          # Run on data/authentic + data/forged
    python eval.py --data-dir path/to/data  # Custom data directory
    python eval.py --csv manifest.csv       # CSV with columns: path, label (authentic/forged)
    python eval.py --threshold 0.65         # Custom decision threshold
"""
import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from certusdoc.pipeline import CertusDocPipeline
from certusdoc.models import RiskLevel


@dataclass
class EvalResult:
    file_path: str
    label: str  # "authentic" or "forged"
    dis_score: float
    risk_level: str
    forgery_type: str
    agent_scores: dict
    processing_ms: float
    prediction: str  # "authentic" or "forged" (based on threshold)
    correct: bool


@dataclass
class EvalSummary:
    total: int = 0
    authentic_count: int = 0
    forged_count: int = 0
    true_positives: int = 0   # forged correctly detected
    true_negatives: int = 0   # authentic correctly passed
    false_positives: int = 0  # authentic incorrectly flagged
    false_negatives: int = 0  # forged incorrectly passed
    results: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return (self.true_positives + self.true_negatives) / self.total if self.total > 0 else 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    @property
    def avg_authentic_score(self) -> float:
        scores = [r.dis_score for r in self.results if r.label == "authentic"]
        return float(np.mean(scores)) if scores else 0

    @property
    def avg_forged_score(self) -> float:
        scores = [r.dis_score for r in self.results if r.label == "forged"]
        return float(np.mean(scores)) if scores else 0


def load_coco_dataset(data_dir: str) -> list[tuple[str, str]]:
    """
    Load files from a COCO-format dataset (Roboflow export).
    Looks for _annotations.coco.json in the directory.
    Images with annotations are forged; those without are authentic.
    If all images have annotations, treat them ALL as forged.
    """
    extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    files = []

    # Search for train/, valid/, test/ subdirectories
    subdirs = [data_dir]
    for sub in ["train", "valid", "test"]:
        p = Path(data_dir) / sub
        if p.is_dir():
            subdirs.append(str(p))

    for subdir in subdirs:
        coco_path = Path(subdir) / "_annotations.coco.json"
        if not coco_path.exists():
            continue

        with open(coco_path, "r") as f:
            coco = json.load(f)

        # Build set of image IDs that have annotations
        annotated_ids = set()
        for ann in coco.get("annotations", []):
            annotated_ids.add(ann["image_id"])

        # All images with annotations are forged
        for img_info in coco.get("images", []):
            img_path = Path(subdir) / img_info["file_name"]
            if img_path.exists() and img_path.suffix.lower() in extensions:
                # In a forgery detection dataset, annotated = forged
                label = "forged" if img_info["id"] in annotated_ids else "authentic"
                files.append((str(img_path), label))

    return files


def discover_files(data_dir: str) -> list[tuple[str, str]]:
    """
    Discover test files from a directory structure.
    Expected: data_dir/authentic/*.{png,jpg,pdf,...}
              data_dir/forged/*.{png,jpg,pdf,...}
    Also supports: data_dir/Au/*, data_dir/Tp/* (CASIA format)
    """
    extensions = {".png", ".jpg", ".jpeg", ".pdf", ".tiff", ".bmp", ".webp"}
    files = []

    # Standard layout: authentic/ + forged/
    label_dirs = {
        "authentic": ["authentic", "real", "genuine", "original", "Au"],
        "forged": ["forged", "fake", "tampered", "manipulated", "Tp", "spliced"],
    }

    for label, dir_names in label_dirs.items():
        for dname in dir_names:
            d = Path(data_dir) / dname
            if d.is_dir():
                for f in sorted(d.iterdir()):
                    if f.suffix.lower() in extensions:
                        files.append((str(f), label))

    if not files:
        # Flat directory — try to infer from filename
        d = Path(data_dir)
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in extensions:
                name_lower = f.stem.lower()
                if any(kw in name_lower for kw in ["auth", "real", "genuine", "orig"]):
                    files.append((str(f), "authentic"))
                elif any(kw in name_lower for kw in ["forg", "fake", "tamp", "splice"]):
                    files.append((str(f), "forged"))

    return files


def load_csv_manifest(csv_path: str) -> list[tuple[str, str]]:
    """Load files from a CSV manifest with columns: path, label."""
    files = []
    base_dir = Path(csv_path).parent

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("path") or row.get("file") or row.get("filename")
            label = row.get("label") or row.get("class") or row.get("ground_truth")

            if not path or not label:
                continue

            # Resolve relative paths
            full_path = Path(path) if Path(path).is_absolute() else base_dir / path
            label_norm = "authentic" if label.lower() in ("authentic", "real", "genuine", "0") else "forged"
            files.append((str(full_path), label_norm))

    return files


def run_evaluation(
    pipeline: CertusDocPipeline,
    files: list[tuple[str, str]],
    threshold: float = 0.65,
) -> EvalSummary:
    """Run the pipeline on all files and compute metrics."""
    summary = EvalSummary()

    for i, (file_path, label) in enumerate(files):
        if not Path(file_path).exists():
            summary.errors.append(f"File not found: {file_path}")
            continue

        summary.total += 1
        if label == "authentic":
            summary.authentic_count += 1
        else:
            summary.forged_count += 1

        try:
            report = pipeline.analyze(file_path)

            prediction = "authentic" if report.dis_score >= threshold else "forged"
            correct = prediction == label

            result = EvalResult(
                file_path=file_path,
                label=label,
                dis_score=report.dis_score,
                risk_level=report.risk_level.value,
                forgery_type=report.primary_forgery_type.value,
                agent_scores={r.agent_name: round(r.score, 4) for r in report.agent_results},
                processing_ms=report.processing_time_ms,
                prediction=prediction,
                correct=correct,
            )
            summary.results.append(result)

            if label == "forged" and prediction == "forged":
                summary.true_positives += 1
            elif label == "authentic" and prediction == "authentic":
                summary.true_negatives += 1
            elif label == "authentic" and prediction == "forged":
                summary.false_positives += 1
            elif label == "forged" and prediction == "authentic":
                summary.false_negatives += 1

            status = "OK" if correct else "MISS"
            print(f"  [{i+1}/{len(files)}] {status} | {Path(file_path).name:<40} "
                  f"| label={label:<10} | DIS={report.dis_score:.4f} "
                  f"| pred={prediction:<10} | {report.risk_level.value}")

        except Exception as e:
            summary.errors.append(f"Error on {file_path}: {str(e)}")
            print(f"  [{i+1}/{len(files)}] ERR | {Path(file_path).name:<40} | {str(e)[:60]}")

    return summary


def print_report(summary: EvalSummary, threshold: float):
    """Print evaluation report."""
    print("\n" + "=" * 70)
    print("CERTUSDOC EVALUATION REPORT")
    print("=" * 70)

    print(f"\nDataset: {summary.total} files "
          f"({summary.authentic_count} authentic, {summary.forged_count} forged)")
    print(f"Decision threshold: {threshold}")

    if summary.errors:
        print(f"Errors: {len(summary.errors)}")
        for e in summary.errors[:5]:
            print(f"  - {e}")

    print(f"\n--- Confusion Matrix ---")
    print(f"  True Positives  (forged -> forged):    {summary.true_positives}")
    print(f"  True Negatives  (authentic -> authentic): {summary.true_negatives}")
    print(f"  False Positives (authentic -> forged):  {summary.false_positives}")
    print(f"  False Negatives (forged -> authentic):  {summary.false_negatives}")

    print(f"\n--- Metrics ---")
    print(f"  Accuracy:  {summary.accuracy:.4f}")
    print(f"  Precision: {summary.precision:.4f}")
    print(f"  Recall:    {summary.recall:.4f}")
    print(f"  F1 Score:  {summary.f1:.4f}")

    print(f"\n--- Score Distribution ---")
    print(f"  Avg authentic DIS: {summary.avg_authentic_score:.4f}")
    print(f"  Avg forged DIS:    {summary.avg_forged_score:.4f}")
    print(f"  Score gap:         {summary.avg_authentic_score - summary.avg_forged_score:.4f}")

    if summary.results:
        auth_scores = sorted([r.dis_score for r in summary.results if r.label == "authentic"])
        forg_scores = sorted([r.dis_score for r in summary.results if r.label == "forged"])

        if auth_scores:
            print(f"  Authentic range:   [{min(auth_scores):.4f}, {max(auth_scores):.4f}]")
        if forg_scores:
            print(f"  Forged range:      [{min(forg_scores):.4f}, {max(forg_scores):.4f}]")

    # Worst misses
    false_negs = [r for r in summary.results if r.label == "forged" and not r.correct]
    false_pos = [r for r in summary.results if r.label == "authentic" and not r.correct]

    if false_negs:
        print(f"\n--- Worst False Negatives (forged but passed) ---")
        for r in sorted(false_negs, key=lambda x: -x.dis_score)[:5]:
            print(f"  {Path(r.file_path).name}: DIS={r.dis_score:.4f} "
                  f"agents={r.agent_scores}")

    if false_pos:
        print(f"\n--- Worst False Positives (authentic but flagged) ---")
        for r in sorted(false_pos, key=lambda x: x.dis_score)[:5]:
            print(f"  {Path(r.file_path).name}: DIS={r.dis_score:.4f} "
                  f"agents={r.agent_scores}")

    print("\n" + "=" * 70)


def export_results(summary: EvalSummary, output_path: str):
    """Export results to JSON."""
    data = {
        "metrics": {
            "accuracy": summary.accuracy,
            "precision": summary.precision,
            "recall": summary.recall,
            "f1": summary.f1,
            "avg_authentic_dis": summary.avg_authentic_score,
            "avg_forged_dis": summary.avg_forged_score,
        },
        "confusion_matrix": {
            "tp": summary.true_positives,
            "tn": summary.true_negatives,
            "fp": summary.false_positives,
            "fn": summary.false_negatives,
        },
        "results": [
            {
                "file": r.file_path,
                "label": r.label,
                "dis_score": r.dis_score,
                "risk_level": r.risk_level,
                "forgery_type": r.forgery_type,
                "agent_scores": r.agent_scores,
                "prediction": r.prediction,
                "correct": r.correct,
                "processing_ms": r.processing_ms,
            }
            for r in summary.results
        ],
        "errors": summary.errors,
    }

    # Convert numpy types to native Python for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(_convert(data), f, indent=2)
    print(f"\nResults exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CertusDoc Dataset Evaluation")
    parser.add_argument("--data-dir", default="data",
                        help="Directory with authentic/ and forged/ subdirs")
    parser.add_argument("--csv", default=None,
                        help="CSV manifest (columns: path, label)")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="DIS threshold for authentic/forged classification (default: 0.65)")
    parser.add_argument("--output", default="eval_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max files to evaluate (for quick tests)")
    args = parser.parse_args()

    # Discover files — try COCO format first, then standard layout, then CSV
    if args.csv:
        files = load_csv_manifest(args.csv)
    else:
        # Try COCO format (Roboflow)
        files = load_coco_dataset(args.data_dir)
        if not files:
            files = discover_files(args.data_dir)

    if not files:
        print(f"No test files found in '{args.data_dir}'. "
              f"Expected authentic/ and forged/ subdirs, or _annotations.coco.json.")
        sys.exit(1)

    if args.limit:
        files = files[:args.limit]

    print(f"Found {len(files)} files ({sum(1 for _, l in files if l == 'authentic')} authentic, "
          f"{sum(1 for _, l in files if l == 'forged')} forged)")
    print(f"Threshold: {args.threshold}\n")

    # Initialize pipeline
    pipeline = CertusDocPipeline()

    # Run evaluation
    start = time.time()
    summary = run_evaluation(pipeline, files, threshold=args.threshold)
    elapsed = time.time() - start

    # Print report
    print_report(summary, args.threshold)
    print(f"\nTotal evaluation time: {elapsed:.1f}s "
          f"({elapsed/len(files):.1f}s/file avg)" if files else "")

    # Multi-threshold analysis
    if summary.results:
        print("\n--- Multi-Threshold Analysis ---")
        print(f"  {'Threshold':>10} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6} | {'FP':>3} | {'FN':>3}")
        print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*9}-+-{'-'*6}-+-{'-'*6}-+-{'-'*3}-+-{'-'*3}")

        best_f1 = 0
        best_threshold = args.threshold
        for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            tp = sum(1 for r in summary.results if r.label == "forged" and r.dis_score < t)
            tn = sum(1 for r in summary.results if r.label == "authentic" and r.dis_score >= t)
            fp = sum(1 for r in summary.results if r.label == "authentic" and r.dis_score < t)
            fn = sum(1 for r in summary.results if r.label == "forged" and r.dis_score >= t)
            total = tp + tn + fp + fn
            acc = (tp + tn) / total if total > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            marker = " <-- optimal" if f1 > best_f1 else ""
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
            print(f"  {t:>10.2f} | {acc:>8.4f} | {prec:>9.4f} | {rec:>6.4f} | {f1:>6.4f} | {fp:>3} | {fn:>3}{marker}")

        print(f"\n  Optimal threshold: {best_threshold:.2f} (F1={best_f1:.4f})")

    # Export
    export_results(summary, args.output)

    # Export markdown summary
    md_path = args.output.replace(".json", "_summary.md")
    with open(md_path, "w") as f:
        f.write(f"# CertusDoc Evaluation Summary\n\n")
        f.write(f"**Dataset:** {summary.total} files "
                f"({summary.authentic_count} authentic, {summary.forged_count} forged)\n")
        f.write(f"**Threshold:** {args.threshold}\n\n")
        f.write(f"## Metrics\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Accuracy | {summary.accuracy:.4f} |\n")
        f.write(f"| Precision | {summary.precision:.4f} |\n")
        f.write(f"| Recall | {summary.recall:.4f} |\n")
        f.write(f"| F1 Score | {summary.f1:.4f} |\n")
        f.write(f"| Avg Authentic DIS | {summary.avg_authentic_score:.4f} |\n")
        f.write(f"| Avg Forged DIS | {summary.avg_forged_score:.4f} |\n\n")
        f.write(f"## Confusion Matrix\n")
        f.write(f"| | Predicted Forged | Predicted Authentic |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| Actually Forged | {summary.true_positives} (TP) | {summary.false_negatives} (FN) |\n")
        f.write(f"| Actually Authentic | {summary.false_positives} (FP) | {summary.true_negatives} (TN) |\n")
    print(f"Summary exported to {md_path}")


if __name__ == "__main__":
    main()
