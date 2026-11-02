#!/usr/bin/env python3
"""
OmniDocBench Evaluation Runner for RustO OCR Pipeline.
Reference: https://github.com/byrizki/rusto-rs/issues/17

Evaluates RustO OCR on the OmniDocBench benchmark dataset pinned to:
- Dataset:    Hugging Face opendatalab/OmniDocBench @ aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec
- Evaluator:  GitHub opendatalab/OmniDocBench       @ f133a71e9e91c3621c7ce8994200a7b394a06eb3
"""

import argparse
import json
import os
import random
import subprocess
import sys
for p in ["/usr/lib/python3/dist-packages", "/usr/local/lib/python3/dist-packages"]:
    if p not in sys.path and os.path.exists(p):
        sys.path.append(p)
import time
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RustO on OmniDocBench")
    parser.add_argument(
        "--tier",
        choices=["tiny", "small", "medium"],
        default="tiny",
        help="PP-OCRv6 model tier to evaluate (default: tiny)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to directory containing det.onnx, rec.onnx, dict.txt (defaults to models/PPOCR_v6_{tier})",
    )
    parser.add_argument(
        "--binary",
        type=str,
        default=None,
        help="Path to rusto-rs binary (defaults to target/release/rusto-rs)",
    )
    parser.add_argument(
        "--subset",
        choices=["english", "all", "sample"],
        default="english",
        help="Dataset subset: english (755 pages, issue 17), sample (101 seeded pages), or all",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Run evaluation on a single image filename (e.g., newspaper_USAToday-2025-01-08@magazinesclubnew_page_012.png)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to first N images",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Directory to store markdown predictions (default: benchmarks/omnidocbench/predictions/{tier})",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip OCR inference and evaluate existing predictions only",
    )
    return parser.parse_args()


def load_ground_truth(gt_path: Path):
    if not gt_path.exists():
        sys.exit(f"Ground truth not found at {gt_path}. Run scripts/download_omnidocbench.sh first.")
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_samples(gt_data, subset: str, single_image: str | None = None, limit: int | None = None):
    if single_image:
        samples = [
            s for s in gt_data
            if s.get("page_info", {}).get("image_path") == single_image
            or os.path.basename(s.get("page_info", {}).get("image_path", "")) == single_image
        ]
        if not samples:
            sys.exit(f"Image {single_image} not found in OmniDocBench.json")
        return samples

    if subset == "english":
        samples = [
            s for s in gt_data
            if s.get("page_info", {}).get("page_attribute", {}).get("language") == "english"
        ]
    elif subset == "sample":
        en_samples = [
            s for s in gt_data
            if s.get("page_info", {}).get("page_attribute", {}).get("language") == "english"
        ]
        random.seed(42)
        samples = en_samples[:101]
    else:
        samples = gt_data

    if limit is not None and limit > 0:
        samples = samples[:limit]

    return samples


def run_inference(samples, binary_path: Path, det_model: Path, rec_model: Path, dict_file: Path,
                  images_dir: Path, predictions_dir: Path):
    predictions_dir.mkdir(parents=True, exist_ok=True)
    timings = []

    print(f"\n==> Running RustO inference on {len(samples)} images...")
    print(f"Binary:     {binary_path}")
    print(f"Det model:  {det_model}")
    print(f"Rec model:  {rec_model}")
    print(f"Dict:       {dict_file}")
    print(f"Output dir: {predictions_dir}\n")

    for idx, sample in enumerate(samples, 1):
        img_name = sample.get("page_info", {}).get("image_path")
        if not img_name:
            continue
        img_path = images_dir / img_name
        if not img_path.exists():
            print(f"[{idx}/{len(samples)}] Warning: Image file not found: {img_path}. Skipping.")
            continue

        pred_stem = Path(img_name).stem
        pred_path = predictions_dir / f"{pred_stem}.md"

        cmd = [
            str(binary_path),
            "--det-model", str(det_model),
            "--rec-model", str(rec_model),
            "--dict", str(dict_file),
            "--format", "text",
            str(img_path),
        ]

        t0 = time.perf_counter()
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings.append(elapsed_ms)

            with open(pred_path, "w", encoding="utf-8") as f:
                f.write(res.stdout)

            if idx % 10 == 0 or idx == len(samples) or len(samples) <= 10:
                print(f"[{idx}/{len(samples)}] {img_name}: {elapsed_ms:.1f}ms -> {pred_path.name}")
        except subprocess.CalledProcessError as e:
            print(f"[{idx}/{len(samples)}] Error processing {img_name}: {e.stderr}")

    return timings


def compute_normalized_edit_distance(s1: str, s2: str) -> float:
    """Computes standard normalized edit distance: Levenshtein(s1, s2) / max(len(s1), len(s2))."""
    if s1 == s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 1.0

    # Try fast C Levenshtein if available
    try:
        import Levenshtein
        dist = Levenshtein.distance(s1, s2)
        return dist / max(len1, len2)
    except ImportError:
        pass

    # Dynamic programming fallback
    dp = list(range(len2 + 1))
    for i, c1 in enumerate(s1):
        new_dp = [i + 1] * (len2 + 1)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            new_dp[j + 1] = min(dp[j + 1] + 1, new_dp[j] + 1, dp[j] + cost)
        dp = new_dp
    return dp[len2] / max(len1, len2)


def evaluate_text_blocks(samples, predictions_dir: Path):
    """
    Evaluates text-block level accuracy matching Issue #17:
    Compares the full concatenated text of ground truth text blocks (in reading order)
    against the OCR output text.
    """
    edit_distances = []
    page_results = []

    for sample in samples:
        img_name = sample.get("page_info", {}).get("image_path")
        pred_stem = Path(img_name).stem
        pred_path = predictions_dir / f"{pred_stem}.md"

        if not pred_path.exists():
            continue

        with open(pred_path, "r", encoding="utf-8") as f:
            pred_text = f.read().strip()

        # Extract scored text blocks from GT in annotated reading order
        gt_blocks = [
            d for d in sample.get("layout_dets", [])
            if d.get("category_type") == "text_block" and not d.get("ignore", False)
        ]
        gt_blocks.sort(key=lambda x: x.get("order", 0))
        gt_text = "\n".join(b.get("text", "").strip() for b in gt_blocks if b.get("text", "").strip())

        if not gt_text:
            continue

        # Clean/normalize whitespace for comparison
        norm_gt = " ".join(gt_text.split())
        norm_pred = " ".join(pred_text.split())

        ned = compute_normalized_edit_distance(norm_gt, norm_pred)
        edit_distances.append(ned)
        page_results.append({
            "image": img_name,
            "normalized_edit_distance": ned,
            "gt_chars": len(norm_gt),
            "pred_chars": len(norm_pred),
        })

    return edit_distances, page_results


def print_summary(timings, edit_distances, page_results):
    print("\n" + "=" * 60)
    print("                EVALUATION SUMMARY")
    print("=" * 60)

    if timings:
        timings_sorted = sorted(timings)
        median_time = timings_sorted[len(timings_sorted) // 2]
        p95_idx = int(len(timings_sorted) * 0.95)
        p95_time = timings_sorted[min(p95_idx, len(timings_sorted) - 1)]
        mean_time = sum(timings) / len(timings)
        print("\n### Timing Performance (ms)")
        print(f"- Total pages timed: {len(timings)}")
        print(f"- Mean latency:     {mean_time:.2f} ms")
        print(f"- Median latency:   {median_time:.2f} ms")
        print(f"- P95 latency:      {p95_time:.2f} ms")

    if edit_distances:
        mean_ned = sum(edit_distances) / len(edit_distances)
        print("\n### Quality Metrics")
        print(f"- Pages evaluated:               {len(edit_distances)}")
        print(f"- Mean Normalized Edit Distance: {mean_ned:.4f} (lower is better)")

        if len(page_results) <= 10:
            print("\n### Individual Page Results:")
            for p in page_results:
                print(f"  {p['image']}: NED = {p['normalized_edit_distance']:.4f} (GT: {p['gt_chars']} chars, Pred: {p['pred_chars']} chars)")
        else:
            print("\n### Best 3 Pages:")
            for p in sorted(page_results, key=lambda x: x["normalized_edit_distance"])[:3]:
                print(f"  {p['image']}: NED = {p['normalized_edit_distance']:.4f}")
            print("\n### Worst 3 Pages (often complex multi-column reading order):")
            for p in sorted(page_results, key=lambda x: x["normalized_edit_distance"], reverse=True)[:3]:
                print(f"  {p['image']}: NED = {p['normalized_edit_distance']:.4f}")

    print("=" * 60 + "\n")


def main():
    args = parse_args()
    repo_root = get_repo_root()

    # Paths
    omnidoc_dir = repo_root / "benchmarks" / "omnidocbench"
    dataset_dir = omnidoc_dir / "dataset"
    gt_file = dataset_dir / "OmniDocBench.json"
    images_dir = dataset_dir / "images"

    binary_path = Path(args.binary) if args.binary else repo_root / "target" / "release" / "rusto-rs"
    if not binary_path.exists():
        sys.exit(f"Rusto binary not found at {binary_path}. Run 'cargo build --release' first.")

    model_dir = Path(args.model_dir) if args.model_dir else repo_root / "models" / f"PPOCR_v6_{args.tier}"
    det_model = model_dir / "det.onnx"
    rec_model = model_dir / "rec.onnx"
    dict_file = model_dir / "dict.txt"

    for required_file in [det_model, rec_model, dict_file]:
        if not required_file.exists():
            sys.exit(f"Model file not found: {required_file}. Run scripts/download_models.sh first.")

    pred_dir = Path(args.predictions_dir) if args.predictions_dir else omnidoc_dir / "predictions" / args.tier

    # 1. Load Ground Truth
    gt_data = load_ground_truth(gt_file)
    samples = filter_samples(gt_data, subset=args.subset, single_image=args.image, limit=args.limit)

    print(f"Found {len(samples)} target pages for evaluation.")

    # 2. Run Inference
    timings = []
    if not args.skip_inference:
        timings = run_inference(
            samples=samples,
            binary_path=binary_path,
            det_model=det_model,
            rec_model=rec_model,
            dict_file=dict_file,
            images_dir=images_dir,
            predictions_dir=pred_dir,
        )

    # 3. Quality Evaluation
    edit_distances, page_results = evaluate_text_blocks(samples, pred_dir)

    # 4. Print Summary
    print_summary(timings, edit_distances, page_results)


if __name__ == "__main__":
    main()
