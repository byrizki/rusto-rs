#!/usr/bin/env python3
"""Benchmark and Impact Verification of RustO Auto-Config vs Baseline.

Compares standard baseline OCR (auto_config=False, layout_reading=False)
against Dimension-Adaptive Auto-Config (auto_config=True, layout_reading=False)
and Reading Layout (auto_config=True, layout_reading=True) across synthetic
degraded optical conditions and real-world documents.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages/python"))

import rusto
from rusto import ImageSource, OcrRunOptions, OutputGranularity, RustO

BENCHMARK_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = BENCHMARK_DIR / "fixtures"
RESULTS_DIR = BENCHMARK_DIR / "results"


def percentile_95(data):
    """Compute 95th percentile using standard library."""
    if not data:
        return 0.0
    if len(data) == 1:
        return float(data[0])
    sorted_data = sorted(data)
    idx = int(math.ceil(0.95 * len(sorted_data))) - 1
    return float(sorted_data[max(0, min(idx, len(sorted_data) - 1))])


def ensure_synthetic_fixtures():
    """Ensure synthetic optical condition test fixtures exist."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = {}

    stone_path = FIXTURES_DIR / "carved_gravestone.png"
    deg_path = FIXTURES_DIR / "degraded_broken_text.png"
    grad_path = FIXTURES_DIR / "uneven_lighting.png"
    blur_path = FIXTURES_DIR / "optical_blur.png"

    # Only import numpy / PIL if we actually need to generate missing fixtures
    missing = not (stone_path.exists() and deg_path.exists() and grad_path.exists() and blur_path.exists())
    if missing:
        try:
            import numpy as np
            from PIL import Image, ImageDraw, ImageFilter
        except ImportError as e:
            raise RuntimeError(
                f"Missing fixtures and cannot generate them without numpy and Pillow: {e}"
            ) from e

        # 1. Carved text on white gravestone
        if not stone_path.exists():
            w, h = 600, 240
            stone_bg = np.random.normal(210, 4, (h, w, 3)).clip(195, 225).astype(np.uint8)
            img_stone = Image.fromarray(stone_bg)
            draw = ImageDraw.Draw(img_stone)
            for text, y in [("IN MEMORY OF", 35), ("ALEXANDER HAMILTON", 90), ("1757 - 1804", 160)]:
                x = 50
                draw.text((x + 2, y + 2), text, fill=(125, 125, 125))
                draw.text((x + 1, y + 1), text, fill=(145, 145, 145))
                draw.text((x - 1, y - 1), text, fill=(245, 245, 245))
                draw.text((x, y), text, fill=(210, 210, 210))
            img_stone.save(stone_path)

        # 2. Degraded / Broken Text Shapes
        if not deg_path.exists():
            img_deg = Image.new("RGB", (600, 200), color=(255, 255, 255))
            draw = ImageDraw.Draw(img_deg)
            draw.text((40, 40), "SYSTEM RECOVERY CODE", fill=(20, 20, 20))
            draw.text((40, 110), "AUTH-9821-BATCH-42", fill=(20, 20, 20))
            arr_deg = np.array(img_deg)
            for row in range(45, 160, 4):
                if np.random.rand() > 0.3:
                    arr_deg[row : row + 1, 40:550] = 255
            noise_mask = (np.random.rand(arr_deg.shape[0], arr_deg.shape[1]) < 0.05) & (arr_deg[:, :, 0] < 100)
            arr_deg[noise_mask] = 255
            Image.fromarray(arr_deg).save(deg_path)

        # 3. Severe Illumination Gradient
        if not grad_path.exists():
            img_grad = Image.new("RGB", (650, 220), color=(255, 255, 255))
            draw = ImageDraw.Draw(img_grad)
            draw.text((40, 40), "FINANCIAL ACCOUNT SUMMARY", fill=(10, 10, 10))
            draw.text((40, 95), "BALANCE: $458,920.00 USD", fill=(10, 10, 10))
            draw.text((40, 150), "STATUS: VERIFIED TRANSACTION", fill=(10, 10, 10))
            arr_grad = np.array(img_grad, dtype=np.float32)
            ramp = np.linspace(1.0, 0.22, 650)[None, :, None]
            arr_grad = (arr_grad * ramp).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr_grad).save(grad_path)

        # 4. Optical Blur
        if not blur_path.exists():
            img_blur = Image.new("RGB", (600, 180), color=(250, 250, 250))
            draw = ImageDraw.Draw(img_blur)
            draw.text((40, 40), "OUT OF FOCUS LENS SAMPLE", fill=(30, 30, 30))
            draw.text((40, 100), "CALIBRATION TEST PASS 100%", fill=(30, 30, 30))
            img_blur = img_blur.filter(ImageFilter.GaussianBlur(radius=1.6))
            img_blur.save(blur_path)

    fixtures["Carved White Gravestone"] = {
        "path": stone_path,
        "ground_truth": ["IN MEMORY OF", "ALEXANDER HAMILTON", "1757 - 1804"],
        "description": "Engraved text on high-luma limestone (low-contrast sunken shadow grooves)",
    }
    fixtures["Degraded / Broken Text"] = {
        "path": deg_path,
        "ground_truth": ["SYSTEM RECOVERY CODE", "AUTH-9821-BATCH-42"],
        "description": "Fractured text crossbars and eroded strokes from ribbon degradation",
    }
    fixtures["Uneven Illumination Gradient"] = {
        "path": grad_path,
        "ground_truth": ["FINANCIAL ACCOUNT SUMMARY", "BALANCE: $458,920.00 USD", "STATUS: VERIFIED TRANSACTION"],
        "description": "Lighting gradient dropping from 100% luma to 22% deep shadow across text",
    }
    fixtures["Optical Defocus Blur"] = {
        "path": blur_path,
        "ground_truth": ["OUT OF FOCUS LENS SAMPLE", "CALIBRATION TEST PASS 100%"],
        "description": "Camera defocus blur (Gaussian radius 1.6)",
    }

    return fixtures


def get_real_world_fixtures():
    """Retrieve existing real fixtures from the repository."""
    fixtures = {}

    idcard_path = REPO_ROOT / "models/test_images/idcard.jpg"
    if idcard_path.exists():
        fixtures["ID Card / Passport (Real)"] = {
            "path": idcard_path,
            "ground_truth": ["UNITED", "BRITAIN", "PASSPORT"],
            "description": "Guilloche security background pattern + uneven lighting (1024x721)",
        }

    chronam_path = REPO_ROOT / "tests/fixtures/chronam-ad-page.png"
    if chronam_path.exists():
        fixtures["Historical Newspaper Page (Real)"] = {
            "path": chronam_path,
            "ground_truth": ["All Orders in our", "Dry Goods"],
            "description": "Dense high-resolution historical newspaper scan with small font (999x743)",
        }

    invoice_path = REPO_ROOT / "models/test_images/invoice1.jpg"
    if invoice_path.exists():
        fixtures["Scanned Invoice (Real)"] = {
            "path": invoice_path,
            "ground_truth": ["INVOICE"],
            "description": "Scanned document with numeric tables and low contrast (359x600)",
        }

    example_path = REPO_ROOT / "models/test_images/example1.png"
    if example_path.exists():
        fixtures["Standard Document Example1 (Real)"] = {
            "path": example_path,
            "ground_truth": ["General"],
            "description": "Standard document scan fixture (462x656)",
        }

    return fixtures


def benchmark_fixture(
    engine,
    img_path,
    auto_config: bool,
    layout_reading: bool = False,
    iterations: int = 5,
    warmups: int = 2,
):
    """Run warmups and timed benchmark iterations."""
    src = ImageSource.PATH(str(img_path))
    opts = OcrRunOptions(
        output=OutputGranularity.LINES,
        auto_config=auto_config,
        layout_reading=layout_reading,
    )

    for _ in range(warmups):
        engine.detect_text(src, opts)

    durations_ms = []
    last_res = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        last_res = engine.detect_text(src, opts)
        t1 = time.perf_counter()
        durations_ms.append((t1 - t0) * 1000.0)

    results = last_res.results if last_res.is_STRUCTURED() else []
    texts = [r.text for r in results]
    scores = [r.score for r in results]
    avg_score = float(statistics.mean(scores)) if scores else 0.0
    total_chars = sum(len(t) for t in texts)

    return {
        "durations_ms": durations_ms,
        "mean_ms": float(statistics.mean(durations_ms)),
        "median_ms": float(statistics.median(durations_ms)),
        "min_ms": float(min(durations_ms)),
        "max_ms": float(max(durations_ms)),
        "p95_ms": percentile_95(durations_ms),
        "line_count": len(results),
        "total_chars": total_chars,
        "avg_confidence": avg_score,
        "recognized_texts": texts,
    }


def compute_match_score(recognized_texts, ground_truth):
    """Check how many ground truth phrases are found in recognized text."""
    joined = " ".join(recognized_texts).upper()
    matched = 0
    for gt in ground_truth:
        if gt.upper() in joined:
            matched += 1
    return matched, len(ground_truth)


def main():
    parser = argparse.ArgumentParser(description="RustO Auto-Config vs Baseline Benchmark")
    parser.add_argument("--iterations", type=int, default=5, help="Number of timed benchmark runs (default: 5)")
    parser.add_argument("--warmups", type=int, default=2, help="Number of warmup iterations (default: 2)")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "auto_config_benchmark_results.json",
        help="Path to save benchmark results JSON",
    )
    args = parser.parse_args()

    print("=" * 90)
    print("RustO Auto-Config vs Baseline vs Reading Layout Benchmark & Impact Analysis")
    print("=" * 90)

    det_path = REPO_ROOT / "models/PPOCR_v6_tiny/det.rten"
    rec_path = REPO_ROOT / "models/PPOCR_v6_tiny/rec.rten"
    dict_path = REPO_ROOT / "models/PPOCR_v6_tiny/dict.txt"

    if not (det_path.exists() and rec_path.exists() and dict_path.exists()):
        print(f"Error: Model files missing in {det_path.parent}", file=sys.stderr)
        sys.exit(1)

    config = {
        "detection": {"modelPath": str(det_path), "enabled": True},
        "recognition": {"modelPath": str(rec_path), "dictPath": str(dict_path), "enabled": True},
    }
    print(f"Loading RustO engine with {det_path.name} & {rec_path.name}...")
    engine = RustO.initialize_from_json(json.dumps(config))
    print("Engine initialized successfully.\n")

    synthetic_fixtures = ensure_synthetic_fixtures()
    real_fixtures = get_real_world_fixtures()
    all_fixtures = {**synthetic_fixtures, **real_fixtures}

    summary_records = []

    for name, meta in all_fixtures.items():
        img_path = meta["path"]
        gt = meta.get("ground_truth", [])
        desc = meta.get("description", "")

        print(f"--> Testing: {name}")
        print(f"    Description: {desc}")
        print(f"    File: {img_path.name}")

        base = benchmark_fixture(
            engine,
            img_path,
            auto_config=False,
            layout_reading=False,
            iterations=args.iterations,
            warmups=args.warmups,
        )
        auto = benchmark_fixture(
            engine,
            img_path,
            auto_config=True,
            layout_reading=False,
            iterations=args.iterations,
            warmups=args.warmups,
        )
        layout = benchmark_fixture(
            engine,
            img_path,
            auto_config=True,
            layout_reading=True,
            iterations=args.iterations,
            warmups=args.warmups,
        )

        overhead_ms = auto["mean_ms"] - base["mean_ms"]
        overhead_pct = (overhead_ms / base["mean_ms"]) * 100.0 if base["mean_ms"] > 0 else 0.0

        base_match, total_gt = compute_match_score(base["recognized_texts"], gt)
        auto_match, _ = compute_match_score(auto["recognized_texts"], gt)
        layout_match, _ = compute_match_score(layout["recognized_texts"], gt)

        print(
            f"    Baseline      : {base['mean_ms']:7.2f}ms | Lines: {base['line_count']:3d} | "
            f"Chars: {base['total_chars']:4d} | Conf: {base['avg_confidence'] * 100:5.1f}% | Matches: {base_match}/{total_gt}"
        )
        print(
            f"    Auto-Config   : {auto['mean_ms']:7.2f}ms | Lines: {auto['line_count']:3d} | "
            f"Chars: {auto['total_chars']:4d} | Conf: {auto['avg_confidence'] * 100:5.1f}% | Matches: {auto_match}/{total_gt} | Speedup: {overhead_pct:+.1f}%"
        )
        print(
            f"    Auto + Layout : {layout['mean_ms']:7.2f}ms | Lines: {layout['line_count']:3d} | "
            f"Chars: {layout['total_chars']:4d} | Conf: {layout['avg_confidence'] * 100:5.1f}% | Matches: {layout_match}/{total_gt}\n"
        )

        summary_records.append({
            "fixture": name,
            "description": desc,
            "baseline_mean_ms": base["mean_ms"],
            "auto_mean_ms": auto["mean_ms"],
            "layout_mean_ms": layout["mean_ms"],
            "overhead_ms": overhead_ms,
            "overhead_pct": overhead_pct,
            "baseline_lines": base["line_count"],
            "auto_lines": auto["line_count"],
            "layout_lines": layout["line_count"],
            "baseline_chars": base["total_chars"],
            "auto_chars": auto["total_chars"],
            "layout_chars": layout["total_chars"],
            "baseline_conf": base["avg_confidence"],
            "auto_conf": auto["avg_confidence"],
            "layout_conf": layout["avg_confidence"],
            "baseline_matches": f"{base_match}/{total_gt}",
            "auto_matches": f"{auto_match}/{total_gt}",
            "layout_matches": f"{layout_match}/{total_gt}",
            "sample_texts_base": base["recognized_texts"][:3],
            "sample_texts_auto": auto["recognized_texts"][:3],
            "sample_texts_layout": layout["recognized_texts"][:3],
        })

    engine.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary_records, f, indent=2)
    print(f"Detailed benchmark results saved to: {args.output}")


if __name__ == "__main__":
    main()
