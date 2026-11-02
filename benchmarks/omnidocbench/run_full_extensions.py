#!/usr/bin/env python3
"""
Extend benchmark metrics with full-pipeline variants for RapidOCR and PaddleOCR
(with text angle classifier cls.onnx) using the exact same 30-document sample.
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add dist-packages if needed
for p in ["/usr/lib/python3/dist-packages", "/usr/local/lib/python3/dist-packages"]:
    if p not in sys.path and os.path.exists(p):
        sys.path.append(p)

import Levenshtein
from rapidocr_onnxruntime import RapidOCR
from paddleocr import PaddleOCR
from run_full_benchmark import (
    load_ground_truth,
    select_stratified_sample,
    get_ground_truth_text,
    compute_ned,
    measure_peak_rss_mb,
    compute_stats,
    REPO_ROOT,
    MODELS_DIR,
    IMAGES_DIR,
    PRED_DIR,
    RESULTS_DIR,
    VENV_BIN,
)

def run_rapidocr_full(samples, gt_texts, tier, warmup_img):
    key = f"rapidocr_full_{tier}"
    pred_dir = PRED_DIR / key
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_dir = MODELS_DIR / f"PPOCR_v6_{tier}"
    opts_dir = MODELS_DIR / "PPOCR_v6_tiny"

    engine = RapidOCR(
        det_model_path=str(model_dir / "det.onnx"),
        rec_model_path=str(model_dir / "rec.onnx"),
        rec_keys_path=str(model_dir / "dict.txt"),
        cls_model_path=str(opts_dir / "cls.onnx"),
    )

    # Warmup
    engine(str(IMAGES_DIR / warmup_img))

    # Peak RSS
    python_bin = str(VENV_BIN / "python")
    test_script = (
        f"from rapidocr_onnxruntime import RapidOCR; "
        f"e = RapidOCR(det_model_path='{model_dir}/det.onnx', rec_model_path='{model_dir}/rec.onnx', rec_keys_path='{model_dir}/dict.txt', cls_model_path='{opts_dir}/cls.onnx'); "
        f"e('{IMAGES_DIR / warmup_img}')"
    )
    peak_rss = measure_peak_rss_mb([python_bin, "-c", test_script])

    timings, neds, page_neds = [], [], {}
    print(f"\n===> Running [{key}] across {len(samples)} pages...")
    for idx, s in enumerate(samples, 1):
        img_name = s["page_info"]["image_path"]
        img_path = IMAGES_DIR / img_name
        stem = Path(img_name).stem
        pred_file = pred_dir / f"{stem}.txt"

        t0 = time.perf_counter()
        res, _ = engine(str(img_path))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed_ms)

        lines = [r[1] for r in res] if res else []
        pred_text = "\n".join(lines)
        with open(pred_file, "w", encoding="utf-8") as f:
            f.write(pred_text)

        pred_norm = " ".join(pred_text.split())
        gt_norm = gt_texts.get(img_name, "")
        ned = compute_ned(gt_norm, pred_norm)
        neds.append(ned)
        page_neds[img_name] = ned

        if idx % 10 == 0 or idx == len(samples) or "USAToday" in img_name:
            print(f"  [{idx}/{len(samples)}] {img_name[:35]:35s} | {elapsed_ms:6.1f}ms | NED: {ned:.4f}")

    stats = compute_stats(timings, neds)
    stats.update({
        "name": f"RapidOCR Full (with cls) + PP-OCRv6 {tier.capitalize()}",
        "target": "Target 3 (Full)",
        "binary_size_mb": 22.7,
        "model_size_mb": (model_dir / "det.onnx").stat().st_size / (1024*1024) + (model_dir / "rec.onnx").stat().st_size / (1024*1024) + (opts_dir / "cls.onnx").stat().st_size / (1024*1024),
        "engine_runtime": "Python (ONNX Runtime)",
        "peak_rss_mb": round(peak_rss, 1),
        "usatoday_ned": round(page_neds.get(warmup_img, 1.0), 4),
    })
    return key, stats

def run_paddleocr_full(samples, gt_texts, tier, warmup_img):
    key = f"paddleocr_full_{tier}"
    pred_dir = PRED_DIR / key
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_dir = MODELS_DIR / f"PPOCR_v6_{tier}"
    opts_dir = MODELS_DIR / "PPOCR_v6_tiny"

    ocr = PaddleOCR(
        use_onnx=True,
        det_model_dir=str(model_dir / "det.onnx"),
        rec_model_dir=str(model_dir / "rec.onnx"),
        rec_char_dict_path=str(model_dir / "dict.txt"),
        cls_model_dir=str(opts_dir / "cls.onnx"),
        use_angle_cls=True,
        show_log=False,
    )

    # Warmup
    ocr.ocr(str(IMAGES_DIR / warmup_img), cls=True)

    # Peak RSS
    python_bin = str(VENV_BIN / "python")
    test_script = (
        f"from paddleocr import PaddleOCR; "
        f"o = PaddleOCR(use_onnx=True, det_model_dir='{model_dir}/det.onnx', rec_model_dir='{model_dir}/rec.onnx', rec_char_dict_path='{model_dir}/dict.txt', cls_model_dir='{opts_dir}/cls.onnx', use_angle_cls=True, show_log=False); "
        f"o.ocr('{IMAGES_DIR / warmup_img}', cls=True)"
    )
    peak_rss = measure_peak_rss_mb([python_bin, "-c", test_script])

    timings, neds, page_neds = [], [], {}
    print(f"\n===> Running [{key}] across {len(samples)} pages...")
    for idx, s in enumerate(samples, 1):
        img_name = s["page_info"]["image_path"]
        img_path = IMAGES_DIR / img_name
        stem = Path(img_name).stem
        pred_file = pred_dir / f"{stem}.txt"

        t0 = time.perf_counter()
        res = ocr.ocr(str(img_path), cls=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed_ms)

        lines = [r[1][0] for r in res[0]] if res and res[0] else []
        pred_text = "\n".join(lines)
        with open(pred_file, "w", encoding="utf-8") as f:
            f.write(pred_text)

        pred_norm = " ".join(pred_text.split())
        gt_norm = gt_texts.get(img_name, "")
        ned = compute_ned(gt_norm, pred_norm)
        neds.append(ned)
        page_neds[img_name] = ned

        if idx % 10 == 0 or idx == len(samples) or "USAToday" in img_name:
            print(f"  [{idx}/{len(samples)}] {img_name[:35]:35s} | {elapsed_ms:6.1f}ms | NED: {ned:.4f}")

    stats = compute_stats(timings, neds)
    stats.update({
        "name": f"PaddleOCR Full (with cls) + PP-OCRv6 {tier.capitalize()}",
        "target": "Target 4 (Full)",
        "binary_size_mb": 120.1,
        "model_size_mb": (model_dir / "det.onnx").stat().st_size / (1024*1024) + (model_dir / "rec.onnx").stat().st_size / (1024*1024) + (opts_dir / "cls.onnx").stat().st_size / (1024*1024),
        "engine_runtime": "Python (PaddleOCR ONNX)",
        "peak_rss_mb": round(peak_rss, 1),
        "usatoday_ned": round(page_neds.get(warmup_img, 1.0), 4),
    })
    return key, stats

def main():
    gt_data = load_ground_truth()
    samples = select_stratified_sample(gt_data, sample_size=30)
    gt_texts = {s["page_info"]["image_path"]: get_ground_truth_text(s) for s in samples}
    warmup_img = "newspaper_USAToday-2025-01-08@magazinesclubnew_page_012.png"

    with open(RESULTS_DIR / "benchmark_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    for tier in ["tiny", "small"]:
        k, st = run_rapidocr_full(samples, gt_texts, tier, warmup_img)
        metrics[k] = st

    for tier in ["tiny", "small"]:
        k, st = run_paddleocr_full(samples, gt_texts, tier, warmup_img)
        metrics[k] = st

    with open(RESULTS_DIR / "benchmark_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("\n✓ Successfully updated benchmark_metrics.json with RapidOCR Full and PaddleOCR Full!")

if __name__ == "__main__":
    main()

