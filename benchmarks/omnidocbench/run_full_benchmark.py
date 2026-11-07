#!/usr/bin/env python3
"""
Comprehensive OmniDocBench Benchmark Runner across 8 OCR Pipeline Targets:
1. RustO + PP-OCRv6 Tiny & Small (Base)
2. RustO Full Pipeline + PP-OCRv6 Tiny & Small (with cls, orient, unwarp, layout)
3. RapidOCR + PP-OCRv6 Tiny & Small
4. PaddleOCR + PP-OCRv6 Tiny & Small
5. ocrs (RTen original by Robert Knight)
6. Indra OCR (official ONNX Runtime pipeline)
7. RustO + Indra models (Base: det + rec, top-to-bottom order)
8. RustO Full Pipeline + Indra models (orient + det + rec + order transformer)

Outputs:
- Predictions in benchmarks/omnidocbench/predictions/<pipeline>/
- Results in benchmarks/omnidocbench/results/benchmark_metrics.json
- Comprehensive report in benchmarks/omnidocbench/OMNIDOCBENCH_EVALUATION_REPORT.md
"""

import gc
import json
import logging
import os
import random
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add dist-packages if needed
for p in ["/usr/lib/python3/dist-packages", "/usr/local/lib/python3/dist-packages"]:
    if p not in sys.path and os.path.exists(p):
        sys.path.append(p)

import Levenshtein
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OMNIDOC_DIR = REPO_ROOT / "benchmarks" / "omnidocbench"
DATASET_DIR = OMNIDOC_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
GT_PATH = DATASET_DIR / "OmniDocBench.json"
MODELS_DIR = REPO_ROOT / "models"
PRED_DIR = OMNIDOC_DIR / "predictions"
RESULTS_DIR = OMNIDOC_DIR / "results"
REPORT_PATH = OMNIDOC_DIR / "OMNIDOCBENCH_EVALUATION_REPORT.md"
RUSTO_BIN = REPO_ROOT / "target" / "release" / "rusto"
if not RUSTO_BIN.exists():
    RUSTO_BIN = REPO_ROOT / "target" / "release" / "rusto-rs"
OCRS_BIN = OMNIDOC_DIR / "ocrs_bin" / "bin" / "ocrs"
VENV_BIN = OMNIDOC_DIR / ".venv" / "bin"

PRED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Suppress noisy library logs
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddlex").setLevel(logging.ERROR)


def load_ground_truth():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def select_stratified_sample(gt_data, sample_size=30):
    english = [
        s for s in gt_data
        if s.get("page_info", {}).get("page_attribute", {}).get("language") == "english"
    ]
    by_type = defaultdict(list)
    for s in english:
        src = s.get("page_info", {}).get("page_attribute", {}).get("data_source", "unknown")
        img_name = s.get("page_info", {}).get("image_path")
        if (IMAGES_DIR / img_name).exists():
            by_type[src].append(s)

    usatoday_target = "newspaper_USAToday-2025-01-08@magazinesclubnew_page_012.png"
    usatoday = [s for s in english if s.get("page_info", {}).get("image_path") == usatoday_target][0]

    rng = random.Random(42)
    sampled = [usatoday]

    proportions = {
        "academic_literature": 7,
        "book": 5,
        "PPT2PDF": 4,
        "magazine": 4,
        "exam_paper": 4,
        "newspaper": 3,
        "colorful_textbook": 2,
    }

    for src, count in proportions.items():
        pool = [s for s in by_type[src] if s.get("page_info", {}).get("image_path") != usatoday_target]
        sampled.extend(rng.sample(pool, min(count, len(pool))))

    return sampled[:sample_size]


def get_ground_truth_text(sample):
    gt_blocks = [
        d for d in sample.get("layout_dets", [])
        if d.get("category_type") == "text_block" and not d.get("ignore", False)
    ]
    gt_blocks.sort(key=lambda x: x.get("order", 0))
    raw_text = "\n".join(b.get("text", "").strip() for b in gt_blocks if b.get("text", "").strip())
    return " ".join(raw_text.split())


def compute_ned(gt_norm: str, pred_norm: str) -> float:
    if gt_norm == pred_norm:
        return 0.0
    len1, len2 = len(gt_norm), len(pred_norm)
    if len1 == 0 or len2 == 0:
        return 1.0
    return Levenshtein.distance(gt_norm, pred_norm) / max(len1, len2)


def get_binary_size_mb(path: Path) -> float:
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0.0


def get_model_size_mb(paths: list[Path]) -> float:
    total = 0
    for p in paths:
        if p.exists():
            total += p.stat().st_size
    return total / (1024 * 1024)


def measure_peak_rss_mb(command: list[str]) -> float:
    time_cmd = ["/usr/bin/time", "-v"] + command
    try:
        proc = subprocess.run(time_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in proc.stderr.splitlines():
            if "Maximum resident set size" in line:
                kbytes = float(line.split(":")[-1].strip())
                return kbytes / 1024.0
    except Exception as e:
        print(f"Error measuring RSS: {e}")
    return 0.0


class BenchmarkRunner:
    def __init__(self, samples):
        self.samples = samples
        self.gt_texts = {s["page_info"]["image_path"]: get_ground_truth_text(s) for s in samples}

    def run_cli_pipeline(self, pipeline_key, make_cmd, warmup_img):
        pred_dir = PRED_DIR / pipeline_key
        pred_dir.mkdir(parents=True, exist_ok=True)

        # 1. Warmup
        warmup_cmd = make_cmd(str(IMAGES_DIR / warmup_img))
        subprocess.run(warmup_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 2. Measure Peak RSS on sample image
        peak_rss = measure_peak_rss_mb(warmup_cmd)

        timings = []
        neds = []
        page_neds = {}

        print(f"\n===> Running [{pipeline_key}] across {len(self.samples)} pages...")
        for idx, s in enumerate(self.samples, 1):
            img_name = s["page_info"]["image_path"]
            img_path = IMAGES_DIR / img_name
            stem = Path(img_name).stem
            pred_file = pred_dir / f"{stem}.txt"

            cmd = make_cmd(str(img_path))
            t0 = time.perf_counter()
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings.append(elapsed_ms)

            pred_text = res.stdout.strip()
            with open(pred_file, "w", encoding="utf-8") as f:
                f.write(pred_text)

            pred_norm = " ".join(pred_text.split())
            gt_norm = self.gt_texts.get(img_name, "")
            ned = compute_ned(gt_norm, pred_norm)
            neds.append(ned)
            page_neds[img_name] = ned

            if idx % 10 == 0 or idx == len(self.samples) or "USAToday" in img_name:
                print(f"  [{idx}/{len(self.samples)}] {img_name[:35]:35s} | {elapsed_ms:6.1f}ms | NED: {ned:.4f}")

        return timings, neds, peak_rss, page_neds

    def run_rapidocr(self, pipeline_key, tier, warmup_img):
        from rapidocr_onnxruntime import RapidOCR
        pred_dir = PRED_DIR / pipeline_key
        pred_dir.mkdir(parents=True, exist_ok=True)

        model_dir = MODELS_DIR / f"PPOCR_v6_{tier}"
        engine = RapidOCR(
            det_model_path=str(model_dir / "det.onnx"),
            rec_model_path=str(model_dir / "rec.onnx"),
            rec_keys_path=str(model_dir / "dict.txt"),
        )

        # Warmup
        engine(str(IMAGES_DIR / warmup_img))

        # Peak RSS
        python_bin = str(VENV_BIN / "python")
        test_script = (
            f"from rapidocr_onnxruntime import RapidOCR; "
            f"e = RapidOCR(det_model_path='{model_dir}/det.onnx', rec_model_path='{model_dir}/rec.onnx', rec_keys_path='{model_dir}/dict.txt'); "
            f"e('{IMAGES_DIR / warmup_img}')"
        )
        peak_rss = measure_peak_rss_mb([python_bin, "-c", test_script])

        timings = []
        neds = []
        page_neds = {}

        print(f"\n===> Running [{pipeline_key}] across {len(self.samples)} pages...")
        for idx, s in enumerate(self.samples, 1):
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
            gt_norm = self.gt_texts.get(img_name, "")
            ned = compute_ned(gt_norm, pred_norm)
            neds.append(ned)
            page_neds[img_name] = ned

            if idx % 10 == 0 or idx == len(self.samples) or "USAToday" in img_name:
                print(f"  [{idx}/{len(self.samples)}] {img_name[:35]:35s} | {elapsed_ms:6.1f}ms | NED: {ned:.4f}")

        del engine
        gc.collect()
        return timings, neds, peak_rss, page_neds

    def run_paddleocr(self, pipeline_key, tier, warmup_img):
        from paddleocr import PaddleOCR
        pred_dir = PRED_DIR / pipeline_key
        pred_dir.mkdir(parents=True, exist_ok=True)

        model_dir = MODELS_DIR / f"PPOCR_v6_{tier}"
        ocr = PaddleOCR(
            use_onnx=True,
            det_model_dir=str(model_dir / "det.onnx"),
            rec_model_dir=str(model_dir / "rec.onnx"),
            rec_char_dict_path=str(model_dir / "dict.txt"),
            use_angle_cls=False,
            show_log=False,
        )

        # Warmup
        ocr.ocr(str(IMAGES_DIR / warmup_img), cls=False)

        # Peak RSS
        python_bin = str(VENV_BIN / "python")
        test_script = (
            f"from paddleocr import PaddleOCR; "
            f"o = PaddleOCR(use_onnx=True, det_model_dir='{model_dir}/det.onnx', rec_model_dir='{model_dir}/rec.onnx', rec_char_dict_path='{model_dir}/dict.txt', use_angle_cls=False, show_log=False); "
            f"o.ocr('{IMAGES_DIR / warmup_img}', cls=False)"
        )
        peak_rss = measure_peak_rss_mb([python_bin, "-c", test_script])

        timings = []
        neds = []
        page_neds = {}

        print(f"\n===> Running [{pipeline_key}] across {len(self.samples)} pages...")
        for idx, s in enumerate(self.samples, 1):
            img_name = s["page_info"]["image_path"]
            img_path = IMAGES_DIR / img_name
            stem = Path(img_name).stem
            pred_file = pred_dir / f"{stem}.txt"

            t0 = time.perf_counter()
            res = ocr.ocr(str(img_path), cls=False)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings.append(elapsed_ms)

            lines = [r[1][0] for r in res[0]] if res and res[0] else []
            pred_text = "\n".join(lines)
            with open(pred_file, "w", encoding="utf-8") as f:
                f.write(pred_text)

            pred_norm = " ".join(pred_text.split())
            gt_norm = self.gt_texts.get(img_name, "")
            ned = compute_ned(gt_norm, pred_norm)
            neds.append(ned)
            page_neds[img_name] = ned

            if idx % 10 == 0 or idx == len(self.samples) or "USAToday" in img_name:
                print(f"  [{idx}/{len(self.samples)}] {img_name[:35]:35s} | {elapsed_ms:6.1f}ms | NED: {ned:.4f}")

        del ocr
        gc.collect()
        return timings, neds, peak_rss, page_neds

    def run_indra_inprocess(self, pipeline_key, mode, warmup_img):
        from indra_ocr.pipeline import OrtOCR
        pred_dir = PRED_DIR / pipeline_key
        pred_dir.mkdir(parents=True, exist_ok=True)

        engine = OrtOCR(str(MODELS_DIR / "indra"))

        # Warmup
        img_warm = Image.open(str(IMAGES_DIR / warmup_img)).convert("L")
        gray_warm = np.array(img_warm)
        if mode == "full":
            engine(gray_warm, turn=True)
        else:
            db = engine.detect(gray_warm)
            engine.recognize(gray_warm, db)

        # Peak RSS
        python_bin = str(VENV_BIN / "python")
        if mode == "full":
            test_script = (
                f"from indra_ocr.pipeline import OrtOCR; from PIL import Image; import numpy as np; "
                f"e = OrtOCR('{MODELS_DIR / 'indra'}'); "
                f"g = np.array(Image.open('{IMAGES_DIR / warmup_img}').convert('L')); "
                f"e(g, turn=True)"
            )
        else:
            test_script = (
                f"from indra_ocr.pipeline import OrtOCR; from PIL import Image; import numpy as np; "
                f"e = OrtOCR('{MODELS_DIR / 'indra'}'); "
                f"g = np.array(Image.open('{IMAGES_DIR / warmup_img}').convert('L')); "
                f"d = e.detect(g); e.recognize(g, d)"
            )
        peak_rss = measure_peak_rss_mb([python_bin, "-c", test_script])

        timings = []
        neds = []
        page_neds = {}

        print(f"\n===> Running [{pipeline_key}] across {len(self.samples)} pages...")
        for idx, s in enumerate(self.samples, 1):
            img_name = s["page_info"]["image_path"]
            img_path = IMAGES_DIR / img_name
            stem = Path(img_name).stem
            pred_file = pred_dir / f"{stem}.txt"

            img = Image.open(str(img_path)).convert("L")
            gray = np.array(img)

            t0 = time.perf_counter()
            if mode == "full":
                res = engine(gray, turn=True)
                pred_text = res.text
            else:
                det_boxes = engine.detect(gray)
                rec_res = engine.recognize(gray, det_boxes)
                # Base pipeline: Standard top-to-bottom reading order by (y0, x0)
                pairs = sorted(zip(det_boxes, rec_res), key=lambda p: (p[0][1], p[0][0]))
                pred_text = "\n".join(r[0] for b, r in pairs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings.append(elapsed_ms)

            with open(pred_file, "w", encoding="utf-8") as f:
                f.write(pred_text)

            pred_norm = " ".join(pred_text.split())
            gt_norm = self.gt_texts.get(img_name, "")
            ned = compute_ned(gt_norm, pred_norm)
            neds.append(ned)
            page_neds[img_name] = ned

            if idx % 10 == 0 or idx == len(self.samples) or "USAToday" in img_name:
                print(f"  [{idx}/{len(self.samples)}] {img_name[:35]:35s} | {elapsed_ms:6.1f}ms | NED: {ned:.4f}")

        del engine
        gc.collect()
        return timings, neds, peak_rss, page_neds

    def run_paddle_structure(self, pipeline_key, warmup_img):
        import cv2
        from paddleocr import PPStructure
        from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
        pred_dir = PRED_DIR / pipeline_key
        pred_dir.mkdir(parents=True, exist_ok=True)

        engine = PPStructure(table=False, ocr=True, show_log=False, lang="en")

        # Warmup
        warm_mat = cv2.imread(str(IMAGES_DIR / warmup_img))
        if warm_mat is not None:
            engine(warm_mat)

        python_bin = str(VENV_BIN / "python")
        test_script = (
            f"import cv2; from paddleocr import PPStructure; "
            f"e = PPStructure(table=False, ocr=True, show_log=False, lang='en'); "
            f"img = cv2.imread('{IMAGES_DIR / warmup_img}'); "
            f"e(img)"
        )
        peak_rss = measure_peak_rss_mb([python_bin, "-c", test_script])

        timings = []
        neds = []
        page_neds = {}

        print(f"\n===> Running [{pipeline_key}] across {len(self.samples)} pages...")
        for idx, s in enumerate(self.samples, 1):
            img_name = s["page_info"]["image_path"]
            img_path = IMAGES_DIR / img_name
            stem = Path(img_name).stem
            pred_file = pred_dir / f"{stem}.txt"

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            t0 = time.perf_counter()
            res = engine(img)
            sorted_res = sorted_layout_boxes(res, w)
            lines = []
            for r in sorted_res:
                for l in r.get("res", []):
                    if isinstance(l, dict) and "text" in l:
                        lines.append(l["text"])
                    elif isinstance(l, tuple) and len(l) >= 2:
                        lines.append(l[1][0] if isinstance(l[1], tuple) else str(l[1]))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings.append(elapsed_ms)

            pred_text = "\n".join(lines)
            with open(pred_file, "w", encoding="utf-8") as f:
                f.write(pred_text)

            pred_norm = " ".join(pred_text.split())
            gt_norm = self.gt_texts.get(img_name, "")
            ned = compute_ned(gt_norm, pred_norm)
            neds.append(ned)
            page_neds[img_name] = ned

            if idx % 10 == 0 or idx == len(self.samples) or "USAToday" in img_name:
                print(f"  [{idx}/{len(self.samples)}] {img_name[:35]:35s} | {elapsed_ms:6.1f}ms | NED: {ned:.4f}")

        del engine
        gc.collect()
        return timings, neds, peak_rss, page_neds

    def run_rapid_structure(self, pipeline_key, tier, warmup_img):
        import cv2
        from rapidocr_onnxruntime import RapidOCR
        from rapid_layout import RapidLayout
        from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
        pred_dir = PRED_DIR / pipeline_key
        pred_dir.mkdir(parents=True, exist_ok=True)

        model_dir = MODELS_DIR / f"PPOCR_v6_{tier}"
        layout_engine = RapidLayout(model_type="pp_layout_publaynet")
        ocr_engine = RapidOCR(
            det_model_path=str(model_dir / "det.onnx"),
            rec_model_path=str(model_dir / "rec.onnx"),
            rec_keys_path=str(model_dir / "dict.txt"),
        )

        # Warmup
        warm_mat = cv2.imread(str(IMAGES_DIR / warmup_img))
        if warm_mat is not None:
            layout_engine(warm_mat)
            ocr_engine(str(IMAGES_DIR / warmup_img))

        python_bin = str(VENV_BIN / "python")
        test_script = (
            f"import cv2; from rapidocr_onnxruntime import RapidOCR; from rapid_layout import RapidLayout; "
            f"l = RapidLayout(model_type='pp_layout_publaynet'); "
            f"o = RapidOCR(det_model_path='{model_dir}/det.onnx', rec_model_path='{model_dir}/rec.onnx', rec_keys_path='{model_dir}/dict.txt'); "
            f"img = cv2.imread('{IMAGES_DIR / warmup_img}'); "
            f"l(img); o('{IMAGES_DIR / warmup_img}')"
        )
        peak_rss = measure_peak_rss_mb([python_bin, "-c", test_script])

        timings = []
        neds = []
        page_neds = {}

        print(f"\n===> Running [{pipeline_key}] across {len(self.samples)} pages...")
        for idx, s in enumerate(self.samples, 1):
            img_name = s["page_info"]["image_path"]
            img_path = IMAGES_DIR / img_name
            stem = Path(img_name).stem
            pred_file = pred_dir / f"{stem}.txt"

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            t0 = time.perf_counter()
            r_layout = layout_engine(img)
            r_ocr, _ = ocr_engine(str(img_path))

            regions = [
                {"type": c, "bbox": [int(b[0]), int(b[1]), int(b[2]), int(b[3])], "res": []}
                for b, c in zip(r_layout.boxes, r_layout.class_names)
            ]
            for line in (r_ocr or []):
                cx = sum(pt[0] for pt in line[0]) / 4.0
                cy = sum(pt[1] for pt in line[0]) / 4.0
                m = False
                for reg in regions:
                    bx1, by1, bx2, by2 = reg["bbox"]
                    if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                        reg["res"].append(line)
                        m = True
                        break
                if not m:
                    min_x = min(pt[0] for pt in line[0])
                    min_y = min(pt[1] for pt in line[0])
                    max_x = max(pt[0] for pt in line[0])
                    max_y = max(pt[1] for pt in line[0])
                    regions.append({
                        "type": "text",
                        "bbox": [int(min_x), int(min_y), int(max_x), int(max_y)],
                        "res": [line],
                    })

            r_sorted = sorted_layout_boxes(regions, w)
            lines = []
            for reg in r_sorted:
                sorted_lines = sorted(reg["res"], key=lambda item: (item[0][0][1], item[0][0][0]))
                for sl in sorted_lines:
                    lines.append(sl[1])

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings.append(elapsed_ms)

            pred_text = "\n".join(lines)
            with open(pred_file, "w", encoding="utf-8") as f:
                f.write(pred_text)

            pred_norm = " ".join(pred_text.split())
            gt_norm = self.gt_texts.get(img_name, "")
            ned = compute_ned(gt_norm, pred_norm)
            neds.append(ned)
            page_neds[img_name] = ned

            if idx % 10 == 0 or idx == len(self.samples) or "USAToday" in img_name:
                print(f"  [{idx}/{len(self.samples)}] {img_name[:35]:35s} | {elapsed_ms:6.1f}ms | NED: {ned:.4f}")

        del layout_engine
        del ocr_engine
        gc.collect()
        return timings, neds, peak_rss, page_neds


def compute_stats(timings, neds):
    s_timings = sorted(timings)
    s_neds = sorted(neds)
    n = len(timings)
    return {
        "mean_latency_ms": round(sum(timings) / n, 2),
        "median_latency_ms": round(s_timings[n // 2], 2),
        "p95_latency_ms": round(s_timings[min(int(n * 0.95), n - 1)], 2),
        "min_latency_ms": round(s_timings[0], 2),
        "max_latency_ms": round(s_timings[-1], 2),
        "mean_ned": round(sum(neds) / n, 4),
        "median_ned": round(s_neds[n // 2], 4),
        "p95_ned": round(s_neds[min(int(n * 0.95), n - 1)], 4),
        "accuracy_pct": round((1.0 - (sum(neds) / n)) * 100.0, 2),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OmniDocBench Benchmark Runner")
    parser.add_argument("--targets", type=str, default="", help="Comma-separated target keys to run, or empty for all")
    args = parser.parse_args()
    selected_targets = set(t.strip() for t in args.targets.split(",") if t.strip())

    print("=" * 70)
    print("       OMNIDOCBENCH COMPREHENSIVE BENCHMARK EVALUATION")
    print("=" * 70)

    gt_data = load_ground_truth()
    samples = select_stratified_sample(gt_data, sample_size=30)
    print(f"Selected stratified sample of {len(samples)} representative documents.")

    warmup_img = "newspaper_USAToday-2025-01-08@magazinesclubnew_page_012.png"
    runner = BenchmarkRunner(samples)
    results = {}
    if (RESULTS_DIR / "benchmark_metrics.json").exists():
        try:
            with open(RESULTS_DIR / "benchmark_metrics.json", "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            pass

    # Define all model file lists for disk weight measurement
    models_ppocr_tiny_mnn = [
        MODELS_DIR / "PPOCR_v6_tiny" / "det.mnn",
        MODELS_DIR / "PPOCR_v6_tiny" / "rec.mnn",
        MODELS_DIR / "PPOCR_v6_tiny" / "dict.txt",
    ]
    models_opts_mnn = [
        MODELS_DIR / "PPOCR_v6_tiny" / "cls.mnn",
        MODELS_DIR / "PPOCR_v6_tiny" / "orient.mnn",
        MODELS_DIR / "PPOCR_v6_tiny" / "uvdoc.mnn",
        MODELS_DIR / "PPOCR_v6_tiny" / "layout_cdla.mnn",
    ]
    models_ppocr_tiny = [
        MODELS_DIR / "PPOCR_v6_tiny" / "det.onnx",
        MODELS_DIR / "PPOCR_v6_tiny" / "rec.onnx",
        MODELS_DIR / "PPOCR_v6_tiny" / "dict.txt",
    ]
    models_ppocr_small = [
        MODELS_DIR / "PPOCR_v6_small" / "det.onnx",
        MODELS_DIR / "PPOCR_v6_small" / "rec.onnx",
        MODELS_DIR / "PPOCR_v6_small" / "dict.txt",
    ]
    models_opts = [
        MODELS_DIR / "PPOCR_v6_tiny" / "cls.onnx",
        MODELS_DIR / "PPOCR_v6_tiny" / "orient.onnx",
        MODELS_DIR / "PPOCR_v6_tiny" / "uvdoc.onnx",
        MODELS_DIR / "PPOCR_v6_tiny" / "layout_cdla.onnx",
    ]
    models_ocrs = [
        Path.home() / ".cache" / "ocrs" / "text-detection.onnx",
        Path.home() / ".cache" / "ocrs" / "text-recognition.onnx",
    ]
    models_indra_base_mnn = [
        MODELS_DIR / "indra" / "det.mnn",
        MODELS_DIR / "indra" / "rec.mnn",
        MODELS_DIR / "indra" / "meta.json",
    ]
    models_indra_full_mnn = [
        MODELS_DIR / "indra" / "det.mnn",
        MODELS_DIR / "indra" / "orient.rten",
        MODELS_DIR / "indra" / "rec.mnn",
        MODELS_DIR / "indra" / "indra_order.rten",
        MODELS_DIR / "indra" / "meta.json",
    ]
    models_indra_base = [
        MODELS_DIR / "indra" / "det.rten",
        MODELS_DIR / "indra" / "rec.rten",
        MODELS_DIR / "indra" / "meta.json",
    ]
    models_indra_full = [
        MODELS_DIR / "indra" / "det.rten",
        MODELS_DIR / "indra" / "orient.rten",
        MODELS_DIR / "indra" / "rec.rten",
        MODELS_DIR / "indra" / "indra_order.rten",
        MODELS_DIR / "indra" / "meta.json",
    ]
    models_paddle_structure = [
        Path.home() / ".paddleocr" / "whl" / "det" / "en" / "en_PP-OCRv3_det_infer" / "inference.pdmodel",
        Path.home() / ".paddleocr" / "whl" / "det" / "en" / "en_PP-OCRv3_det_infer" / "inference.pdiparams",
        Path.home() / ".paddleocr" / "whl" / "rec" / "en" / "en_PP-OCRv4_rec_infer" / "inference.pdmodel",
        Path.home() / ".paddleocr" / "whl" / "rec" / "en" / "en_PP-OCRv4_rec_infer" / "inference.pdiparams",
        Path.home() / ".paddleocr" / "whl" / "layout" / "picodet_lcnet_x1_0_fgd_layout_infer" / "inference.pdmodel",
        Path.home() / ".paddleocr" / "whl" / "layout" / "picodet_lcnet_x1_0_fgd_layout_infer" / "inference.pdiparams",
    ]
    models_rapid_structure_tiny = [
        MODELS_DIR / "PPOCR_v6_tiny" / "det.onnx",
        MODELS_DIR / "PPOCR_v6_tiny" / "rec.onnx",
        MODELS_DIR / "PPOCR_v6_tiny" / "dict.txt",
        OMNIDOC_DIR / ".venv" / "lib" / "python3.12" / "site-packages" / "rapid_layout" / "models" / "layout_publaynet.onnx",
    ]
    models_rapid_structure_small = [
        MODELS_DIR / "PPOCR_v6_small" / "det.onnx",
        MODELS_DIR / "PPOCR_v6_small" / "rec.onnx",
        MODELS_DIR / "PPOCR_v6_small" / "dict.txt",
        OMNIDOC_DIR / ".venv" / "lib" / "python3.12" / "site-packages" / "rapid_layout" / "models" / "layout_publaynet.onnx",
    ]
    models_rusto_structure_tiny = [
        MODELS_DIR / "PPOCR_v6_tiny" / "det.mnn",
        MODELS_DIR / "PPOCR_v6_tiny" / "rec.mnn",
        MODELS_DIR / "PPOCR_v6_tiny" / "dict.txt",
        MODELS_DIR / "PPOCR_v6_tiny" / "layout_cdla.mnn",
    ]
    models_rusto_structure_small = [
        MODELS_DIR / "PPOCR_v6_small" / "det.onnx",
        MODELS_DIR / "PPOCR_v6_small" / "rec.onnx",
        MODELS_DIR / "PPOCR_v6_small" / "dict.txt",
        MODELS_DIR / "PPOCR_v6_tiny" / "layout_cdla.onnx",
    ]

    pipeline_metadata = {
        "rusto_tiny": {
            "name": "RustO + PP-OCRv6 Tiny (Base)",
            "target": "Target 1",
            "binary_size_mb": get_binary_size_mb(RUSTO_BIN),
            "model_size_mb": get_model_size_mb(models_ppocr_tiny_mnn),
            "engine_runtime": "Pure-Rust MNN (rusto-nn)",
        },
        "rusto_mnn_tiny": {
            "name": "RustO + PP-OCRv6 Tiny (Pure-Rust MNN)",
            "target": "Target 1 (MNN)",
            "binary_size_mb": get_binary_size_mb(RUSTO_BIN),
            "model_size_mb": get_model_size_mb(models_ppocr_tiny_mnn),
            "engine_runtime": "Pure-Rust MNN (rusto-nn)",
        },
        "rusto_rten_tiny": {
            "name": "RustO + PP-OCRv6 Tiny (RTen)",
            "target": "Target 1 (RTen)",
            "binary_size_mb": get_binary_size_mb(RUSTO_BIN),
            "model_size_mb": get_model_size_mb(models_ppocr_tiny),
            "engine_runtime": "Native Rust (RTen)",
        },
        "rusto_small": {
            "name": "RustO + PP-OCRv6 Small (Base)",
            "target": "Target 1",
            "binary_size_mb": get_binary_size_mb(RUSTO_BIN),
            "model_size_mb": get_model_size_mb(models_ppocr_small),
            "engine_runtime": "Native Rust (RTen)",
        },
        "rusto_rten_small": {
            "name": "RustO + PP-OCRv6 Small (RTen)",
            "target": "Target 1 (RTen)",
            "binary_size_mb": get_binary_size_mb(RUSTO_BIN),
            "model_size_mb": get_model_size_mb(models_ppocr_small),
            "engine_runtime": "Native Rust (RTen)",
        },
        "rusto_structure_tiny": {
            "name": "RustO + PP-Structure Tiny",
            "target": "Target 1 (Structure)",
            "binary_size_mb": get_binary_size_mb(RUSTO_BIN),
            "model_size_mb": get_model_size_mb(models_rusto_structure_tiny),
            "engine_runtime": "Pure-Rust MNN (rusto-nn)",
        },
        "rusto_structure_small": {
            "name": "RustO + PP-Structure Small",
            "target": "Target 1 (Structure)",
            "binary_size_mb": get_binary_size_mb(RUSTO_BIN),
            "model_size_mb": get_model_size_mb(models_rusto_structure_small),
            "engine_runtime": "Native Rust (RTen)",
        },
        "rusto_full_tiny": {
            "name": "RustO Full Pipeline + PP-OCRv6 Tiny",
            "target": "Target 2",
            "binary_size_mb": get_binary_size_mb(RUSTO_BIN),
            "model_size_mb": get_model_size_mb(models_ppocr_tiny_mnn + models_opts_mnn),
            "engine_runtime": "Pure-Rust MNN (rusto-nn)",
        },
        "rusto_mnn_full_tiny": {
            "name": "RustO Full Pipeline + PP-OCRv6 Tiny (Pure-Rust MNN)",
            "target": "Target 2 (MNN)",
            "binary_size_mb": get_binary_size_mb(RUSTO_BIN),
            "model_size_mb": get_model_size_mb(models_ppocr_tiny_mnn + models_opts_mnn),
            "engine_runtime": "Pure-Rust MNN (rusto-nn)",
        },
        "rusto_full_small": {
            "name": "RustO Full Pipeline + PP-OCRv6 Small",
            "target": "Target 2",
            "binary_size_mb": get_binary_size_mb(RUSTO_BIN),
            "model_size_mb": get_model_size_mb(models_ppocr_small + models_opts),
            "engine_runtime": "Native Rust (RTen)",
        },
        "rapidocr_tiny": {
            "name": "RapidOCR + PP-OCRv6 Tiny",
            "target": "Target 3",
            "binary_size_mb": 22.7, # ONNX Runtime C++ shared lib
            "model_size_mb": get_model_size_mb(models_ppocr_tiny),
            "engine_runtime": "Python (ONNX Runtime)",
        },
        "rapidocr_small": {
            "name": "RapidOCR + PP-OCRv6 Small",
            "target": "Target 3",
            "binary_size_mb": 22.7,
            "model_size_mb": get_model_size_mb(models_ppocr_small),
            "engine_runtime": "Python (ONNX Runtime)",
        },
        "paddleocr_tiny": {
            "name": "PaddleOCR + PP-OCRv6 Tiny",
            "target": "Target 4",
            "binary_size_mb": 120.1, # PaddlePaddle shared runtime
            "model_size_mb": get_model_size_mb(models_ppocr_tiny),
            "engine_runtime": "Python (PaddleOCR ONNX)",
        },
        "paddleocr_small": {
            "name": "PaddleOCR + PP-OCRv6 Small",
            "target": "Target 4",
            "binary_size_mb": 120.1,
            "model_size_mb": get_model_size_mb(models_ppocr_small),
            "engine_runtime": "Python (PaddleOCR ONNX)",
        },
        "ocrs": {
            "name": "ocrs (RTen original)",
            "target": "Target 5",
            "binary_size_mb": get_binary_size_mb(OCRS_BIN),
            "model_size_mb": get_model_size_mb(models_ocrs),
            "engine_runtime": "Native Rust (RTen)",
        },
        "indra_ocr": {
            "name": "Indra OCR (Official Pipeline)",
            "target": "Target 6",
            "binary_size_mb": 22.7,
            "model_size_mb": get_model_size_mb(models_indra_full),
            "engine_runtime": "Python (ONNX Runtime)",
        },
        "rusto_indra_base": {
            "name": "RustO + Indra Models (Base)",
            "target": "Target 7",
            "binary_size_mb": get_binary_size_mb(RUSTO_BIN),
            "model_size_mb": get_model_size_mb(models_indra_base_mnn),
            "engine_runtime": "Pure-Rust MNN (rusto-nn)",
        },
        "rusto_mnn_indra_base": {
            "name": "RustO + Indra Models (Pure-Rust MNN Base)",
            "target": "Target 7 (MNN)",
            "binary_size_mb": get_binary_size_mb(RUSTO_BIN),
            "model_size_mb": get_model_size_mb(models_indra_base_mnn),
            "engine_runtime": "Pure-Rust MNN (rusto-nn)",
        },
        "rusto_rten_indra_base": {
            "name": "RustO + Indra Models (RTen Base)",
            "target": "Target 7 (RTen)",
            "binary_size_mb": get_binary_size_mb(RUSTO_BIN),
            "model_size_mb": get_model_size_mb(models_indra_base),
            "engine_runtime": "Native Rust (RTen)",
        },
        "rusto_indra_full": {
            "name": "RustO Full Pipeline + Indra Models",
            "target": "Target 8",
            "binary_size_mb": get_binary_size_mb(RUSTO_BIN),
            "model_size_mb": get_model_size_mb(models_indra_full_mnn),
            "engine_runtime": "Pure-Rust MNN (rusto-nn)",
        },
        "rusto_mnn_indra_full": {
            "name": "RustO Full Pipeline + Indra Models (Pure-Rust MNN)",
            "target": "Target 8 (MNN)",
            "binary_size_mb": get_binary_size_mb(RUSTO_BIN),
            "model_size_mb": get_model_size_mb(models_indra_full_mnn),
            "engine_runtime": "Pure-Rust MNN (rusto-nn)",
        },
        "rusto_rten_indra_full": {
            "name": "RustO Full Pipeline + Indra Models (RTen)",
            "target": "Target 8 (RTen)",
            "binary_size_mb": get_binary_size_mb(RUSTO_BIN),
            "model_size_mb": get_model_size_mb(models_indra_full),
            "engine_runtime": "Native Rust (RTen)",
        },
        "paddle_structure": {
            "name": "PaddleOCR + PP-Structure",
            "target": "Target 4 (Structure)",
            "binary_size_mb": 120.1,
            "model_size_mb": get_model_size_mb(models_paddle_structure),
            "engine_runtime": "Python (PaddleStructure)",
        },
        "rapid_structure_tiny": {
            "name": "RapidOCR + RapidLayout Tiny",
            "target": "Target 3 (Structure)",
            "binary_size_mb": 22.7,
            "model_size_mb": get_model_size_mb(models_rapid_structure_tiny),
            "engine_runtime": "Python (ONNX Runtime)",
        },
        "rapid_structure_small": {
            "name": "RapidOCR + RapidLayout Small",
            "target": "Target 3 (Structure)",
            "binary_size_mb": 22.7,
            "model_size_mb": get_model_size_mb(models_rapid_structure_small),
            "engine_runtime": "Python (ONNX Runtime)",
        },
    }

    # 1. Target 1: RustO Base (MNN for tiny, RTen/ONNX for small)
    for tier in ["tiny", "small"]:
        key = f"rusto_{tier}"
        if selected_targets and key not in selected_targets:
            continue
        m_dir = MODELS_DIR / f"PPOCR_v6_{tier}"
        det_file = "det.mnn" if tier == "tiny" else "det.onnx"
        rec_file = "rec.mnn" if tier == "tiny" else "rec.onnx"
        def make_cmd(path, m_dir=m_dir, det_file=det_file, rec_file=rec_file):
            return [
                str(RUSTO_BIN),
                "--det-model", str(m_dir / det_file),
                "--rec-model", str(m_dir / rec_file),
                "--dict", str(m_dir / "dict.txt"),
                "--det-limit", "960",
                "--format", "text",
                path,
            ]
        timings, neds, rss, page_neds = runner.run_cli_pipeline(key, make_cmd, warmup_img)
        stats = compute_stats(timings, neds)
        stats.update(pipeline_metadata[key])
        stats["peak_rss_mb"] = round(rss, 1)
        stats["usatoday_ned"] = round(page_neds.get(warmup_img, 1.0), 4)
        results[key] = stats

    # 1.1 Pure-Rust MNN (rusto-nn) vs RTen (rusto-rten) Explicit Targets
    if not selected_targets or "rusto_mnn_tiny" in selected_targets:
        m_dir = MODELS_DIR / "PPOCR_v6_tiny"
        def make_cmd(path, m_dir=m_dir):
            return [
                str(RUSTO_BIN),
                "--det-model", str(m_dir / "det.mnn"),
                "--rec-model", str(m_dir / "rec.mnn"),
                "--dict", str(m_dir / "dict.txt"),
                "--det-limit", "960",
                "--format", "text",
                path,
            ]
        timings, neds, rss, page_neds = runner.run_cli_pipeline("rusto_mnn_tiny", make_cmd, warmup_img)
        stats = compute_stats(timings, neds)
        stats.update(pipeline_metadata["rusto_mnn_tiny"])
        stats["peak_rss_mb"] = round(rss, 1)
        stats["usatoday_ned"] = round(page_neds.get(warmup_img, 1.0), 4)
        results["rusto_mnn_tiny"] = stats

    if not selected_targets or "rusto_rten_tiny" in selected_targets:
        m_dir = MODELS_DIR / "PPOCR_v6_tiny"
        def make_cmd(path, m_dir=m_dir):
            return [
                str(RUSTO_BIN),
                "--det-model", str(m_dir / "det.rten"),
                "--rec-model", str(m_dir / "rec.rten"),
                "--dict", str(m_dir / "dict.txt"),
                "--det-limit", "960",
                "--format", "text",
                path,
            ]
        timings, neds, rss, page_neds = runner.run_cli_pipeline("rusto_rten_tiny", make_cmd, warmup_img)
        stats = compute_stats(timings, neds)
        stats.update(pipeline_metadata["rusto_rten_tiny"])
        stats["peak_rss_mb"] = round(rss, 1)
        stats["usatoday_ned"] = round(page_neds.get(warmup_img, 1.0), 4)
        results["rusto_rten_tiny"] = stats

    if not selected_targets or "rusto_rten_small" in selected_targets:
        m_dir = MODELS_DIR / "PPOCR_v6_small"
        def make_cmd(path, m_dir=m_dir):
            return [
                str(RUSTO_BIN),
                "--det-model", str(m_dir / "det.onnx"),
                "--rec-model", str(m_dir / "rec.onnx"),
                "--dict", str(m_dir / "dict.txt"),
                "--det-limit", "960",
                "--format", "text",
                path,
            ]
        timings, neds, rss, page_neds = runner.run_cli_pipeline("rusto_rten_small", make_cmd, warmup_img)
        stats = compute_stats(timings, neds)
        stats.update(pipeline_metadata["rusto_rten_small"])
        stats["peak_rss_mb"] = round(rss, 1)
        stats["usatoday_ned"] = round(page_neds.get(warmup_img, 1.0), 4)
        results["rusto_rten_small"] = stats

    # 1.5 Target 1 (Structure): RustO + PP-Structure
    for tier in ["tiny", "small"]:
        key = f"rusto_structure_{tier}"
        if selected_targets and key not in selected_targets:
            continue
        m_dir = MODELS_DIR / f"PPOCR_v6_{tier}"
        opts_dir = MODELS_DIR / "PPOCR_v6_tiny"
        det_file = "det.mnn" if tier == "tiny" else "det.onnx"
        rec_file = "rec.mnn" if tier == "tiny" else "rec.onnx"
        layout_file = "layout_cdla.mnn" if tier == "tiny" else "layout_cdla.onnx"
        def make_cmd(path, m_dir=m_dir, opts_dir=opts_dir, det_file=det_file, rec_file=rec_file, layout_file=layout_file):
            return [
                str(RUSTO_BIN),
                "--det-model", str(m_dir / det_file),
                "--rec-model", str(m_dir / rec_file),
                "--dict", str(m_dir / "dict.txt"),
                "--layout-model", str(opts_dir / layout_file), "--enable-layout",
                "--det-limit", "960",
                "--format", "text",
                path,
            ]
        timings, neds, rss, page_neds = runner.run_cli_pipeline(key, make_cmd, warmup_img)
        stats = compute_stats(timings, neds)
        stats.update(pipeline_metadata[key])
        stats["peak_rss_mb"] = round(rss, 1)
        stats["usatoday_ned"] = round(page_neds.get(warmup_img, 1.0), 4)
        results[key] = stats

    # 2. Target 2: RustO Full Pipeline
    for tier in ["tiny", "small"]:
        key = f"rusto_full_{tier}"
        if selected_targets and key not in selected_targets:
            continue
        m_dir = MODELS_DIR / f"PPOCR_v6_{tier}"
        opts_dir = MODELS_DIR / "PPOCR_v6_tiny"
        det_file = "det.mnn" if tier == "tiny" else "det.onnx"
        rec_file = "rec.mnn" if tier == "tiny" else "rec.onnx"
        cls_file = "cls.mnn" if tier == "tiny" else "cls.onnx"
        orient_file = "orient.mnn" if tier == "tiny" else "orient.onnx"
        layout_file = "layout_cdla.mnn" if tier == "tiny" else "layout_cdla.onnx"
        def make_cmd(path, m_dir=m_dir, opts_dir=opts_dir, det_file=det_file, rec_file=rec_file, cls_file=cls_file, orient_file=orient_file, layout_file=layout_file):
            return [
                str(RUSTO_BIN),
                "--det-model", str(m_dir / det_file),
                "--rec-model", str(m_dir / rec_file),
                "--dict", str(m_dir / "dict.txt"),
                "--cls-model", str(opts_dir / cls_file), "--enable-cls",
                "--orient-model", str(opts_dir / orient_file), "--enable-orient",
                "--layout-model", str(opts_dir / layout_file), "--enable-layout",
                "--det-limit", "960",
                "--format", "text",
                path,
            ]
        timings, neds, rss, page_neds = runner.run_cli_pipeline(key, make_cmd, warmup_img)
        stats = compute_stats(timings, neds)
        stats.update(pipeline_metadata[key])
        stats["peak_rss_mb"] = round(rss, 1)
        stats["usatoday_ned"] = round(page_neds.get(warmup_img, 1.0), 4)
        results[key] = stats

    if not selected_targets or "rusto_mnn_full_tiny" in selected_targets:
        m_dir = MODELS_DIR / "PPOCR_v6_tiny"
        opts_dir = MODELS_DIR / "PPOCR_v6_tiny"
        def make_cmd(path, m_dir=m_dir, opts_dir=opts_dir):
            return [
                str(RUSTO_BIN),
                "--det-model", str(m_dir / "det.mnn"),
                "--rec-model", str(m_dir / "rec.mnn"),
                "--dict", str(m_dir / "dict.txt"),
                "--cls-model", str(opts_dir / "cls.mnn"), "--enable-cls",
                "--orient-model", str(opts_dir / "orient.mnn"), "--enable-orient",
                "--layout-model", str(opts_dir / "layout_cdla.mnn"), "--enable-layout",
                "--det-limit", "960",
                "--format", "text",
                path,
            ]
        timings, neds, rss, page_neds = runner.run_cli_pipeline("rusto_mnn_full_tiny", make_cmd, warmup_img)
        stats = compute_stats(timings, neds)
        stats.update(pipeline_metadata["rusto_mnn_full_tiny"])
        stats["peak_rss_mb"] = round(rss, 1)
        stats["usatoday_ned"] = round(page_neds.get(warmup_img, 1.0), 4)
        results["rusto_mnn_full_tiny"] = stats

    # 3. Target 3: RapidOCR
    for tier in ["tiny", "small"]:
        key = f"rapidocr_{tier}"
        if selected_targets and key not in selected_targets:
            continue
        timings, neds, rss, page_neds = runner.run_rapidocr(key, tier, warmup_img)
        stats = compute_stats(timings, neds)
        stats.update(pipeline_metadata[key])
        stats["peak_rss_mb"] = round(rss, 1)
        stats["usatoday_ned"] = round(page_neds.get(warmup_img, 1.0), 4)
        results[key] = stats

    # 4. Target 4: PaddleOCR
    for tier in ["tiny", "small"]:
        key = f"paddleocr_{tier}"
        if selected_targets and key not in selected_targets:
            continue
        timings, neds, rss, page_neds = runner.run_paddleocr(key, tier, warmup_img)
        stats = compute_stats(timings, neds)
        stats.update(pipeline_metadata[key])
        stats["peak_rss_mb"] = round(rss, 1)
        stats["usatoday_ned"] = round(page_neds.get(warmup_img, 1.0), 4)
        results[key] = stats

    # 5. Target 5: ocrs
    if not selected_targets or "ocrs" in selected_targets:
        def make_ocrs_cmd(path):
            return [str(OCRS_BIN), path]
        timings, neds, rss, page_neds = runner.run_cli_pipeline("ocrs", make_ocrs_cmd, warmup_img)
        stats = compute_stats(timings, neds)
        stats.update(pipeline_metadata["ocrs"])
        stats["peak_rss_mb"] = round(rss, 1)
        stats["usatoday_ned"] = round(page_neds.get(warmup_img, 1.0), 4)
        results["ocrs"] = stats

    # 6. Target 6: indra_ocr (official CLI)
    if not selected_targets or "indra_ocr" in selected_targets:
        def make_indra_cmd(path):
            return [
                str(VENV_BIN / "python"), "-m", "indra_ocr.cli",
                path, "--model-dir", str(MODELS_DIR / "indra"), "--format", "text"
            ]
        timings, neds, rss, page_neds = runner.run_cli_pipeline("indra_ocr", make_indra_cmd, warmup_img)
        stats = compute_stats(timings, neds)
        stats.update(pipeline_metadata["indra_ocr"])
        stats["peak_rss_mb"] = round(rss, 1)
        stats["usatoday_ned"] = round(page_neds.get(warmup_img, 1.0), 4)
        results["indra_ocr"] = stats

    # 7 & 8: Target 7 & 8 (Indra Base & Full) via Native RustO CLI (Pure-Rust MNN default and RTen)
    for engine_type, engine_key_base, engine_key_full in [
        ("mnn", "rusto_mnn_indra_base", "rusto_mnn_indra_full"),
        ("rten", "rusto_rten_indra_base", "rusto_rten_indra_full"),
    ]:
        det_ext = "mnn" if engine_type == "mnn" else "rten"
        rec_ext = "mnn" if engine_type == "mnn" else "rten"
        m_dir = MODELS_DIR / "indra"

        # Base (Det + Rec)
        if not selected_targets or engine_key_base in selected_targets or (engine_type == "mnn" and "rusto_indra_base" in selected_targets):
            def make_indra_base_cmd(path, m_dir=m_dir, det_ext=det_ext, rec_ext=rec_ext):
                return [
                    str(RUSTO_BIN),
                    "--det-model", str(m_dir / f"det.{det_ext}"),
                    "--rec-model", str(m_dir / f"rec.{rec_ext}"),
                    "--dict", str(m_dir / "meta.json"),
                    "--format", "text",
                    path,
                ]
            timings, neds, rss, page_neds = runner.run_cli_pipeline(engine_key_base, make_indra_base_cmd, warmup_img)
            stats = compute_stats(timings, neds)
            stats.update(pipeline_metadata[engine_key_base])
            stats["peak_rss_mb"] = round(rss, 1)
            stats["usatoday_ned"] = round(page_neds.get(warmup_img, 1.0), 4)
            results[engine_key_base] = stats
            if engine_type == "mnn":
                results["rusto_indra_base"] = dict(stats)
                results["rusto_indra_base"].update(pipeline_metadata["rusto_indra_base"])

        # Full (Det + Rec + Orient + Reading Order)
        if not selected_targets or engine_key_full in selected_targets or (engine_type == "mnn" and "rusto_indra_full" in selected_targets):
            def make_indra_full_cmd(path, m_dir=m_dir, det_ext=det_ext, rec_ext=rec_ext):
                return [
                    str(RUSTO_BIN),
                    "--det-model", str(m_dir / f"det.{det_ext}"),
                    "--rec-model", str(m_dir / f"rec.{rec_ext}"),
                    "--dict", str(m_dir / "meta.json"),
                    "--orient-model", str(m_dir / "orient.rten"), "--enable-orient",
                    "--reading-order",
                    "--format", "text",
                    path,
                ]
            timings, neds, rss, page_neds = runner.run_cli_pipeline(engine_key_full, make_indra_full_cmd, warmup_img)
            stats = compute_stats(timings, neds)
            stats.update(pipeline_metadata[engine_key_full])
            stats["peak_rss_mb"] = round(rss, 1)
            stats["usatoday_ned"] = round(page_neds.get(warmup_img, 1.0), 4)
            results[engine_key_full] = stats
            if engine_type == "mnn":
                results["rusto_indra_full"] = dict(stats)
                results["rusto_indra_full"].update(pipeline_metadata["rusto_indra_full"])

    # 9. Target 9: PaddleOCR PP-Structure
    if not selected_targets or "paddle_structure" in selected_targets:
        timings, neds, rss, page_neds = runner.run_paddle_structure("paddle_structure", warmup_img=warmup_img)
        stats = compute_stats(timings, neds)
        stats.update(pipeline_metadata["paddle_structure"])
        stats["peak_rss_mb"] = round(rss, 1)
        stats["usatoday_ned"] = round(page_neds.get(warmup_img, 1.0), 4)
        results["paddle_structure"] = stats

    # 10. Target 10: RapidOCR + RapidLayout Structure
    for tier in ["tiny", "small"]:
        key = f"rapid_structure_{tier}"
        if selected_targets and key not in selected_targets:
            continue
        timings, neds, rss, page_neds = runner.run_rapid_structure(key, tier, warmup_img=warmup_img)
        stats = compute_stats(timings, neds)
        stats.update(pipeline_metadata[key])
        stats["peak_rss_mb"] = round(rss, 1)
        stats["usatoday_ned"] = round(page_neds.get(warmup_img, 1.0), 4)
        results[key] = stats

    # Save metrics JSON
    with open(RESULTS_DIR / "benchmark_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved metrics JSON to {RESULTS_DIR / 'benchmark_metrics.json'}")

    print("\n" + "=" * 70)
    print("                    FINAL BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<38} | {'Latency':<8} | {'P95':<8} | {'RSS (MB)':<8} | {'NED':<7} | {'Acc %':<6} | {'USAToday'}")
    print("-" * 96)
    for k, v in results.items():
        print(f"{v['name']:<38} | {v['mean_latency_ms']:6.1f}ms | {v['p95_latency_ms']:6.1f}ms | {v['peak_rss_mb']:7.1f}M | {v['mean_ned']:6.4f} | {v['accuracy_pct']:5.1f}% | {v['usatoday_ned']:6.4f}")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    main()

