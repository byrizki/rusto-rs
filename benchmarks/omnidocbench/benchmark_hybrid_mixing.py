#!/usr/bin/env python3
"""
Research Benchmark: Standalone & Mixed Usage of PP-OCR and Indra OCR Models.
Evaluates cross-model combinations across the 30 stratified OmniDocBench pages:
1. Standard PP-OCR (Det + Rec, 1D sort)
2. Hybrid: PP-OCR (Det + Rec) + Indra Neural Reading Order (order.onnx)
3. Standard PP-OCR Small (Det + Rec, 1D sort)
4. Hybrid: PP-OCR Small (Det + Rec) + Indra Neural Reading Order (order.onnx)
5. Hybrid: Indra Detection + PP-OCR Recognition (1D sort)
6. Hybrid: Indra Detection + PP-OCR Recognition + Indra Neural Reading Order (order.onnx)
7. Hybrid: PP-OCR Detection + Indra Recognition (1D sort)
8. Hybrid: PP-OCR Detection + Indra Recognition + Indra Neural Reading Order (order.onnx)
9. Indra Native Base (Det + Rec, 1D sort)
10. Indra Native Full Pipeline (Det + Orient + Rec + Order)
"""

import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import Levenshtein
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from indra_ocr.pipeline import OrtOCR

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OMNIDOC_DIR = REPO_ROOT / "benchmarks" / "omnidocbench"
DATASET_DIR = OMNIDOC_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
GT_PATH = DATASET_DIR / "OmniDocBench.json"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = OMNIDOC_DIR / "results"

logging.getLogger("ppocr").setLevel(logging.ERROR)


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


class HybridEvaluator:
    def __init__(self, samples):
        self.samples = samples
        self.warmup_img = "newspaper_USAToday-2025-01-08@magazinesclubnew_page_012.png"

        # Initialize base engines
        print("Initializing PP-OCRv6 Tiny...")
        self.pp_tiny = RapidOCR(
            det_model_path=str(MODELS_DIR / "PPOCR_v6_tiny" / "det.onnx"),
            rec_model_path=str(MODELS_DIR / "PPOCR_v6_tiny" / "rec.onnx"),
            rec_keys_path=str(MODELS_DIR / "PPOCR_v6_tiny" / "dict.txt"),
        )
        print("Initializing PP-OCRv6 Small...")
        self.pp_small = RapidOCR(
            det_model_path=str(MODELS_DIR / "PPOCR_v6_small" / "det.onnx"),
            rec_model_path=str(MODELS_DIR / "PPOCR_v6_small" / "rec.onnx"),
            rec_keys_path=str(MODELS_DIR / "PPOCR_v6_small" / "dict.txt"),
        )
        print("Initializing Indra OCR Engine...")
        self.indra = OrtOCR(str(MODELS_DIR / "indra"))

    def run_pipeline(self, pipeline_fn, name):
        print(f"\n========================================================")
        print(f" Running Evaluation: {name}")
        print(f"========================================================")
        # Warmup
        warm_path = IMAGES_DIR / self.warmup_img
        pipeline_fn(warm_path)

        timings = []
        neds = []
        usatoday_ned = 1.0

        for idx, sample in enumerate(self.samples, 1):
            img_name = sample["page_info"]["image_path"]
            img_path = IMAGES_DIR / img_name
            gt_text = get_ground_truth_text(sample)

            t0 = time.perf_counter()
            pred_text = pipeline_fn(img_path)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            pred_norm = " ".join(pred_text.split())
            ned = compute_ned(gt_text, pred_norm)

            timings.append(elapsed_ms)
            neds.append(ned)

            if img_name == self.warmup_img:
                usatoday_ned = ned

            if idx % 10 == 0 or idx == len(self.samples) or img_name == self.warmup_img:
                print(f"  [{idx:2d}/{len(self.samples)}] {img_name[:35]:35s} | {elapsed_ms:6.1f}ms | NED: {ned:.4f}")

        mean_lat = float(np.mean(timings))
        p95_lat = float(np.percentile(timings, 95))
        mean_ned = float(np.mean(neds))
        acc = (1.0 - mean_ned) * 100.0

        return {
            "name": name,
            "mean_latency_ms": round(mean_lat, 1),
            "p95_latency_ms": round(p95_lat, 1),
            "mean_ned": round(mean_ned, 4),
            "accuracy_pct": round(acc, 2),
            "usatoday_ned": round(usatoday_ned, 4),
        }

    # Pipeline 1: PP-OCR Tiny (Base 1D)
    def pp_tiny_base(self, img_path):
        res, _ = self.pp_tiny(str(img_path))
        if not res:
            return ""
        sorted_res = sorted(res, key=lambda r: (min(pt[1] for pt in r[0]), min(pt[0] for pt in r[0])))
        return " ".join(r[1] for r in sorted_res)

    # Pipeline 2: PP-OCR Tiny + Indra Order
    def pp_tiny_indra_order(self, img_path):
        img = Image.open(img_path)
        W, H = img.size
        res, _ = self.pp_tiny(str(img_path))
        if not res:
            return ""
        boxes, texts = [], []
        for item in res:
            pts = np.array(item[0])
            boxes.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
            texts.append(item[1])
        order, _, _ = self.indra.reading_order(np.array(boxes, dtype=np.float32), texts, (W, H))
        return " ".join(texts[i] for i in order)

    # Pipeline 3: PP-OCR Small (Base 1D)
    def pp_small_base(self, img_path):
        res, _ = self.pp_small(str(img_path))
        if not res:
            return ""
        sorted_res = sorted(res, key=lambda r: (min(pt[1] for pt in r[0]), min(pt[0] for pt in r[0])))
        return " ".join(r[1] for r in sorted_res)

    # Pipeline 4: PP-OCR Small + Indra Order
    def pp_small_indra_order(self, img_path):
        img = Image.open(img_path)
        W, H = img.size
        res, _ = self.pp_small(str(img_path))
        if not res:
            return ""
        boxes, texts = [], []
        for item in res:
            pts = np.array(item[0])
            boxes.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
            texts.append(item[1])
        order, _, _ = self.indra.reading_order(np.array(boxes, dtype=np.float32), texts, (W, H))
        return " ".join(texts[i] for i in order)

    # Pipeline 5: Indra Det + PP-OCR Tiny Rec (Base 1D)
    def indra_det_pp_tiny_rec_base(self, img_path):
        img = Image.open(img_path)
        gray = np.array(img.convert("L"))
        cv_img = cv2.imread(str(img_path))
        boxes = self.indra.detect(gray)
        if len(boxes) == 0:
            return ""
        # 1D Sort boxes
        sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        crops = []
        for b in sorted_boxes:
            x0, y0, x1, y1 = [int(round(v)) for v in b[:4]]
            crop = cv_img[y0:y1, x0:x1]
            crops.append(crop if crop.size > 0 else np.zeros((10, 10, 3), dtype=np.uint8))
        rec_res, _ = self.pp_tiny.text_rec(crops)
        return " ".join(r[0] for r in rec_res if r[0])

    # Pipeline 6: Indra Det + PP-OCR Tiny Rec + Indra Order
    def indra_det_pp_tiny_rec_indra_order(self, img_path):
        img = Image.open(img_path)
        W, H = img.size
        gray = np.array(img.convert("L"))
        cv_img = cv2.imread(str(img_path))
        boxes = self.indra.detect(gray)
        if len(boxes) == 0:
            return ""
        crops = []
        for b in boxes:
            x0, y0, x1, y1 = [int(round(v)) for v in b[:4]]
            crop = cv_img[y0:y1, x0:x1]
            crops.append(crop if crop.size > 0 else np.zeros((10, 10, 3), dtype=np.uint8))
        rec_res, _ = self.pp_tiny.text_rec(crops)
        texts = [r[0] for r in rec_res]
        order, _, _ = self.indra.reading_order(boxes, texts, (W, H))
        return " ".join(texts[i] for i in order if texts[i])

    # Pipeline 7: PP-OCR Tiny Det + Indra Rec (Base 1D)
    def pp_tiny_det_indra_rec_base(self, img_path):
        img = Image.open(img_path)
        gray = np.array(img.convert("L"))
        cv_img = cv2.imread(str(img_path))
        dt_boxes, _ = self.pp_tiny.text_det(cv_img)
        if dt_boxes is None or len(dt_boxes) == 0:
            return ""
        # Convert 4-point polygons to [N, 4]
        boxes_4 = []
        for pts in dt_boxes:
            boxes_4.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
        boxes_np = np.array(boxes_4, dtype=np.float32)
        sorted_indices = sorted(range(len(boxes_np)), key=lambda i: (boxes_np[i][1], boxes_np[i][0]))
        sorted_boxes = boxes_np[sorted_indices]
        rec_out = self.indra.recognize(gray, sorted_boxes)
        return " ".join(t[0] for t in rec_out if t[0])

    # Pipeline 8: PP-OCR Tiny Det + Indra Rec + Indra Order
    def pp_tiny_det_indra_rec_indra_order(self, img_path):
        img = Image.open(img_path)
        W, H = img.size
        gray = np.array(img.convert("L"))
        cv_img = cv2.imread(str(img_path))
        dt_boxes, _ = self.pp_tiny.text_det(cv_img)
        if dt_boxes is None or len(dt_boxes) == 0:
            return ""
        boxes_4 = []
        for pts in dt_boxes:
            boxes_4.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
        boxes_np = np.array(boxes_4, dtype=np.float32)
        rec_out = self.indra.recognize(gray, boxes_np)
        texts = [t[0] for t in rec_out]
        order, _, _ = self.indra.reading_order(boxes_np, texts, (W, H))
        return " ".join(texts[i] for i in order if texts[i])

    # Pipeline 9: Indra Native Base (Det + Rec, 1D sort)
    def indra_native_base(self, img_path):
        img = Image.open(img_path)
        gray = np.array(img.convert("L"))
        boxes = self.indra.detect(gray)
        if len(boxes) == 0:
            return ""
        sorted_boxes = np.array(sorted(boxes, key=lambda b: (b[1], b[0])), dtype=np.float32)
        rec_out = self.indra.recognize(gray, sorted_boxes)
        return " ".join(t[0] for t in rec_out if t[0])

    # Pipeline 10: Indra Native Full Pipeline
    def indra_native_full(self, img_path):
        img = Image.open(img_path)
        gray = np.array(img.convert("L"))
        res = self.indra(gray, turn=True)
        return " ".join(line.text for line in res.lines if line.text)


def main():
    gt_data = load_ground_truth()
    samples = select_stratified_sample(gt_data, sample_size=30)
    print(f"Loaded {len(samples)} stratified document evaluation pages.")

    evaluator = HybridEvaluator(samples)

    benchmarks = [
        # Base baselines
        (evaluator.pp_tiny_base, "1. PP-OCRv6 Tiny (Base Det+Rec, 1D sort)"),
        (evaluator.pp_small_base, "2. PP-OCRv6 Small (Base Det+Rec, 1D sort)"),
        (evaluator.indra_native_base, "3. Indra OCR (Base Det+Rec, 1D sort)"),
        # Hybrid Reading Order
        (evaluator.pp_tiny_indra_order, "4. Hybrid: PP-OCR Tiny (Det+Rec) + Indra Order"),
        (evaluator.pp_small_indra_order, "5. Hybrid: PP-OCR Small (Det+Rec) + Indra Order"),
        # Hybrid Detection & Recognition
        (evaluator.indra_det_pp_tiny_rec_base, "6. Hybrid: Indra Det + PP-OCR Tiny Rec (1D sort)"),
        (evaluator.indra_det_pp_tiny_rec_indra_order, "7. Hybrid: Indra Det + PP-OCR Tiny Rec + Indra Order"),
        (evaluator.pp_tiny_det_indra_rec_base, "8. Hybrid: PP-OCR Tiny Det + Indra Rec (1D sort)"),
        (evaluator.pp_tiny_det_indra_rec_indra_order, "9. Hybrid: PP-OCR Tiny Det + Indra Rec + Indra Order"),
        # Native Full
        (evaluator.indra_native_full, "10. Indra Native Full Pipeline (Det+Orient+Rec+Order)"),
    ]

    all_results = []
    for fn, name in benchmarks:
        res = evaluator.run_pipeline(fn, name)
        all_results.append(res)

    print("\n" + "=" * 90)
    print("           STANDALONE & HYBRID MODEL MIXING BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Pipeline Configuration':<50} | {'Latency':<8} | {'P95':<8} | {'NED':<7} | {'Acc %':<6} | {'USAToday'}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['name']:<50} | {r['mean_latency_ms']:6.1f}ms | {r['p95_latency_ms']:6.1f}ms | {r['mean_ned']:6.4f} | {r['accuracy_pct']:5.1f}% | {r['usatoday_ned']:6.4f}")
    print("=" * 90 + "\n")

    out_file = RESULTS_DIR / "hybrid_mixing_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Saved results to {out_file}")


if __name__ == "__main__":
    main()

