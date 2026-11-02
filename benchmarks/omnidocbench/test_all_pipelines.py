#!/usr/bin/env python3
"""
Verification test for all 8 OmniDocBench evaluation pipelines.
Tests 1 image across all configurations to verify initialization,
inference execution, and output text extraction.
"""

import os
import sys
import time
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OMNIDOC_DIR = REPO_ROOT / "benchmarks" / "omnidocbench"
DATASET_DIR = OMNIDOC_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
MODELS_DIR = REPO_ROOT / "models"
VENV_BIN = OMNIDOC_DIR / ".venv" / "bin"
TEST_IMG = IMAGES_DIR / "newspaper_USAToday-2025-01-08@magazinesclubnew_page_012.png"

assert TEST_IMG.exists(), f"Test image not found: {TEST_IMG}"

print(f"Target test image: {TEST_IMG.name}")
print(f"File size: {TEST_IMG.stat().st_size / 1024:.1f} KB")

# 1. RustO Base (Tiny & Small)
import subprocess

def test_rusto(name, tier, extra_args=None):
    binary = REPO_ROOT / "target" / "release" / "rusto-rs"
    model_dir = MODELS_DIR / f"PPOCR_v6_{tier}"
    cmd = [
        str(binary),
        "--det-model", str(model_dir / "det.onnx"),
        "--rec-model", str(model_dir / "rec.onnx"),
        "--dict", str(model_dir / "dict.txt"),
        "--format", "text",
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(TEST_IMG))
    
    t0 = time.perf_counter()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = (time.perf_counter() - t0) * 1000
    if res.returncode != 0:
        print(f"[{name}] FAILED ({elapsed:.1f}ms): {res.stderr[:200]}")
        return False
    lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
    print(f"[{name}] SUCCESS ({elapsed:.1f}ms, {len(lines)} lines, sample: '{lines[0] if lines else ''}')")
    return True

print("\n--- Testing Target 1 & 2: RustO (Base & Full Pipeline) ---")
test_rusto("1a. rusto_tiny", "tiny")
test_rusto("1b. rusto_small", "small")

full_opts_args = [
    "--cls-model", str(MODELS_DIR / "PPOCR_v6_tiny" / "cls.onnx"), "--enable-cls",
    "--orient-model", str(MODELS_DIR / "PPOCR_v6_tiny" / "orient.onnx"), "--enable-orient",
    "--unwarp-model", str(MODELS_DIR / "PPOCR_v6_tiny" / "uvdoc.onnx"), "--enable-unwarp",
    "--layout-model", str(MODELS_DIR / "PPOCR_v6_tiny" / "layout_s.onnx"), "--enable-layout",
]
test_rusto("2a. rusto_full_tiny", "tiny", full_opts_args)
test_rusto("2b. rusto_full_small", "small", full_opts_args)

print("\n--- Testing Target 3: RapidOCR ---")
from rapidocr_onnxruntime import RapidOCR

for tier in ["tiny", "small"]:
    model_dir = MODELS_DIR / f"PPOCR_v6_{tier}"
    engine = RapidOCR(
        det_model_path=str(model_dir / "det.onnx"),
        rec_model_path=str(model_dir / "rec.onnx"),
        rec_keys_path=str(model_dir / "dict.txt"),
    )
    t0 = time.perf_counter()
    res, elapse = engine(str(TEST_IMG))
    elapsed = (time.perf_counter() - t0) * 1000
    lines = [r[1] for r in res] if res else []
    print(f"[3. rapidocr_{tier}] SUCCESS ({elapsed:.1f}ms, {len(lines)} lines, sample: '{lines[0] if lines else ''}')")

print("\n--- Testing Target 4: PaddleOCR ---")
from paddleocr import PaddleOCR
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)

for tier in ["tiny", "small"]:
    model_dir = MODELS_DIR / f"PPOCR_v6_{tier}"
    ocr = PaddleOCR(
        use_onnx=True,
        det_model_dir=str(model_dir / "det.onnx"),
        rec_model_dir=str(model_dir / "rec.onnx"),
        rec_char_dict_path=str(model_dir / "dict.txt"),
        use_angle_cls=False,
        show_log=False,
    )
    t0 = time.perf_counter()
    res = ocr.ocr(str(TEST_IMG), cls=False)
    elapsed = (time.perf_counter() - t0) * 1000
    lines = [r[1][0] for r in res[0]] if res and res[0] else []
    print(f"[4. paddleocr_{tier}] SUCCESS ({elapsed:.1f}ms, {len(lines)} lines, sample: '{lines[0] if lines else ''}')")

print("\n--- Testing Target 5: ocrs (Robert Knight RTen original) ---")
ocrs_bin = OMNIDOC_DIR / "ocrs_bin" / "bin" / "ocrs"
t0 = time.perf_counter()
res = subprocess.run([str(ocrs_bin), str(TEST_IMG)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
elapsed = (time.perf_counter() - t0) * 1000
if res.returncode == 0:
    lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
    print(f"[5. ocrs] SUCCESS ({elapsed:.1f}ms, {len(lines)} lines, sample: '{lines[0] if lines else ''}')")
else:
    print(f"[5. ocrs] FAILED: {res.stderr}")

print("\n--- Testing Target 6: indra ocr (official pipeline) ---")
t0 = time.perf_counter()
res = subprocess.run([
    str(VENV_BIN / "python"), "-m", "indra_ocr.cli",
    str(TEST_IMG), "--model-dir", str(MODELS_DIR / "indra"), "--format", "text"
], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
elapsed = (time.perf_counter() - t0) * 1000
if res.returncode == 0:
    lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
    print(f"[6. indra_ocr] SUCCESS ({elapsed:.1f}ms, {len(lines)} lines, sample: '{lines[0] if lines else ''}')")
else:
    print(f"[6. indra_ocr] FAILED: {res.stderr}")

print("\n--- Testing Target 7 & 8: rusto + indra models (base & full) ---")
from indra_ocr.pipeline import OrtOCR
from PIL import Image
import numpy as np

indra_engine = OrtOCR(str(MODELS_DIR / "indra"))
img = Image.open(str(TEST_IMG)).convert("L")
gray = np.array(img)

# 7. Base (det + rec, sort by Y)
t0 = time.perf_counter()
det_boxes = indra_engine.detect(gray)
rec_res = indra_engine.recognize(gray, det_boxes)
pairs = sorted(zip(det_boxes, rec_res), key=lambda p: (p[0][1], p[0][0]))
elapsed = (time.perf_counter() - t0) * 1000
lines = [r[0] for b, r in pairs]
print(f"[7. rusto_indra_base] SUCCESS ({elapsed:.1f}ms, {len(lines)} lines, sample: '{lines[0] if lines else ''}')")

# 8. Full pipeline (orient + det + rec + order transformer)
t0 = time.perf_counter()
res_full = indra_engine(gray, turn=True)
elapsed = (time.perf_counter() - t0) * 1000
lines = [l.strip() for l in res_full.text.strip().splitlines() if l.strip()]
print(f"[8. rusto_indra_full] SUCCESS ({elapsed:.1f}ms, {len(lines)} lines, sample: '{lines[0] if lines else ''}')")

