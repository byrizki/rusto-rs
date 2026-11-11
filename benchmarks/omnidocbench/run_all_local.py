#!/usr/bin/env python3
"""
Run OmniDocBench benchmark targets locally across Raw OCR and Full Pipeline suites matching .github/workflows/benchmark.yml
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_PYTHON = REPO_ROOT / "benchmarks" / "omnidocbench" / ".venv" / "bin" / "python3"
PYTHON_BIN = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
RUNNER_SCRIPT = REPO_ROOT / "benchmarks" / "omnidocbench" / "run_full_benchmark.py"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "omnidocbench" / "results"

RAW_TARGETS = [
    ("paddleocr_tiny", "(raw-a) Python PaddleOCR + PP-OCRv6 Tiny"),
    ("rapidocr_tiny", "(raw-b) Python RapidOCR + PP-OCRv6 Tiny"),
    ("mnn_ocr_tiny", "(raw-c) MNN OCR Standalone (Direct) + PP-OCRv6"),
    ("rten_ocr_direct", "(raw-d) RTen OCR Direct + PP-OCRv6"),
    ("ocrs", "(raw-e) RTen + ocrs CLI"),
    ("rusto_nn_ppocr", "(raw-f) Rusto NN OCR (Pure-Rust MNN) + PP-OCRv6"),
    ("rusto_rten_tiny", "(raw-g) RTen OCR (Rusto Binding) + PP-OCRv6"),
    ("rusto_mnn_tiny", "(raw-h) MNN OCR (Rusto Binding) + PP-OCRv6"),
    ("rusto_nn_indra", "(raw-i) Rusto NN OCR (Pure-Rust MNN) + Indra OCR (Base)"),
    ("rusto_rten_indra", "(raw-j) RTen OCR (Rusto Binding) + Indra OCR (Base)"),
    ("rusto_mnn_indra", "(raw-k) MNN OCR (Rusto Binding) + Indra OCR (Base)"),
    ("tesseract3", "(raw-l) Tesseract OCR (v5 / PSM 3 Fully Automatic)"),
    ("rusto_nn_tiny_fp16", "(raw-m) Rusto NN OCR (Pure-Rust MNN) + PP-OCRv6 Tiny FP16"),
    ("rusto_nn_tiny_int8", "(raw-n) Rusto NN OCR (Pure-Rust MNN) + PP-OCRv6 Tiny INT8"),
    ("rusto_rten_tiny_int8", "(raw-o) RTen OCR (Rusto Binding) + PP-OCRv6 Tiny INT8"),
    ("rusto_nn_small_int8", "(raw-p) Rusto NN OCR (Pure-Rust MNN) + PP-OCRv6 Small INT8"),
    ("rusto_nn_medium_int8", "(raw-q) Rusto NN OCR (Pure-Rust MNN) + PP-OCRv6 Medium INT8"),
    ("rapidocr_tiny_fp16", "(raw-r) Python RapidOCR + PP-OCRv6 Tiny FP16"),
    ("rapidocr_tiny_int8", "(raw-s) Python RapidOCR + PP-OCRv6 Tiny INT8"),
]

FULL_PIPELINE_TARGETS = [
    ("indra_ocr", "(pipe-a) Python Indra OCR (Official CLI Pipeline)"),
    ("paddle_structure", "(pipe-b) Python PaddleOCR + PP-Structure (PicoDet Layout)"),
    ("rapid_structure_tiny", "(pipe-c) Python RapidOCR + RapidLayout Tiny (PubLayNet)"),
    ("rusto_full_tiny", "(pipe-d) RustO NN Full Pipeline + PP-OCRv6 (CDLA Layout)"),
    ("rusto_rten_full_tiny", "(pipe-e) RustO RTen Full Pipeline + PP-OCRv6 (CDLA Layout)"),
    ("rusto_full_publaynet", "(pipe-f) RustO NN Full Pipeline + PP-OCRv6 (PubLayNet)"),
    ("rusto_rten_full_publaynet", "(pipe-g) RustO RTen Full Pipeline + PP-OCRv6 (PubLayNet)"),
    ("rusto_indra_full", "(pipe-h) RustO NN Full Pipeline + Indra Models (Indra Order)"),
    ("rusto_rten_indra_full", "(pipe-i) RustO RTen Full Pipeline + Indra Models (Indra Order)"),
    ("rusto_order_heuristic", "(pipe-j) RustO NN + Pure-Rust Heuristic Order"),
    ("rusto_rten_order_heuristic", "(pipe-k) RustO RTen + Pure-Rust Heuristic Order"),
]

def main():
    parser = argparse.ArgumentParser(description="Run OmniDocBench benchmark targets locally")
    parser.add_argument("--suite", choices=["all", "raw-ocr", "full-pipeline"], default="all", help="Suite category to evaluate")
    parser.add_argument("--sample-size", type=str, default="30", help="Sample size (default: 30)")
    parser.add_argument("--language", choices=["english", "all"], default="english", help="Language subset")
    args = parser.parse_args()

    if args.suite == "raw-ocr":
        active_targets = RAW_TARGETS
    elif args.suite == "full-pipeline":
        active_targets = FULL_PIPELINE_TARGETS
    else:
        active_targets = RAW_TARGETS + FULL_PIPELINE_TARGETS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    print("=" * 80)
    print(f"STARTING LOCAL OMNIDOCBENCH BENCHMARK (SUITE: {args.suite.upper()}, {len(active_targets)} TARGETS)")
    print("=" * 80)

    for idx, (target_key, target_name) in enumerate(active_targets, 1):
        output_file = RESULTS_DIR / f"metric_{target_key}.json"
        print(f"\n[{idx}/{len(active_targets)}] Running Target: {target_name} ({target_key})...")
        t0 = time.perf_counter()

        cmd = [
            PYTHON_BIN,
            str(RUNNER_SCRIPT),
            "--targets", target_key,
            "--sample-size", args.sample_size,
            "--language", args.language,
            "--output-file", str(output_file),
        ]

        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            print(f"✗ Target {target_key} FAILED with code {proc.returncode} in {elapsed:.1f}s")
        else:
            print(f"✓ Target {target_key} completed in {elapsed:.1f}s -> {output_file.name}")

    print("\n" + "=" * 80)
    print("ALL TARGETS FINISHED. AGGREGATING RESULTS & GENERATING REPORT...")
    print("=" * 80)

    merge_cmd = [
        PYTHON_BIN,
        str(RUNNER_SCRIPT),
        "--merge-dir", str(RESULTS_DIR),
        "--generate-report",
    ]
    subprocess.run(merge_cmd, cwd=str(REPO_ROOT), check=True)
    total_elapsed = time.perf_counter() - total_start
    print(f"\n✓ Completed benchmark in {total_elapsed/60.0:.2f} minutes.")

if __name__ == "__main__":
    main()

