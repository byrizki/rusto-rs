#!/usr/bin/env python3
"""
Converts all ONNX models in models/ to MNN and RTen formats, ensuring
100% format coverage (.onnx, .mnn, .rten) across all tiers.
"""

import os
import subprocess
import sys
from pathlib import Path

MNN_CONVERT = Path("crates/rusto-nn/tools/bin/MNNConvert").resolve()
RTEN_CONVERT = Path("benchmarks/omnidocbench/.venv/bin/rten-convert").resolve()
MODELS_DIR = Path("models").resolve()

def convert_to_mnn(onnx_file: Path, mnn_file: Path) -> bool:
    if mnn_file.is_file() and mnn_file.stat().st_size > 0:
        return True
    print(f"  [MNN] Converting {onnx_file.relative_to(MODELS_DIR)} -> {mnn_file.name}...")
    cmd = [
        str(MNN_CONVERT),
        "-f", "ONNX",
        "--modelFile", str(onnx_file),
        "--MNNModel", str(mnn_file),
        "--bizCode", "MNN",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode == 0 and mnn_file.is_file() and mnn_file.stat().st_size > 0:
        print(f"    ✓ MNN converted ({mnn_file.stat().st_size // 1024} KB)")
        return True
    else:
        print(f"    ⚠️ MNN convert failed: {res.stderr.decode('utf-8', errors='ignore')[:200]}")
        return False

def convert_to_rten(onnx_file: Path, rten_file: Path) -> bool:
    if rten_file.is_file() and rten_file.stat().st_size > 0:
        return True
    print(f"  [RTen] Converting {onnx_file.relative_to(MODELS_DIR)} -> {rten_file.name}...")
    cmd = [
        str(RTEN_CONVERT),
        str(onnx_file),
        str(rten_file),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode == 0 and rten_file.is_file() and rten_file.stat().st_size > 0:
        print(f"    ✓ RTen converted ({rten_file.stat().st_size // 1024} KB)")
        return True
    else:
        print(f"    ⚠️ RTen convert failed: {res.stderr.decode('utf-8', errors='ignore')[:200]}")
        return False

def main() -> None:
    if not MNN_CONVERT.is_file():
        print(f"Error: MNNConvert not found at {MNN_CONVERT}", file=sys.stderr)
        sys.exit(1)
    if not RTEN_CONVERT.is_file():
        print(f"Error: rten-convert not found at {RTEN_CONVERT}", file=sys.stderr)
        sys.exit(1)

    print("=== Scanning and Converting Models to MNN and RTen ===")
    onnx_files = sorted(MODELS_DIR.rglob("*.onnx"))
    
    success_mnn = 0
    success_rten = 0
    total = 0

    for onnx in onnx_files:
        # Skip tract-specific export files
        if onnx.name.endswith("_tract.onnx"):
            continue
        total += 1
        mnn = onnx.with_suffix(".mnn")
        rten = onnx.with_suffix(".rten")
        
        if convert_to_mnn(onnx, mnn):
            success_mnn += 1
        if convert_to_rten(onnx, rten):
            success_rten += 1

    print(f"\nFinished: {total} models processed.")
    print(f"  MNN format available: {success_mnn}/{total}")
    print(f"  RTen format available: {success_rten}/{total}")

if __name__ == "__main__":
    main()

