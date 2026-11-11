#!/usr/bin/env python3
"""
Converts all ONNX models in models/ to MNN and RTen formats, and optionally
generates PP-OCRv6 tiny FP16 models (.onnx and .mnn).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MNN_CONVERT = Path("crates/rusto-nn/tools/bin/MNNConvert").resolve()
RTEN_CONVERT = Path("benchmarks/omnidocbench/.venv/bin/rten-convert").resolve()
MODELS_DIR = Path("models").resolve()
FP16_MAX = 65504.0


def convert_onnx_to_fp16(in_path: Path, out_path: Path) -> bool:
    """Convert an FP32 ONNX model to FP16 while preserving FP32 graph I/O and Resize scales."""
    if out_path.is_file() and out_path.stat().st_size > 0:
        return True
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        print(f"    ⚠️ Missing onnx/numpy for FP16 conversion: {exc}", file=sys.stderr)
        return False

    print(f"  [ONNX-FP16] Converting {in_path.relative_to(MODELS_DIR)} -> {out_path.relative_to(MODELS_DIR)}...")
    model = onnx.load(str(in_path))
    graph = model.graph

    # Resize roi (idx 1) and scales (idx 2) must stay FLOAT for onnx.checker & ORT compatibility.
    resize_float_inputs: set[str] = set()
    other_consumers: set[str] = set()
    for node in graph.node:
        for idx, name in enumerate(node.input):
            if not name:
                continue
            if node.op_type == "Resize" and idx in (1, 2):
                resize_float_inputs.add(name)
            else:
                other_consumers.add(name)
    keep_float = resize_float_inputs - other_consumers

    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT and init.name not in keep_float:
            arr = numpy_helper.to_array(init)
            arr_f16 = np.clip(arr, -FP16_MAX, FP16_MAX).astype(np.float16)
            init.CopyFrom(numpy_helper.from_array(arr_f16, name=init.name))

    for node in graph.node:
        if node.op_type in ("Constant", "ConstantOfShape"):
            out_name = node.output[0] if node.output else ""
            if out_name in keep_float:
                continue
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t)
                    arr_f16 = np.clip(arr, -FP16_MAX, FP16_MAX).astype(np.float16)
                    attr.t.CopyFrom(numpy_helper.from_array(arr_f16))

    # Clear stale FP32 value_info so onnx.checker validates the new FP16 intermediate types cleanly.
    del graph.value_info[:]

    input_casts = []
    for inp in graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.FLOAT:
            orig = inp.name
            cast_out = f"{orig}_fp16"
            for node in graph.node:
                for i, name in enumerate(node.input):
                    if name == orig:
                        node.input[i] = cast_out
            input_casts.append(
                helper.make_node("Cast", [orig], [cast_out], name=f"{orig}_cast_fp16", to=TensorProto.FLOAT16)
            )

    output_casts = []
    for out in graph.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT:
            orig = out.name
            internal = f"{orig}_fp16"
            for node in graph.node:
                for i, name in enumerate(node.output):
                    if name == orig:
                        node.output[i] = internal
                for i, name in enumerate(node.input):
                    if name == orig:
                        node.input[i] = internal
            output_casts.append(
                helper.make_node("Cast", [internal], [orig], name=f"{orig}_cast_fp32", to=TensorProto.FLOAT)
            )

    if input_casts or output_casts:
        nodes = list(graph.node)
        del graph.node[:]
        graph.node.extend(input_casts + nodes + output_casts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    onnx.checker.check_model(str(out_path))
    print(f"    ✓ ONNX FP16 converted ({out_path.stat().st_size // 1024} KB)")
    return True


def convert_to_mnn(
    onnx_file: Path,
    mnn_file: Path,
    *,
    fp16: bool = False,
    weight_quant_bits: int | None = None,
    hqq: bool = False,
) -> bool:
    if mnn_file.is_file() and mnn_file.stat().st_size > 0:
        return True
    print(f"  [MNN] Converting {onnx_file.relative_to(MODELS_DIR)} -> {mnn_file.relative_to(MODELS_DIR)}...")
    mnn_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(MNN_CONVERT),
        "-f", "ONNX",
        "--modelFile", str(onnx_file),
        "--MNNModel", str(mnn_file),
        "--bizCode", "MNN",
        "--useGeluApproximation", "0",
    ]
    if fp16:
        cmd.append("--fp16")
    if weight_quant_bits is not None:
        cmd.extend(["--weightQuantBits", str(weight_quant_bits)])
        if hqq:
            cmd.append("--hqq")
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


def generate_tiny_fp16() -> bool:
    """Generate PPOCR_v6_tiny_fp16/{det,rec}.{onnx,mnn} + dict.txt from PPOCR_v6_tiny."""
    src_dir = MODELS_DIR / "PPOCR_v6_tiny"
    dst_dir = MODELS_DIR / "PPOCR_v6_tiny_fp16"
    if not src_dir.is_dir():
        print(f"Error: source directory {src_dir} not found", file=sys.stderr)
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in ("det", "rec"):
        src_onnx = src_dir / f"{name}.onnx"
        if not src_onnx.is_file():
            print(f"Error: missing {src_onnx}", file=sys.stderr)
            ok = False
            continue
        dst_onnx = dst_dir / f"{name}.onnx"
        ok &= convert_onnx_to_fp16(src_onnx, dst_onnx)
        ok &= convert_to_mnn(dst_onnx, dst_dir / f"{name}.mnn", fp16=True)
    dict_src = src_dir / "dict.txt"
    dict_dst = dst_dir / "dict.txt"
    if dict_src.is_file() and not dict_dst.is_file():
        shutil.copy2(dict_src, dict_dst)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert models in models/ to MNN/RTen and optional FP16.")
    parser.add_argument(
        "--generate-tiny-fp16",
        action="store_true",
        help="Generate models/PPOCR_v6_tiny_fp16/{det,rec}.{onnx,mnn} from models/PPOCR_v6_tiny",
    )
    args = parser.parse_args()

    if not MNN_CONVERT.is_file():
        print(f"Error: MNNConvert not found at {MNN_CONVERT}", file=sys.stderr)
        sys.exit(1)
    if not RTEN_CONVERT.is_file():
        print(f"Error: rten-convert not found at {RTEN_CONVERT}", file=sys.stderr)
        sys.exit(1)

    if args.generate_tiny_fp16:
        print("=== Generating PP-OCRv6 Tiny FP16 Models ===")
        if not generate_tiny_fp16():
            sys.exit(1)

    print("=== Scanning and Converting Models to MNN and RTen ===")
    onnx_files = sorted(MODELS_DIR.rglob("*.onnx"))

    success_mnn = 0
    success_rten = 0
    total_mnn = 0
    total_rten = 0

    for onnx_path in onnx_files:
        # Skip tract-specific export files
        if onnx_path.name.endswith("_tract.onnx"):
            continue
        parent_name = onnx_path.parent.name
        mnn = onnx_path.with_suffix(".mnn")
        rten = onnx_path.with_suffix(".rten")

        total_mnn += 1
        if parent_name.endswith("_fp16"):
            if convert_to_mnn(onnx_path, mnn, fp16=True):
                success_mnn += 1
            continue

        if parent_name.endswith("_int8") and onnx_path.stem == "det":
            src_fp32 = MODELS_DIR / parent_name.removesuffix("_int8") / onnx_path.name
            if src_fp32.is_file():
                if convert_to_mnn(src_fp32, mnn, weight_quant_bits=8, hqq=True):
                    success_mnn += 1
            elif mnn.is_file() and mnn.stat().st_size > 0:
                success_mnn += 1
        else:
            if convert_to_mnn(onnx_path, mnn):
                success_mnn += 1

        total_rten += 1
        if convert_to_rten(onnx_path, rten):
            success_rten += 1

    print(f"\nFinished: {total_mnn} models scanned.")
    print(f"  MNN format available: {success_mnn}/{total_mnn}")
    print(f"  RTen format available: {success_rten}/{total_rten}")

if __name__ == "__main__":
    main()

