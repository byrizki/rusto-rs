#!/usr/bin/env python3
"""
Quantize PaddleOCR PP-OCRv6 models (det and rec) to INT8 format.

Applies dynamic quantization:
- Detector (det.onnx): standard QInt8 dynamic quantization.
- Recognizer (rec.onnx): dynamic quantization with op_types=['Gather', 'Transpose', 'MatMul'],
  reduce_range=True, and per_channel=True to prevent CTC blank token dominance on CPU / SIMD.

Optionally converts quantized ONNX models to RTen FlatBuffers (.rten) format.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_rten_convert_cmd():
    """Find the rten-convert executable."""
    if shutil.which("rten-convert"):
        return ["rten-convert"]
    try:
        import rten_convert.converter  # noqa: F401
        return [
            sys.executable,
            "-c",
            "import sys; from rten_convert.converter import main; sys.argv = ['rten-convert'] + sys.argv[1:]; sys.exit(main())",
        ]
    except ImportError:
        pass
    if os.path.isfile("/tmp/rten-env/bin/rten-convert"):
        return ["/tmp/rten-env/bin/rten-convert"]
    user_local = os.path.expanduser("~/.local/bin/rten-convert")
    if os.path.isfile(user_local):
        return [user_local]
    return None


def quantize_det(in_path: str, out_path: str):
    """Quantize detection model using dynamic int8."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"  Quantizing detector: {in_path} -> {out_path}...")
    quantize_dynamic(
        model_input=in_path,
        model_output=out_path,
        weight_type=QuantType.QInt8,
    )
    in_size = os.path.getsize(in_path) / (1024 * 1024)
    out_size = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  ✓ Detector size: {in_size:.2f} MB -> {out_size:.2f} MB")


def quantize_rec(in_path: str, out_path: str):
    """Quantize recognition model using CTC-safe dynamic int8."""
    from onnxruntime.quantization import quantize_dynamic

    print(f"  Quantizing recognizer: {in_path} -> {out_path}...")
    quantize_dynamic(
        model_input=in_path,
        model_output=out_path,
        op_types_to_quantize=["Gather", "Transpose", "MatMul"],
        reduce_range=True,
        per_channel=True,
        extra_options={"EnableSubgraph": True},
    )
    in_size = os.path.getsize(in_path) / (1024 * 1024)
    out_size = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  ✓ Recognizer size: {in_size:.2f} MB -> {out_size:.2f} MB")


def convert_to_rten(in_path: str, out_path: str, rten_cmd):
    """Convert an ONNX model to .rten format."""
    print(f"  Converting to RTen: {in_path} -> {out_path}...")
    cmd = rten_cmd + [in_path, out_path]
    subprocess.check_call(cmd)
    out_size = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  ✓ RTen model: {out_size:.2f} MB")


def quantize_tier(tier: str, input_dir: Path, output_dir: Path, convert_rten: bool = True, clean_onnx: bool = False):
    """Quantize a single tier."""
    print(f"\n==========================================")
    print(f"  Quantizing PP-OCRv6 {tier.upper()} to INT8")
    print(f"==========================================")

    output_dir.mkdir(parents=True, exist_ok=True)

    det_in = input_dir / "det.onnx"
    rec_in = input_dir / "rec.onnx"
    dict_in = input_dir / "dict.txt"

    if not det_in.exists() or not rec_in.exists():
        repo_root = Path(__file__).resolve().parent.parent
        dl_script = repo_root / "scripts" / "download_models.sh"
        if dl_script.exists():
            print(f"  det.onnx or rec.onnx missing in {input_dir}, downloading via download_models.sh...")
            subprocess.run(["bash", str(dl_script), "--model", "ppocrv6", "--tier", tier, "--output-dir", str(input_dir)], check=False)

    if not det_in.exists() or not rec_in.exists():
        print(f"Error: Missing det.onnx or rec.onnx in {input_dir}", file=sys.stderr)
        return False

    det_onnx_out = output_dir / "det.onnx"
    rec_onnx_out = output_dir / "rec.onnx"

    quantize_det(str(det_in), str(det_onnx_out))
    quantize_rec(str(rec_in), str(rec_onnx_out))

    # Copy dictionary
    if dict_in.exists():
        shutil.copy2(str(dict_in), str(output_dir / "dict.txt"))
        print(f"  ✓ Copied dictionary: dict.txt")

    if convert_rten:
        rten_cmd = get_rten_convert_cmd()
        if rten_cmd is None:
            print("  Warning: rten-convert not found. Skipping RTen conversion.", file=sys.stderr)
        else:
            det_rten_out = output_dir / "det.rten"
            rec_rten_out = output_dir / "rec.rten"
            convert_to_rten(str(det_onnx_out), str(det_rten_out), rten_cmd)
            convert_to_rten(str(rec_onnx_out), str(rec_rten_out), rten_cmd)

            if clean_onnx:
                det_onnx_out.unlink(missing_ok=True)
                rec_onnx_out.unlink(missing_ok=True)
                print("  ✓ Removed intermediate quantized ONNX files")

    return True


def main():
    parser = argparse.ArgumentParser(description="Quantize PP-OCRv6 models to INT8")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Quantize all PP-OCRv6 tiers (tiny, small, medium)",
    )
    parser.add_argument(
        "--tier",
        choices=["tiny", "small", "medium", "all"],
        default="all",
        help="Tier to quantize (default: all)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path to input models directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to output models directory",
    )
    parser.add_argument(
        "--no-rten",
        action="store_true",
        help="Skip RTen conversion",
    )
    parser.add_argument(
        "--clean-onnx",
        action="store_true",
        help="Remove intermediate quantized ONNX files after RTen conversion",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    effective_tier = "all" if args.all else args.tier
    tiers = ["tiny", "small", "medium"] if effective_tier == "all" else [effective_tier]

    success = True
    for t in tiers:
        in_dir = (
            Path(args.input_dir)
            if args.input_dir and args.tier != "all"
            else repo_root / "models" / f"PPOCR_v6_{t}"
        )
        out_dir = (
            Path(args.output_dir)
            if args.output_dir and args.tier != "all"
            else repo_root / "models" / f"PPOCR_v6_{t}_int8"
        )
        ok = quantize_tier(
            tier=t,
            input_dir=in_dir,
            output_dir=out_dir,
            convert_rten=not args.no_rten,
            clean_onnx=args.clean_onnx,
        )
        if not ok:
            success = False

    if success:
        print("\n✅ All PP-OCRv6 INT8 quantization completed successfully!")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
