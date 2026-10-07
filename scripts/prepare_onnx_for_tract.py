#!/usr/bin/env python3
"""
Prepares and conditions PaddleOCR / RapidOCR ONNX models for execution with Sonos Tract.

Paddle2ONNX exports models with symbolic `value_info` annotations (e.g. DynamicDimension.0)
which cause Tract's shape unification to fail with symbol/value collisions.
This script cleans value_info and sets concrete input shapes for direct Tract optimization.
"""

import sys
import os
import argparse
import onnx

def clean_onnx_for_tract(input_path: str, output_path: str, batch: int = 1, height: int = None, width: int = None):
    print(f"Loading ONNX model: {input_path}")
    model = onnx.load(input_path)
    
    # 1. Clear all intermediate value_info annotations that contain hardcoded symbols
    while len(model.graph.value_info) > 0:
        model.graph.value_info.pop()
        
    # 2. Configure input dimensions
    if len(model.graph.input) > 0:
        inp = model.graph.input[0]
        dims = inp.type.tensor_type.shape.dim
        if len(dims) == 4:
            # Batch
            dims[0].ClearField("dim_param")
            dims[0].dim_value = batch
            
            # Channels (usually 3)
            if not dims[1].dim_value:
                dims[1].dim_value = 3
                
            # Height
            if height is not None:
                dims[2].ClearField("dim_param")
                dims[2].dim_value = height
            elif dims[2].dim_param:
                # Default detection: 640, default recognition: 48
                dims[2].ClearField("dim_param")
                dims[2].dim_value = 48 if "rec" in os.path.basename(input_path).lower() else 640

            # Width
            if width is not None:
                dims[3].ClearField("dim_param")
                dims[3].dim_value = width
            elif dims[3].dim_param:
                dims[3].ClearField("dim_param")
                dims[3].dim_value = 320 if "rec" in os.path.basename(input_path).lower() else 640

    # 3. Clear symbolic strings in graph outputs
    for out in model.graph.output:
        for d in out.type.tensor_type.shape.dim:
            if d.dim_param:
                d.ClearField("dim_param")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model, output_path)
    print(f"✓ Saved tract-compatible model: {output_path} ({os.path.getsize(output_path)} bytes)")

def main():
    parser = argparse.ArgumentParser(description="Clean ONNX model for Sonos Tract compatibility")
    parser.add_argument("input", help="Path to input .onnx file")
    parser.add_argument("output", help="Path to output .onnx file")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--height", type=int, default=None, help="Input height")
    parser.add_argument("--width", type=int, default=None, help="Input width")
    args = parser.parse_args()

    clean_onnx_for_tract(args.input, args.output, args.batch, args.height, args.width)

if __name__ == "__main__":
    main()
