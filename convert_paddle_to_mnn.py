#!/usr/bin/env python3
"""
Convert OCR models to MNN format.

Supports:
- Paddle models -> ONNX -> MNN
- ONNX models -> MNN
"""

import os
import sys
import subprocess
import argparse
import yaml
from pathlib import Path
from enum import Enum


class InputFormat(Enum):
    PADDLE = "paddle"
    ONNX = "onnx"
    AUTO = "auto"


def check_dependencies():
    """Check required dependencies"""
    # Check paddle2onnx
    try:
        subprocess.run(['paddle2onnx', '--version'], 
                      capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Error: paddle2onnx not found. Install: pip install paddle2onnx")
        return False
    
    # Check mnnconvert
    try:
        subprocess.run(['mnnconvert', '--version'], 
                      capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Error: mnnconvert not found. Install from: https://github.com/alibaba/MNN")
        return False
    
    return True


def convert_paddle_to_onnx(model_dir):
    """Convert Paddle model to ONNX format"""
    model_path = Path(model_dir)
    inference_json = model_path / "inference.json"
    inference_pdiparams = model_path / "inference.pdiparams"
    output_onnx = model_path / "model.onnx"
    
    if not inference_json.exists() or not inference_pdiparams.exists():
        return False
    
    if output_onnx.exists():
        return True
    
    cmd = [
        'paddle2onnx',
        '--model_dir', '.',
        '--model_filename', 'inference.json',
        '--params_filename', 'inference.pdiparams',
        '--save_file', 'model.onnx'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(model_path))
        if result.returncode != 0:
            print(f"  [ONNX] Failed: {result.stderr.strip().split(chr(10))[-1]}")
            return False
        print(f"  [ONNX] ✓")
        return True
    except Exception as e:
        print(f"  [ONNX] Error: {e}")
        return False


def convert_onnx_to_mnn(model_dir, use_fp16=True, onnx_filename="model.onnx", mnn_filename="model.mnn"):
    """Convert ONNX model to MNN format"""
    model_path = Path(model_dir)
    input_onnx = model_path / onnx_filename
    output_mnn = model_path / mnn_filename
    
    if not input_onnx.exists():
        return False
    
    if output_mnn.exists():
        return True
    
    cmd = [
        'mnnconvert',
        '-f', 'ONNX',
        '--modelFile', str(input_onnx.name),
        '--MNNModel', str(output_mnn.name),
        '--bizCode', 'mnn'
    ]
    
    if use_fp16:
        cmd.append('--fp16')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(model_path))
        if result.returncode != 0:
            print(f"  [MNN] Failed: {result.stderr.strip().split(chr(10))[-1]}")
            return False
        print(f"  [MNN] ✓ (fp16={use_fp16})")
        return True
    except Exception as e:
        print(f"  [MNN] Error: {e}")
        return False


def extract_character_dict(model_dir):
    """Extract character dictionary from inference.yml"""
    model_path = Path(model_dir)
    inference_yml = model_path / "inference.yml"
    output_txt = model_path / "ppocr_keys.txt"
    
    if not inference_yml.exists():
        return False
    
    if output_txt.exists():
        return True
    
    try:
        with open(inference_yml, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        character_dict = None
        if 'PostProcess' in data:
            if isinstance(data['PostProcess'], dict):
                character_dict = data['PostProcess'].get('character_dict')
            elif isinstance(data['PostProcess'], list):
                for item in data['PostProcess']:
                    if isinstance(item, dict) and 'character_dict' in item:
                        character_dict = item['character_dict']
                        break
        
        if not character_dict:
            return False
        
        with open(output_txt, 'w', encoding='utf-8') as f:
            for char in character_dict:
                char_str = str(char) if char is not None else ''
                f.write(char_str + '\n')
        
        print(f"  [Dict] ✓ ({len(character_dict)} chars)")
        return True
    
    except Exception as e:
        print(f"  [Dict] Error: {e}")
        return False


def detect_input_format(model_dir):
    """Auto-detect input model format"""
    model_path = Path(model_dir)
    
    # Check for Paddle model files
    has_paddle = (model_path / "inference.json").exists() and \
                 (model_path / "inference.pdiparams").exists()
    
    # Check for ONNX model files
    has_onnx = any(model_path.glob("*.onnx"))
    
    if has_paddle:
        return InputFormat.PADDLE
    elif has_onnx:
        return InputFormat.ONNX
    else:
        return None


def convert_model(model_dir, use_fp16=True, input_format=InputFormat.AUTO):
    """Convert single model directory"""
    model_path = Path(model_dir)
    model_name = model_path.name
    
    print(f"\n{model_name}:")
    
    # Auto-detect format if needed
    if input_format == InputFormat.AUTO:
        detected_format = detect_input_format(model_path)
        if detected_format is None:
            print("  [ERROR] Could not detect input format (no Paddle or ONNX files found)")
            return {'success': False}
        input_format = detected_format
        print(f"  [Detected] {input_format.value.upper()} format")
    
    results = {
        'paddle_to_onnx': False,
        'onnx_to_mnn': False,
        'extract_dict': False,
        'success': False
    }
    
    # Convert based on input format
    if input_format == InputFormat.PADDLE:
        # Paddle -> ONNX -> MNN pipeline
        results['paddle_to_onnx'] = convert_paddle_to_onnx(model_path)
        
        if results['paddle_to_onnx']:
            results['onnx_to_mnn'] = convert_onnx_to_mnn(model_path, use_fp16)
        
        results['extract_dict'] = extract_character_dict(model_path)
    
    elif input_format == InputFormat.ONNX:
        # Direct ONNX -> MNN conversion
        onnx_files = list(model_path.glob("*.onnx"))
        if onnx_files:
            onnx_file = onnx_files[0]
            mnn_file = onnx_file.stem + ".mnn"
            results['onnx_to_mnn'] = convert_onnx_to_mnn(
                model_path, 
                use_fp16, 
                onnx_filename=onnx_file.name,
                mnn_filename=mnn_file
            )
            print(f"  [Input] {onnx_file.name}")
            print(f"  [Output] {mnn_file}")
    
    results['success'] = any([results['paddle_to_onnx'], results['onnx_to_mnn'], results['extract_dict']])
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Convert OCR models to MNN format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect and convert all models in ocr directory (with FP16)
  python convert_paddle_to_mnn.py
  
  # Specify OCR directory
  python convert_paddle_to_mnn.py --ocr-dir ./my_ocr_models
  
  # Convert only Paddle models
  python convert_paddle_to_mnn.py --format paddle
  
  # Convert only ONNX models
  python convert_paddle_to_mnn.py --format onnx
  
  # Disable FP16
  python convert_paddle_to_mnn.py --no-fp16
  
  # Convert single model directory
  python convert_paddle_to_mnn.py --model ./path/to/model
        """
    )
    
    parser.add_argument(
        '--ocr-dir',
        type=str,
        default='./ocr',
        help='OCR models root directory (default: ./ocr)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Single model directory to convert (overrides --ocr-dir)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['auto', 'paddle', 'onnx'],
        default='auto',
        help='Input model format (default: auto-detect)'
    )
    
    parser.add_argument(
        '--no-fp16',
        action='store_true',
        help='Disable FP16 precision (default: enabled)'
    )
    
    args = parser.parse_args()
    
    use_fp16 = not args.no_fp16
    input_format = InputFormat(args.format)
    
    # Determine model directories to process
    if args.model:
        model_dirs = [Path(args.model)]
        if not model_dirs[0].exists():
            print(f"Error: Model directory not found: {model_dirs[0]}")
            sys.exit(1)
    else:
        ocr_dir = Path(args.ocr_dir)
        if not ocr_dir.exists():
            print(f"Error: OCR directory not found: {ocr_dir}")
            sys.exit(1)
        model_dirs = [d for d in ocr_dir.iterdir() if d.is_dir()]
    
    print(f"OCR Model to MNN Converter")
    print(f"Input format: {input_format.value.upper()}")
    print(f"FP16: {use_fp16}")
    if args.model:
        print(f"Model: {Path(args.model).absolute()}")
    else:
        print(f"OCR dir: {Path(args.ocr_dir).absolute()}")
    print()
    
    if not check_dependencies():
        sys.exit(1)
    
    if not model_dirs:
        print(f"Warning: No model directories found in {ocr_dir}")
        sys.exit(0)
    
    print(f"Found {len(model_dirs)} models")
    
    total = len(model_dirs)
    success_count = 0
    failed_models = []
    
    for model_dir in sorted(model_dirs):
        try:
            results = convert_model(model_dir, use_fp16, input_format)
            
            if results.get('success', False):
                success_count += 1
            else:
                failed_models.append(model_dir.name)
        
        except Exception as e:
            print(f"  Error: {e}")
            failed_models.append(model_dir.name)
    
    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{total} successful")
    
    if failed_models:
        print(f"Failed: {len(failed_models)}")
        for model_name in failed_models:
            print(f"  - {model_name}")
    
    print()


if __name__ == '__main__':
    main()
