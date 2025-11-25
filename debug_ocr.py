#!/usr/bin/env python3
"""
Debug OCR using ONNX Runtime and MNN Python bindings directly
"""
import json
import sys
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import cv2

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: onnxruntime not installed. Install with: pip install onnxruntime")

try:
    import MNN
    HAS_MNN = True
except ImportError:
    HAS_MNN = False
    print("Warning: MNN not installed. Install with: pip install MNN")


def load_dict(dict_path):
    """Load character dictionary"""
    with open(dict_path, 'r', encoding='utf-8') as f:
        chars = [line.strip() for line in f]
    return chars


def preprocess_detection(img, limit_side_len=960):
    """Preprocess image for detection model (PaddleOCR style)"""
    h, w = img.shape[:2]
    
    # Calculate resize ratio
    if min(h, w) < limit_side_len:
        if h < w:
            ratio = float(limit_side_len) / h
        else:
            ratio = float(limit_side_len) / w
    else:
        ratio = 1.0
    
    resize_h = int(h * ratio)
    resize_w = int(w * ratio)
    
    # Round to multiple of 32
    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)
    
    # Resize image
    img = cv2.resize(img, (resize_w, resize_h))
    
    # Normalize (BGR format, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = img.astype(np.float32) / 255.0
    img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229  # B
    img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224  # G
    img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225  # R
    
    # Convert to CHW format
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    
    return img, ratio, (resize_h, resize_w)


def preprocess_recognition(img, img_h=48, img_w=320):
    """Preprocess image for recognition model (PaddleOCR style)"""
    h, w = img.shape[:2]
    
    # Calculate aspect ratio
    ratio = w / float(h)
    if math.ceil(img_h * ratio) > img_w:
        resized_w = img_w
    else:
        resized_w = int(math.ceil(img_h * ratio))
    
    # Resize image
    resized_image = cv2.resize(img, (resized_w, img_h))
    
    # Normalize: (pixel/255 - 0.5) / 0.5
    # Model expects BGR format (cv2.imread already gives BGR)
    resized_image = resized_image.astype(np.float32)
    if len(resized_image.shape) == 2:  # Grayscale
        resized_image = resized_image / 255.0
        resized_image = np.expand_dims(resized_image, axis=0)
    else:  # Color (BGR) - transpose to CHW format
        resized_image = resized_image.transpose(2, 0, 1) / 255.0
    
    resized_image = (resized_image - 0.5) / 0.5
    
    # Pad to img_w
    padding_im = np.zeros((3, img_h, img_w), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    padding_im = np.expand_dims(padding_im, axis=0)
    
    return padding_im, resized_w / float(img_w)


def decode_ctc(preds, chars):
    """CTC decode"""
    # Get argmax along character dimension
    preds_idx = np.argmax(preds, axis=2)
    preds_prob = np.max(preds, axis=2)
    
    # Decode
    result = []
    for idx, prob in zip(preds_idx[0], preds_prob[0]):
        if idx > 0 and idx <= len(chars):  # Skip blank (0)
            result.append((chars[idx - 1], prob))
    
    # Remove consecutive duplicates
    decoded = []
    prev_char = None
    for char, prob in result:
        if char != prev_char:
            decoded.append((char, prob))
            prev_char = char
    
    text = ''.join([c for c, _ in decoded])
    confidence = np.mean([p for _, p in decoded]) if decoded else 0.0
    
    return text, confidence

def run_detection_onnx(img, model_path):
    """Run detection using ONNX Runtime"""
    # Preprocess
    input_img, ratio, (h, w) = preprocess_detection(img)
    
    # Run inference
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_img})
    
    # Post-process (simplified - just get bounding boxes)
    # This is a simplified version - full postprocessing is complex
    return [], []  # boxes, scores


def run_recognition_mnn(img, model_path, dict_chars):
    """Run recognition using MNN Python bindings"""
    # Preprocess
    input_img, valid_ratio = preprocess_recognition(img)
    
    # Create MNN interpreter
    interpreter = MNN.Interpreter(model_path)
    
    # Create session with config
    config = {}
    config['precision'] = 'low'  # Use low precision for faster inference
    session = interpreter.createSession(config)
    
    # Get input tensor and resize
    input_tensor = interpreter.getSessionInput(session)
    interpreter.resizeTensor(input_tensor, (1, 3, 48, 320))
    interpreter.resizeSession(session)
    
    # Create temporary tensor and copy data
    # Flatten the input data for MNN
    input_data = input_img.flatten().astype(np.float32)
    tmp_input = MNN.Tensor((1, 3, 48, 320), MNN.Halide_Type_Float, input_data, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)
    
    # Run inference
    interpreter.runSession(session)
    
    # Get output
    output_tensor = interpreter.getSessionOutput(session)
    output_shape = output_tensor.getShape()
    
    # Copy output to numpy
    # Create output buffer with correct shape
    output_size = 1
    for dim in output_shape:
        output_size *= dim
    tmp_output = MNN.Tensor(output_shape, MNN.Halide_Type_Float, 
                           np.zeros(output_size).astype(np.float32), 
                           MNN.Tensor_DimensionType_Caffe)
    output_tensor.copyToHostTensor(tmp_output)
    output_data = np.array(tmp_output.getData(), dtype=np.float32).reshape(output_shape)
    
    # Apply softmax
    output_data = np.exp(output_data) / np.sum(np.exp(output_data), axis=-1, keepdims=True)
    
    # Decode
    text, confidence = decode_ctc(output_data, dict_chars)
    
    return text, confidence


def run_recognition_onnx(img, model_path, dict_chars):
    """Run recognition using ONNX Runtime"""
    # Preprocess
    input_img, valid_ratio = preprocess_recognition(img)
    
    # Run inference
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_img})
    
    # Apply softmax
    output_data = outputs[0]
    output_data = np.exp(output_data) / np.sum(np.exp(output_data), axis=-1, keepdims=True)
    
    # Decode
    text, confidence = decode_ctc(output_data, dict_chars)
    
    return text, confidence


def visualize_detection(image_path, boxes, txts, scores, output_path):
    """Draw bounding boxes and text on image"""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    for i, (box, txt, score) in enumerate(zip(boxes, txts, scores)):
        # Convert box format
        pts = [(pt[0], pt[1]) for pt in box]
        
        # Draw box
        draw.polygon(pts, outline='red', width=2)
        
        # Draw text label
        label = f"{i}: {score:.2f}"
        draw.text(pts[0], label, fill='blue')
    
    img.save(output_path)
    print(f"Saved detection visualization: {output_path}")

def crop_text_regions(image_path, boxes, output_dir):
    """Crop and save individual text regions"""
    img = Image.open(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for i, box in enumerate(boxes):
        # Get bounding rectangle
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        
        # Crop region
        cropped = img.crop((x1, y1, x2, y2))
        crop_path = output_dir / f"crop_{i:02d}.png"
        cropped.save(crop_path)
    
    print(f"Saved {len(boxes)} cropped regions to {output_dir}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_ocr.py <crop_image_or_dir> [--mnn|--onnx]")
        print("Examples:")
        print("  python debug_ocr.py debug_output/example1_crops/crop_00.png --mnn")
        print("  python debug_ocr.py debug_output/example1_crops/ --onnx")
        sys.exit(1)
    
    target_path = Path(sys.argv[1])
    use_mnn = '--mnn' in sys.argv
    use_onnx = '--onnx' in sys.argv
    
    # Default to MNN if both installed, otherwise use what's available
    if not use_mnn and not use_onnx:
        if HAS_MNN:
            use_mnn = True
            print("Using MNN (default)")
        elif HAS_ONNX:
            use_onnx = True
            print("Using ONNX Runtime")
        else:
            print("Error: Neither MNN nor ONNX Runtime is installed")
            print("Install with: pip install MNN onnxruntime")
            sys.exit(1)
    
    # Load dictionary
    dict_path = "models/PPOCR_v5/dict.txt"
    if not Path(dict_path).exists():
        dict_path = "models/raw/en_PP-OCRv4_mobile_rec_infer/ppocr_keys.txt"
    print(f"Loading dictionary: {dict_path}")
    dict_chars = load_dict(dict_path)
    print(f"Loaded {len(dict_chars)} characters\\n")
    
    # Determine model path
    if use_mnn:
        model_path = "models/PPOCR_v5/rec.mnn"
        if not Path(model_path).exists():
            model_path = "models/raw/en_PP-OCRv4_mobile_rec_infer/model.mnn"
        print(f"Using MNN model: {model_path}")
    else:
        model_path = "models/raw/en_PP-OCRv4_mobile_rec_infer/model.onnx"
        print(f"Using ONNX model: {model_path}")
    
    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    # Process image(s)
    if target_path.is_file():
        # Single image
        print(f"\\nProcessing: {target_path}")
        img = cv2.imread(str(target_path))
        if img is None:
            print(f"Error: Could not read image")
            sys.exit(1)
        
        if use_mnn:
            text, confidence = run_recognition_mnn(img, model_path, dict_chars)
        else:
            text, confidence = run_recognition_onnx(img, model_path, dict_chars)
        
        print(f"Result: [{confidence:.2%}] {text}")
        
    elif target_path.is_dir():
        # Directory of images
        image_files = sorted(target_path.glob("*.png"))
        print(f"\\nProcessing {len(image_files)} images from {target_path}\\n")
        
        for img_file in image_files[:10]:  # Limit to first 10
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"{img_file.name}: Error reading image")
                continue
            
            if use_mnn:
                text, confidence = run_recognition_mnn(img, model_path, dict_chars)
            else:
                text, confidence = run_recognition_onnx(img, model_path, dict_chars)
            
            print(f"{img_file.name}: [{confidence:.2%}] {text}")
    else:
        print(f"Error: Path not found: {target_path}")
        sys.exit(1)

if __name__ == '__main__':
    main()
