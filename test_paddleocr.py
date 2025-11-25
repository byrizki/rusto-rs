#!/usr/bin/env python3
"""Test with PaddleOCR official implementation"""
from paddleocr import PaddleOCR
import sys

# Initialize OCR
print("Initializing PaddleOCR...")
ocr = PaddleOCR(use_textline_orientation=False, lang='en')

# Run OCR
image_path = sys.argv[1] if len(sys.argv) > 1 else 'models/test_images/example1.png'
print(f"\nProcessing: {image_path}")
result = ocr.predict(image_path)

# Print results
print(f"\nResult type: {type(result)}")
print(f"Result structure: {result}")
if isinstance(result, dict) and 'ocr_text' in result:
    print(f"\nOCR Text: {result['ocr_text']}")
elif isinstance(result, list) and len(result) > 0:
    print(f"\nFirst item: {result[0]}")
