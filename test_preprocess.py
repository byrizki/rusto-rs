#!/usr/bin/env python3
"""Test preprocessing to compare with Rust"""
import cv2
import numpy as np
import sys

def preprocess_recognition(img, img_h=48, img_w=320):
    """Preprocess image for recognition model (PaddleOCR style)"""
    h, w = img.shape[:2]
    
    # Calculate aspect ratio
    import math
    ratio = w / float(h)
    if math.ceil(img_h * ratio) > img_w:
        resized_w = img_w
    else:
        resized_w = int(math.ceil(img_h * ratio))
    
    print(f"Original size: {w}x{h}")
    print(f"Resized to: {resized_w}x{img_h}")
    
    # Resize image
    resized_image = cv2.resize(img, (resized_w, img_h))
    
    # Check pixel values before normalization
    print(f"Image shape after resize: {resized_image.shape}")
    print(f"Image dtype: {resized_image.dtype}")
    print(f"First pixel (BGR): {resized_image[0, 0]}")
    
    # Normalize: (pixel/255 - 0.5) / 0.5
    # Model expects BGR format (cv2.imread already gives BGR)
    resized_image = resized_image.astype(np.float32)
    resized_image = resized_image.transpose(2, 0, 1) / 255.0
    
    print(f"After transpose and /255: shape={resized_image.shape}")
    print(f"First pixel values (CHW): B={resized_image[0, 0, 0]:.6f}, G={resized_image[1, 0, 0]:.6f}, R={resized_image[2, 0, 0]:.6f}")
    
    resized_image = (resized_image - 0.5) / 0.5
    
    print(f"After normalization: B={resized_image[0, 0, 0]:.6f}, G={resized_image[1, 0, 0]:.6f}, R={resized_image[2, 0, 0]:.6f}")
    
    # Pad to img_w
    padding_im = np.zeros((3, img_h, img_w), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    padding_im = np.expand_dims(padding_im, axis=0)
    
    print(f"Final shape: {padding_im.shape}")
    return padding_im

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "debug_output/example1_crops/crop_00.png"
    img = cv2.imread(img_path)
    print(f"Testing: {img_path}\n")
    result = preprocess_recognition(img)
    print(f"\nMin value: {result.min():.6f}")
    print(f"Max value: {result.max():.6f}")
