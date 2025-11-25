#!/usr/bin/env python3
"""
Create test images for orientation and rectification testing.
Uses example1.png from models/test_images as base image.
Outputs to models/test_images directory.
"""

from PIL import Image, ImageFilter
import numpy as np
import sys
from pathlib import Path

# Configuration
BASE_IMAGE = "models/test_images/example1.png"
OUTPUT_DIR = "models/test_images"

def create_rotated_image(input_path, output_path, angle):
    """Create a high-quality rotated version of the image."""
    img = Image.open(input_path)
    # Use BICUBIC for better quality
    rotated = img.rotate(angle, expand=True, fillcolor='white', resample=Image.BICUBIC)
    rotated.save(output_path, quality=95)
    print(f"✓ Created {output_path.name} (rotated {angle}°)")
    return rotated

def create_curved_image(input_path, output_path):
    """Create a curved/distorted version for rectification testing."""
    img = Image.open(input_path)
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    # Create a mesh grid
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    
    # Apply wave distortion
    amplitude = 20  # pixels
    frequency = 2 * np.pi / width
    
    # Horizontal wave
    xx_distorted = xx + amplitude * np.sin(frequency * yy)
    
    # Clip to valid range
    xx_distorted = np.clip(xx_distorted, 0, width - 1).astype(int)
    
    # Create distorted image
    distorted = np.zeros_like(img_array)
    for i in range(height):
        for j in range(width):
            distorted[i, j] = img_array[i, xx_distorted[i, j]]
    
    result = Image.fromarray(distorted)
    result.save(output_path, quality=95)
    print(f"✓ Created {output_path.name} (curved distortion)")
    return result

def create_perspective_distorted(input_path, output_path):
    """Create perspective-distorted image for rectification testing."""
    img = Image.open(input_path)
    width, height = img.size
    
    # Define perspective transform coefficients
    # This creates a trapezoidal effect
    coeffs = find_coeffs(
        [(0, 0), (width, 0), (width, height), (0, height)],  # original corners
        [(50, 30), (width-30, 50), (width-50, height-30), (30, height-50)]  # distorted corners
    )
    
    distorted = img.transform(
        (width, height),
        Image.PERSPECTIVE,
        coeffs,
        Image.BICUBIC
    )
    distorted.save(output_path, quality=95)
    print(f"✓ Created {output_path.name} (perspective distortion)")
    return distorted

def find_coeffs(source_coords, target_coords):
    """Find perspective transform coefficients."""
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
    
    A = np.array(matrix, dtype=float)
    B = np.array(source_coords).reshape(8)
    
    try:
        res = np.linalg.solve(A, B)
        return np.array(res).tolist()
    except:
        # Fallback to identity-like transform
        return [1, 0, 0, 0, 1, 0, 0, 0]

def create_noisy_rotated(input_path, output_path, angle):
    """Create rotated image with minimal noise to preserve readability."""
    img = Image.open(input_path)
    
    # Rotate with high quality settings
    rotated = img.rotate(angle, expand=True, fillcolor='white', resample=Image.BICUBIC)
    
    # Add very slight noise (reduced from 10 to 3 for better quality)
    img_array = np.array(rotated)
    noise = np.random.normal(0, 3, img_array.shape).astype(np.int16)
    noisy = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    result = Image.fromarray(noisy)
    result.save(output_path, quality=95)
    print(f"✓ Created {output_path.name} (rotated {angle}° with minimal noise)")
    return result

def main():
    # Check if input file exists
    input_file = Path(BASE_IMAGE)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("Please ensure models/test_images/example1.png exists")
        sys.exit(1)
    
    # Check if output directory exists
    output_dir = Path(OUTPUT_DIR)
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        print("Please ensure models/test_images/ directory exists")
        sys.exit(1)
    
    print("=== Creating Test Images ===\n")
    print(f"Base image: {input_file}")
    print(f"Output directory: {output_dir}\n")
    
    # 1. Orientation test images (0°, 90°, 180°, 270°)
    print("1. Orientation Classification Tests:")
    create_rotated_image(input_file, output_dir / "example1_rotate_90.png", 90)
    create_rotated_image(input_file, output_dir / "example1_rotate_180.png", 180)
    create_rotated_image(input_file, output_dir / "example1_rotate_270.png", 270)
    
    # 2. Slight rotation (5-15 degrees) for angle correction
    # Using minimal noise to preserve readability
    print("\n2. Small Rotation Tests (with minimal noise):")
    create_noisy_rotated(input_file, output_dir / "example1_rotate_5.png", 5)
    create_noisy_rotated(input_file, output_dir / "example1_rotate_15.png", 15)
    create_noisy_rotated(input_file, output_dir / "example1_rotate_minus10.png", -10)
    
    # 3. Rectification test images (no barrel - removed as requested)
    print("\n3. Rectification Tests:")
    create_curved_image(input_file, output_dir / "example1_curved.png")
    create_perspective_distorted(input_file, output_dir / "example1_perspective.png")
    
    # 4. Combined tests
    print("\n4. Combined Tests:")
    # Create curved then rotate it
    temp_curved_path = output_dir / "temp_curved.png"
    temp_curved = create_curved_image(input_file, temp_curved_path)
    temp_curved_rotated = temp_curved.rotate(90, expand=True, fillcolor='white', resample=Image.BICUBIC)
    temp_curved_rotated.save(output_dir / "example1_curved_rotated.png", quality=95)
    print(f"✓ Created example1_curved_rotated.png (curved + rotated)")
    
    # Clean up temp file
    if temp_curved_path.exists():
        temp_curved_path.unlink()
    
    print(f"\n=== Done! ===")
    # Count new test images (exclude original example1.png)
    test_images = [f for f in output_dir.glob('example1_*.png')]
    print(f"Created {len(test_images)} test images in {output_dir}/")
    print("\nTest images:")
    for img in sorted(test_images):
        print(f"  - {img.name}")
    
    print("\nUsage:")
    print(f"  cargo run --release --example ocr_with_fixes")
    print(f"  cargo run --release --example ocr_without_fixes")

if __name__ == "__main__":
    main()
