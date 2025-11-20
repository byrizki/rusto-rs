# Pure Rust Migration Guide

## Overview

This codebase now supports TWO variants:
1. **Pure Rust** (default) - No OpenCV dependency
2. **OpenCV** (optional) - Behind `use-opencv` feature flag

## Current Status

✅ **DONE:**
- Added image abstraction layer (`src/image_impl.rs`)
- Updated `Cargo.toml` with feature flags
- OpenCV variant still works: `cargo build --features use-opencv`

🚧 **TODO:**
All modules need migration to use the abstraction layer instead of direct OpenCV calls.

## How to Build

### Pure Rust (default):
```bash
cargo build
```

### With OpenCV:
```bash
cargo build --features use-opencv
```

## Migration Strategy

Each module needs to:
1. Replace `use opencv::...` with `use crate::image_impl::*`
2. Replace `opencv::core::Point2f` with `image_impl::Point2f`
3. Replace `opencv::Result` with `image_impl::Result`
4. Update Mat operations to use abstraction methods

## Modules to Migrate

### 1. `src/det.rs`
**Current OpenCV usage:**
- `opencv::core::Mat`
- `opencv::core::Point2f`

**Migration:**
```rust
// Before:
use opencv::{core::Mat, prelude::MatTraitConst};
fn run(&self, img: &Mat) -> Result<TextDetOutput, EngineError>

// After:
use crate::image_impl::{Mat, Point2f, Result as ImgResult};
fn run(&self, img: &Mat) -> Result<TextDetOutput, EngineError>
```

### 2. `src/rec.rs`
**Current OpenCV usage:**
- `Mat::default()`
- `imgproc::resize()`
- `resized.at_2d::<core::Vec3b>()`

**Migration:**
```rust
// Before:
let mut resized = Mat::default();
imgproc::resize(img, &mut resized, core::Size::new(w, h), ...)

// After:
let mut resized = Mat::default();
crate::image_impl::resize(img, &mut resized, Size::new(w, h), INTER_LINEAR)?;
```

### 3. `src/geometry.rs`
**Most complex - needs:**
- `get_rotate_crop_image()` - uses warpPerspective
- `resize_image_within_bounds()` - uses resize
- Coordinate transformations

**Key changes:**
```rust
// Before:
pub fn get_rotate_crop_image(img: &Mat, points: &[core::Point2f; 4]) -> opencv::Result<Mat>

// After:
use crate::image_impl::*;
pub fn get_rotate_crop_image(img: &Mat, points: &[Point2f; 4]) -> Result<Mat>
```

### 4. `src/postprocess.rs`
**OpenCV usage:**
- `imgproc::fillPoly()` - for creating masks
- `imgproc::findContours()` - for contour detection  
- `imgproc::minAreaRect()` - for minimum area rectangle

**Migration notes:**
- `fillPoly`: Use `imageproc::drawing::draw_polygon()`
- `findContours`: Needs custom implementation or use `contour-tracing` crate
- `minAreaRect`: Already implemented in `image_impl.rs`

### 5. `src/preprocess.rs`
**OpenCV usage:**
- Image resizing
- Image padding
- Normalization

### 6. `src/bin/rapidocr_json.rs`
**Simplest migration:**
```rust
// Before:
use opencv::{core::Point2f, imgcodecs, prelude::*};
let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;

// After:
use rapidocr::image_impl::{Point2f, imread};
let img = imread(img_path)?;
```

### 7. `src/cal_rec_boxes.rs`
Just needs Point2f type alias.

## Testing Strategy

1. Build with OpenCV feature: `cargo build --features use-opencv`
2. Test OpenCV variant: `./target/debug/rapidocr_json <args>`
3. Build pure Rust: `cargo build`  
4. Test pure Rust variant: `./target/debug/rapidocr_json <args>`
5. Compare outputs using `compare_python_rust.py`

## Known Limitations of Pure Rust Version

The pure Rust implementation has some approximations:
- **minAreaRect**: Uses axis-aligned bounding box instead of rotating calipers
- **Perspective transform**: Basic implementation, may have edge cases
- **findContours**: Needs proper implementation (not in image_impl yet)

## Next Steps

1. Start with simplest modules (`bin/rapidocr_json.rs`, `cal_rec_boxes.rs`)
2. Move to image operations (`geometry.rs`, `preprocess.rs`)
3. Handle complex cases (`postprocess.rs` - needs contour implementation)
4. Test thoroughly against Python reference

## Performance Notes

Pure Rust version should be:
- ✅ Easier to cross-compile
- ✅ No C++ runtime dependency
- ✅ Smaller binary size
- ⚠️ Possibly slower without SIMD optimizations
- ⚠️ Different floating-point behavior may cause minor differences

## Example: Complete Module Migration

See `src/image_impl.rs` for the complete abstraction layer implementation with both backends.
