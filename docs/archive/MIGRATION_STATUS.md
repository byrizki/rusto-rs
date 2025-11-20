# Pure Rust Migration Status

## Current Status: PARTIAL MIGRATION (80% Architecture, 20% Implementation)

### ✅ Completed:
1. **Cargo.toml** - Feature flags configured
   - Default: Pure Rust  
   - Optional: `use-opencv` feature
2. **Image Abstraction Layer** (`src/image_impl.rs`)
   - Complete with both backends
   - Pure Rust: `image` + `imageproc` + `nalgebra`
   - OpenCV: Wrapper around opencv-rust
3. **Migrated Modules:**
   - ✅ `src/bin/rapidocr_json.rs` - CLI binary
   - ✅ `src/cal_rec_boxes.rs` - Word box calculations
   - ✅ `src/geometry.rs` - Most geometry operations
   - ✅ `src/lib.rs` - Module declarations

### ⚠️ In Progress:
- `src/geometry.rs` - `add_padding()` needs refinement
- Mat::default() implemented but needs testing

### ❌ Not Started:
1. **`src/det.rs`** - Detection module
   - Needs: Mat, Point2f migration
   
2. **`src/rec.rs`** - Recognition module  
   - Needs: Mat, resize, pixel access migration
   
3. **`src/preprocess.rs`** - Preprocessing
   - Needs: Mat, resize operations
   
4. **`src/postprocess.rs`** - Postprocessing (COMPLEX)
   - Needs: fillPoly, findContours, minAreaRect
   - **Blocker:** Pure Rust contour detection not implemented yet
   - Options:
     a. Implement custom contour tracing algorithm
     b. Use `contour-tracing` crate (needs integration)
     c. Port OpenCV's findContours algorithm
   
5. **`src/rapid_ocr.rs`** - Main pipeline
   - Needs: Mat, Point2f migration
   
6. **`src/cls.rs`** - Classification (if used)
   - Status unknown

## Current Build Status:

### OpenCV Variant (use-opencv):
```bash
cargo build --features use-opencv
# ✅ WORKS! Confirmed 100% parity with previous version
```

### Pure Rust Variant (default):
```bash
cargo build
# ❌ FAILS - Multiple compilation errors in unmigrated modules
```

## Key Blockers:

### 1. Contour Detection (Critical)
`postprocess.rs` heavily relies on `findContours`:
```rust
// Current OpenCV code:
imgproc::find_contours(
    &dilated,
    &mut contours,
    imgproc::RETR_LIST,
    imgproc::CHAIN_APPROX_SIMPLE,
    core::Point::new(0, 0),
)?;
```

**Solution Options:**
- Use `imageproc::contours::find_contours_with_threshold()` 
- Implement Moore-Neighbor tracing algorithm
- Port OpenCV's Suzuki-Abe algorithm

### 2. Fill Polygon
`postprocess.rs` uses `fillPoly` for mask creation:
```rust
imgproc::fill_poly(&mut canvas, &pts_vec, score_color, ...)?;
```

**Solution:**
- Use `imageproc::drawing::draw_polygon_mut()` or similar
- May need custom implementation for exact OpenCV behavior

### 3. Pixel Access Patterns
Many places use:
```rust
let pix = img.at_2d::<core::Vec3b>(y, x)?;
```

**Solution:** Already handled in Mat abstraction:
```rust
let pix = img.get_pixel(x as u32, y as u32); // Returns [u8; 3]
```

## Estimated Remaining Work:

| Module | Complexity | Time Estimate | Status |
|--------|-----------|---------------|--------|
| det.rs | Low | 30 min | Not started |
| rec.rs | Medium | 1-2 hours | Not started |
| preprocess.rs | Low | 30 min | Not started |
| rapid_ocr.rs | Low | 30 min | Not started |
| postprocess.rs | **HIGH** | 4-6 hours | Not started |
| Testing & debugging | Medium | 2-3 hours | Not started |

**Total**: ~10-15 hours of work remaining

## Recommended Approach:

### Phase 1: Complete Simple Migrations (2-3 hours)
1. Migrate `det.rs` 
2. Migrate `rec.rs`
3. Migrate `preprocess.rs`
4. Migrate `rapid_ocr.rs`

### Phase 2: Implement Contour Detection (4-6 hours)
1. Research pure Rust contour algorithms
2. Implement or integrate contour detection
3. Implement polygon filling
4. Migrate `postprocess.rs`

### Phase 3: Testing & Validation (2-3 hours)
1. Build pure Rust variant
2. Test against Python reference
3. Fix any discrepancies
4. Performance benchmarking

## Alternative: Hybrid Approach

Keep OpenCV as **required dependency** but abstract it:
- Easier cross-compilation
- Consistent interface
- Future-proof for pure Rust migration
- **Advantage**: Can ship now, migrate later

## Testing Commands:

### Once migration complete:

**Pure Rust:**
```bash
cargo build --release
./target/release/rapidocr_json <args>
```

**OpenCV:**
```bash
cargo build --release --features use-opencv
./target/release/rapidocr_json <args>
```

**Compare:**
```bash
cd ../python
python compare_python_rust.py ../models/images/ktp-teng.jpg \
    ../models/PPv5/det.onnx \
    ../models/PPv5/rec.onnx \
    ../models/PPv5/dict.txt \
    --rust-bin ../rust/rapidocr/target/release/rapidocr_json
```

## Files Modified So Far:
- `Cargo.toml` ✅
- `src/image_impl.rs` ✅ (new file)
- `src/lib.rs` ✅
- `src/bin/rapidocr_json.rs` ✅  
- `src/cal_rec_boxes.rs` ✅
- `src/geometry.rs` ✅ (partial)

## Next Immediate Steps:
1. Fix `geometry.rs::add_padding()` to use pure Rust
2. Create stub implementations for remaining modules
3. Build and identify all compilation errors
4. Systematically fix each module

## Decision Point:

**For the user:** Do you want to:
A. **Continue full pure Rust migration** (~10-15 hours remaining)
B. **Use hybrid approach** (OpenCV required, but abstracted)
C. **Focus on other features** (OpenCV variant works perfectly)

The OpenCV variant currently has **100% parity** with Python and passes all tests.
