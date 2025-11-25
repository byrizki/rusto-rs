# RustO Examples

This directory contains examples demonstrating various features of the RustO OCR library.

## Basic Examples

### perf_test.rs
Performance testing example that measures OCR throughput and latency.

```bash
cargo run --release --example perf_test
```

## Advanced Features Examples

### orient_example.rs
Document orientation classification using LCNet model.

**Requirements:**
- `models/mnn/PP-LCNet_x1_0_doc_ori_infer.mnn`

```bash
cargo run --release --example orient_example
```

**Features:**
- Detects document rotation (0°, 90°, 180°, 270°)
- Reports confidence scores
- Provides rotation recommendations

### layout_example.rs
Document layout detection using DocLayout model.

**Requirements:**
- `models/mnn/PP-DocLayout-M_infer.mnn` or `models/mnn/PP-DocBlockLayout_infer.mnn`

```bash
cargo run --release --example layout_example
```

**Features:**
- Detects layout regions (text, title, figure, table, etc.)
- Reports bounding boxes and confidence scores
- Identifies 10 different layout element types

### markdown_export.rs
Demonstrates OCR result export in multiple formats.

**Requirements:**
- Basic OCR models (detection + recognition)

```bash
cargo run --release --example markdown_export
```

**Outputs:**
- `output.md` - Markdown formatted results with performance metrics
- `output.txt` - Plain text, ordered by position
- `output_positions.txt` - Text with coordinate information
- `output.json` - JSON format for programmatic access

### complete_pipeline.rs
Complete OCR pipeline with all optional features enabled.

**Requirements:**
- All models (detection, recognition, orientation, rectification, layout)

```bash
cargo run --release --example complete_pipeline
```

**Features:**
- Automatic feature detection (enables only if models are available)
- Combines all processing steps
- Comprehensive result reporting
- Performance metrics for each stage

## Model Files

Place model files in the `models/mnn/` directory:

### Required (for basic OCR):
- `PP-OCRv5_mobile_det.mnn` - Text detection
- `PP-OCRv5_mobile_rec.mnn` or `en_PP-OCRv5_mobile_rec_infer.mnn` - Text recognition
- `ppocr_keys_*.txt` - Dictionary file (e.g., `ppocr_keys_en.txt` for English)

### Optional (for advanced features):
- `PP-LCNet_x1_0_doc_ori_infer.mnn` - Document orientation classification
- `UVDoc_infer.mnn` - Text rectification
- `PP-DocLayout-M_infer.mnn` - Layout detection (compact)
- `PP-DocBlockLayout_infer.mnn` - Layout detection (full)

## Converting Models

Use the provided conversion script to convert Paddle/ONNX models to MNN format:

```bash
# From Paddle format
python convert_paddle_to_mnn.py --model ./paddle_models/PP-LCNet_x1_0_doc_ori_infer

# From ONNX format
python convert_paddle_to_mnn.py --format onnx --model ./onnx_models/UVDoc_infer

# Auto-detect format
python convert_paddle_to_mnn.py --ocr-dir ./ocr_models
```

## Usage Examples

### Basic OCR
```rust
use rusto::{RustO, RustOConfig};

let config = RustOConfig {
    det_model_path: "models/mnn/PP-OCRv5_mobile_det.mnn".to_string(),
    rec_model_path: "models/mnn/en_PP-OCRv5_mobile_rec_infer.mnn".to_string(),
    dict_path: "models/mnn/ppocr_keys_en.txt".to_string(),
};

let mut ocr = RustO::new(config)?;
let results = ocr.ocr("image.jpg")?;

// Export as markdown
let markdown = results.to_markdown();
std::fs::write("output.md", markdown)?;
```

### With Optional Features
```rust
use rusto::rusto_ocr::RustO;
use rusto::{OrientClassifier, OrientConfig};
use std::path::PathBuf;

// Create base OCR
let mut ocr = RustO::new_ppv5(
    "models/mnn/PP-OCRv5_mobile_det.mnn",
    "models/mnn/en_PP-OCRv5_mobile_rec_infer.mnn",
    "models/mnn/ppocr_keys_en.txt",
)?;

// Enable orientation classification
ocr.global.use_orient = true;
let orient_config = OrientConfig::default(
    PathBuf::from("models/mnn/PP-LCNet_x1_0_doc_ori_infer.mnn")
);
let orient = OrientClassifier::new(orient_config)?;
ocr = ocr.with_orient(orient);

// Run OCR with orientation detection
let result = ocr.run("document.jpg")?;

if let Some(orientation) = result.orientation {
    println!("Document is rotated {} degrees", orientation.degrees());
}
```

## Configuration

All features are optional and configurable via `GlobalConfig`:

```rust
let mut config = GlobalConfig::default();

// Standard features (enabled by default)
config.use_det = true;      // Text detection
config.use_cls = true;      // Text angle classification  
config.use_rec = true;      // Text recognition

// Optional features (disabled by default)
config.use_orient = true;   // Document orientation
config.use_rectify = true;  // Text rectification
config.use_layout = true;   // Layout detection

// Quality settings
config.text_score = 0.5;    // Minimum confidence threshold
config.return_word_box = false;  // Word-level bounding boxes
```

## Performance Tips

1. **Use FP16 models** for faster inference (converted with `--fp16` flag)
2. **Batch processing** - Process multiple images in sequence to amortize model loading
3. **Disable unused features** - Only enable orient/rectify/layout when needed
4. **Resize large images** - Set appropriate `max_side_len` in GlobalConfig
5. **Profile your pipeline** - Check elapse times to identify bottlenecks

## Troubleshooting

### Model not found errors
- Ensure model files are in `models/mnn/` directory
- Check file names match exactly (case-sensitive)
- Verify models were converted successfully

### Low accuracy
- Check if correct dictionary file is being used
- Try enabling text rectification for curved text
- Adjust `text_score` threshold in GlobalConfig

### Slow performance
- Use FP16 models
- Disable unused optional features
- Check if running on correct backend (Metal/OpenCL/Vulkan)

## See Also

- [DOCUMENT_OCR_FEATURES.md](../DOCUMENT_OCR_FEATURES.md) - Detailed API documentation
- [MODEL_CONVERSION.md](../MODEL_CONVERSION.md) - Model conversion guide
- [README.md](../README.md) - Main project documentation
