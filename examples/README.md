# Examples

This directory contains example applications demonstrating various features of RustO!

## Document Pipeline Demo

**File:** `doc_pipeline_demo.rs`

Demonstrates the full document processing pipeline with layout analysis and OCR.

### Features
- Layout detection using PP-DocLayout models from RapidDoc
- Text recognition using PaddleOCR models
- Markdown output generation
- Support for multiple layout element types

### Usage

```bash
cargo run --example doc_pipeline_demo -- \
  --image path/to/document.jpg \
  --layout-model models/DocOCR/layout.onnx \
  --det-model models/PPOCR_v5/det.onnx \
  --rec-model models/PPOCR_v5/rec.onnx \
  --keys-path models/PPOCR_v5/dict.txt
```

### Arguments

- `--image`: Path to the input document image
- `--layout-model`: Path to the layout detection model (default: `models/DocOCR/layout.onnx`)
- `--det-model`: Path to the text detection model (default: `models/PPOCR_v5/det.onnx`)
- `--rec-model`: Path to the text recognition model (default: `models/PPOCR_v5/rec.onnx`)
- `--keys-path`: Path to the character dictionary (default: `models/PPOCR_v5/dict.txt`)

### Output

The example generates markdown-formatted output with:
- `# Title` for titles
- `**Header**` for headers
- `*Caption*` for figure/table captions
- Plain text for body text
- Placeholder indicators for figures and tables

### Required Models

You need to download the following models:

1. **Layout Model** (`layout.onnx`):
   - Place in `models/DocOCR/`

2. **Detection Model** (`det.onnx` or `det.rten`):
   - Download using `bash scripts/download_models.sh`

3. **Recognition Model** (`rec.onnx` or `rec.rten`):
   - Download using `bash scripts/download_models.sh`

4. **Dictionary** (`dict.txt`):
   - Included with model downloads

See the main [README.md](../README.md) for model conversion instructions.
