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
  --layout-model models/DocOCR/layout.mnn \
  --det-model models/ch_PP-OCRv4_det_infer.mnn \
  --rec-model models/ch_PP-OCRv4_rec_infer.mnn \
  --keys-path models/ppocr_keys_v1.txt
```

### Arguments

- `--image`: Path to the input document image
- `--layout-model`: Path to the layout detection model (default: `models/DocOCR/layout.mnn`)
- `--det-model`: Path to the text detection model (default: `models/ch_PP-OCRv4_det_infer.mnn`)
- `--rec-model`: Path to the text recognition model (default: `models/ch_PP-OCRv4_rec_infer.mnn`)
- `--keys-path`: Path to the character dictionary (default: `models/ppocr_keys_v1.txt`)

### Output

The example generates markdown-formatted output with:
- `# Title` for titles
- `**Header**` for headers
- `*Caption*` for figure/table captions
- Plain text for body text
- Placeholder indicators for figures and tables

### Required Models

You need to download or convert the following models:

1. **Layout Model** (`layout.mnn`):
   - Download from RapidDoc models or convert from PaddleOCR layout models
   - Place in `models/DocOCR/`

2. **Detection Model** (`det.mnn`):
   - Convert from PaddleOCR detection models using `convert_paddle_to_mnn.py`

3. **Recognition Model** (`rec.mnn`):
   - Convert from PaddleOCR recognition models using `convert_paddle_to_mnn.py`

4. **Dictionary** (`dict.txt`):
   - Download from PaddleOCR or RapidOCR repositories

See the main [README.md](../README.md) for model conversion instructions.
