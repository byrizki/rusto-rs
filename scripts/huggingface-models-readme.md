---
license: mit
pipeline_tag: image-to-text
tags:
- ocr
- mnn
- rten
- onnx
- rust
- text-recognition
- text-detection
- indra-ocr
- paddleocr
---

# RustO Pre-trained OCR Models

Standalone pre-trained OCR models in pure-Rust Alibaba MNN (`rusto-nn`), RTen (`rusto-rten`), and ONNX formats, optimized for [RustO! (`rusto-rs`)](https://github.com/byrizki/rusto-rs).

These models enable zero-dependency, high-speed neural OCR inference across Rust, Python, Node.js, WebAssembly (WASM), Android, iOS, React Native, C# .NET, Go, and Flutter.

## Supported Formats

| Format | Extension | Engine Support | Recommended Environment |
|:---|:---:|:---|:---|
| **MNN** | `.mnn` | `rusto-nn` (Pure-Rust Alibaba MNN) | **Default** for Native (Linux, macOS, Windows, Android, iOS) |
| **RTen** | `.rten` | `rusto-rten` (Pure-Rust RTen) | **Default** for WebAssembly (WASM) & zero-C-toolchain setups |
| **ONNX** | `.onnx` | ONNX Runtime / Direct inspection | Cross-framework inspection, training pipelines, tract |

## Direct URL Access (CORS Enabled)

All model weights and dictionary files are globally accessible with native CORS headers:

```text
https://huggingface.co/byrizki/rusto-models/resolve/main/<folder>/<filename>
```

---

## Model Catalog by Directory

### 1. Indra OCR (`indraocr/`) — Flagship Layout & Multi-Script Engine

| Asset Base | MNN (`.mnn`) | ONNX (`.onnx`) | RTen (`.rten`) | Description |
|:---|:---:|:---:|:---:|:---|
| `indraocr/det` | 384 KB | 381 KB | 396 KB | Base Detection |
| `indraocr/rec` | 4.9 MB | 4.9 MB | 4.9 MB | Base Recognition |
| `indraocr/meta.json` | - | 207 KB | - | Vocabulary Metadata |
| `indraocr/order` | - | 2.9 MB | 2.9 MB | Neural Reading Order & Column Sorter |
| `indraocr/orient` | - | 360 KB | 370 KB | Page Orientation Classifier |
| `indraocr/rec_cjk` | - | 9.8 MB | 9.8 MB | CJK Recognition (Chinese, Japanese, Korean) |
| `indraocr/rec_rtl` | - | 4.6 MB | 4.7 MB | Right-to-Left Recognition (Arabic, Persian, Urdu) |
| `indraocr/rec_world` | - | 5.4 MB | 5.4 MB | Extended World Languages Recognition |
| `indraocr/script` | - | 188 KB | 191 KB | Script & Writing System Classifier |
| `indraocr/mathcls` | - | 187 KB | 191 KB | Math Formula Detection / Classifier |
| `indraocr/math_enc` | - | 9.5 MB | 9.5 MB | Math Formula Encoder |
| `indraocr/math_dec_init` | - | 1.3 MB | 1.4 MB | Math Formula Decoder (Init Step) |
| `indraocr/math_dec_step` | - | 8.1 MB | 8.1 MB | Math Formula Decoder (Autoregressive Step) |
| `indraocr/math_meta.json` | - | 7.7 KB | - | Math LaTeX Vocabulary |

### 2. PP-OCRv6 Tiers

| Folder | Detection Formats | Recognition Formats | Dictionary | Description |
|:---|:---|:---|:---:|:---|
| `ppocrv6-tiny/` | `det.mnn`, `det.onnx`, `det.rten` | `rec.mnn`, `rec.onnx`, `rec.rten` | `dict.txt` (27 KB) | PP-OCRv6 Tiny (Default Ultra-lightweight) |
| `ppocrv6-tiny-int8/` | `det.onnx`, `det.rten` | `rec.onnx`, `rec.rten` | `dict.txt` (27 KB) | PP-OCRv6 Tiny INT8 Quantized |
| `ppocrv6-small/` | `det.onnx` | `rec.onnx` | `dict.txt` (74 KB) | PP-OCRv6 Small High-Recall |
| `ppocrv6-small-int8/` | `det.onnx`, `det.rten` | `rec.onnx`, `rec.rten` | `dict.txt` (74 KB) | PP-OCRv6 Small INT8 Quantized |
| `ppocrv6-medium/` | `det.onnx` | `rec.onnx` | `dict.txt` (74 KB) | PP-OCRv6 Medium High-Capacity |
| `ppocrv6-medium-int8/` | `det.onnx`, `det.rten` | `rec.onnx`, `rec.rten` | `dict.txt` (74 KB) | PP-OCRv6 Medium INT8 Quantized |

### 3. PP-OCRv5 Tiers

| Folder | Detection Formats | Recognition Formats | Dictionary | Description |
|:---|:---|:---|:---:|:---|
| `ppocrv5-mobile/` | `det.onnx`, `det.rten` | `rec.onnx`, `rec.rten` | `dict.txt` (73 KB) | PP-OCRv5 Mobile |
| `ppocrv5-server/` | `det.rten` | `rec.rten` | `dict.txt` (73 KB) | PP-OCRv5 Server |

### 4. PP-OCRv4 Tiers

| Folder | Detection Formats | Recognition Formats | Dictionary | Description |
|:---|:---|:---|:---:|:---|
| `ppocrv4-mobile/` | `det.onnx`, `det.rten` | `rec.onnx`, `rec.rten` | `dict.txt` (26 KB) | PP-OCRv4 Mobile |
| `ppocrv4-server/` | `det.rten` | `rec.rten` | `dict.txt` (26 KB) | PP-OCRv4 Server |

### 5. Document Intelligence & Pipeline Extras (`modules/`)

| Asset Base | MNN (`.mnn`) | ONNX (`.onnx`) | RTen (`.rten`) | Purpose |
|:---|:---:|:---:|:---:|:---|
| `modules/cls` | 519 KB | 572 KB | 605 KB | Text-Line 180° Direction Classifier |
| `modules/orient` | 6.5 MB | 6.5 MB | 6.5 MB | Document Page Orientation Detector |
| `modules/uvdoc` | 31 MB | 31 MB | 31 MB | 3D Document Dewarping & Unwarping |
| `modules/layout_cdla` | 6.9 MB | 7.1 MB | 7.2 MB | Document Layout Analysis (CDLA 10-Class) |
| `modules/layout_publaynet` | 6.9 MB | 7.1 MB | 7.2 MB | Document Layout Analysis (PubLayNet 5-Class) |

### 6. Multilingual Recognizers (`multilingual/`)

| Language | Recognition Assets | Dictionary Asset |
|:---|:---|:---|
| **Arabic** | `ppocrv5_rec_arabic.onnx`, `ppocrv5_rec_arabic.rten` | `ppocrv5_dict_arabic.txt` |
| **Cyrillic** | `ppocrv5_rec_cyrillic.rten` | `ppocrv5_dict_cyrillic.txt` |
| **Devanagari (Hindi)** | `ppocrv5_rec_devanagari.onnx`, `ppocrv5_rec_devanagari.rten` | `ppocrv5_dict_devanagari.txt` |
| **Greek** | `ppocrv5_rec_el.rten` | `ppocrv5_dict_el.txt` |
| **East Slavic** | `ppocrv5_rec_eslav.rten` | `ppocrv5_dict_eslav.txt` |
| **Korean** | `ppocrv5_rec_korean.onnx`, `ppocrv5_rec_korean.rten` | `ppocrv5_dict_korean.txt` |
| **Latin** | `ppocrv5_rec_latin.onnx`, `ppocrv5_rec_latin.rten` | `ppocrv5_dict_latin.txt` |
| **Tamil** | `ppocrv5_rec_ta.rten` | `ppocrv5_dict_ta.txt` |
| **Telugu** | `ppocrv5_rec_te.rten` | `ppocrv5_dict_te.txt` |
| **Thai** | `ppocrv5_rec_th.rten` | `ppocrv5_dict_th.txt` |
| **Japanese** | `ppocrv4_rec_japan.onnx`, `ppocrv4_rec_japan.rten` | `ppocrv4_dict_japan.txt` |
| **Traditional Chinese** | `ppocrv4_rec_chinese_cht.onnx`, `ppocrv4_rec_chinese_cht.rten` | `ppocrv4_dict_chinese_cht.txt` |
| **Kannada** | `ppocrv4_rec_kannada.onnx`, `ppocrv4_rec_kannada.rten` | `ppocrv4_dict_kannada.txt` |

---

## Integrity Verification

SHA256 checksums are recorded in `SHA256SUMS`.
