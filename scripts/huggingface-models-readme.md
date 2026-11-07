---
license: mit
pipeline_tag: image-to-text
tags:
- ocr
- rten
- rust
- text-recognition
- text-detection
- onnx
- indra-ocr
- paddleocr
---

# RustO Pre-trained OCR Models

Standalone pre-trained OCR models in pure-Rust Alibaba MNN (`rusto-nn`) and RTen formats, optimized for [RustO! (`rusto-rs`)](https://github.com/byrizki/rusto-rs).

These models enable zero-dependency, high-speed neural OCR inference across Rust, Python, Node.js, WebAssembly (WASM), Android, iOS, React Native, C# .NET, Go, and Flutter.

## Direct URL Access (CORS Enabled)

All model weights and dictionary files are globally accessible with native CORS headers:

```text
https://huggingface.co/byrizki/rten-models/resolve/main/<folder>/<filename>
```

---

## Model Catalog by Directory

### 1. Indra OCR (`indraocr/`) — Default Flagship

| Path | Size | Description |
|:---|:---:|:---|
| `indraocr/det.rten` | 396 KB | Base Detection |
| `indraocr/rec.rten` | 4.9 MB | Base Recognition |
| `indraocr/meta.json` | 207 KB | Vocabulary Metadata |
| `indraocr/order.rten` | 2.9 MB | Neural Reading Order & Column Sorter |
| `indraocr/orient.rten` | 360 KB | Page Orientation Classifier |
| `indraocr/rec_cjk.rten` | 9.8 MB | CJK Recognition (Chinese, Japanese, Korean) |
| `indraocr/rec_rtl.rten` | 4.7 MB | Right-to-Left Recognition (Arabic, Persian, Urdu) |
| `indraocr/rec_world.rten` | 5.4 MB | Extended World Languages Recognition |
| `indraocr/script.rten` | 190 KB | Script & Writing System Classifier |
| `indraocr/mathcls.rten` | 190 KB | Math Formula Detection / Classifier |
| `indraocr/math_enc.rten` | 9.5 MB | Math Formula Encoder |
| `indraocr/math_dec_init.rten` | 1.3 MB | Math Formula Decoder (Init Step) |
| `indraocr/math_dec_step.rten` | 8.1 MB | Math Formula Decoder (Autoregressive Step) |
| `indraocr/math_meta.json` | 11 KB | Math LaTeX Vocabulary |

### 2. PP-OCRv6 Tiers

| Folder | Detection (`det.rten`) | Recognition (`rec.rten`) | Dictionary (`dict.txt`) | Description |
|:---|:---:|:---:|:---:|:---|
| `ppocrv6-tiny/` | 1.8 MB | 4.3 MB | 27 KB | PP-OCRv6 Tiny (Ultra-lightweight) |
| `ppocrv6-tiny-int8/` | 704 KB | 2.7 MB | 27 KB | PP-OCRv6 Tiny INT8 Quantized |
| `ppocrv6-small/` | 9.6 MB | 21 MB | 27 KB | PP-OCRv6 Small High-Recall |
| `ppocrv6-small-int8/` | 2.7 MB | 14 MB | 27 KB | PP-OCRv6 Small INT8 Quantized |
| `ppocrv6-medium/` | 60 MB | 74 MB | 27 KB | PP-OCRv6 Medium High-Capacity |
| `ppocrv6-medium-int8/` | 16 MB | 61 MB | 27 KB | PP-OCRv6 Medium INT8 Quantized |

### 3. PP-OCRv5 Tiers

| Folder | Detection (`det.rten`) | Recognition (`rec.rten`) | Dictionary (`dict.txt`) | Description |
|:---|:---:|:---:|:---:|:---|
| `ppocrv5-mobile/` | 4.7 MB | 16 MB | 73 KB | PP-OCRv5 Mobile |
| `ppocrv5-server/` | 85 MB | 81 MB | 73 KB | PP-OCRv5 Server |

### 4. PP-OCRv4 Tiers

| Folder | Detection (`det.rten`) | Recognition (`rec.rten`) | Dictionary (`dict.txt`) | Description |
|:---|:---:|:---:|:---:|:---|
| `ppocrv4-mobile/` | 4.6 MB | 11 MB | 26 KB | PP-OCRv4 Mobile |
| `ppocrv4-server/` | 109 MB | 87 MB | 26 KB | PP-OCRv4 Server |

### 5. Multilingual Recognizers (`multilingual/`)

| Language | Recognition Asset | Dictionary Asset |
|:---|:---|:---|
| **Arabic** | `multilingual/ppocrv5_rec_arabic.rten` | `multilingual/ppocrv5_dict_arabic.txt` |
| **Cyrillic** | `multilingual/ppocrv5_rec_cyrillic.rten` | `multilingual/ppocrv5_dict_cyrillic.txt` |
| **Devanagari (Hindi)** | `multilingual/ppocrv5_rec_devanagari.rten` | `multilingual/ppocrv5_dict_devanagari.txt` |
| **Greek** | `multilingual/ppocrv5_rec_el.rten` | `multilingual/ppocrv5_dict_el.txt` |
| **East Slavic** | `multilingual/ppocrv5_rec_eslav.rten` | `multilingual/ppocrv5_dict_eslav.txt` |
| **Korean** | `multilingual/ppocrv5_rec_korean.rten` | `multilingual/ppocrv5_dict_korean.txt` |
| **Latin** | `multilingual/ppocrv5_rec_latin.rten` | `multilingual/ppocrv5_dict_latin.txt` |
| **Tamil** | `multilingual/ppocrv5_rec_ta.rten` | `multilingual/ppocrv5_dict_ta.txt` |
| **Telugu** | `multilingual/ppocrv5_rec_te.rten` | `multilingual/ppocrv5_dict_te.txt` |
| **Thai** | `multilingual/ppocrv5_rec_th.rten` | `multilingual/ppocrv5_dict_th.txt` |
| **Japanese** | `multilingual/ppocrv4_rec_japan.rten` | `multilingual/ppocrv4_dict_japan.txt` |
| **Traditional Chinese** | `multilingual/ppocrv4_rec_chinese_cht.rten` | `multilingual/ppocrv4_dict_chinese_cht.txt` |
| **Kannada** | `multilingual/ppocrv4_rec_kannada.rten` | `multilingual/ppocrv4_dict_kannada.txt` |

### 6. Document Intelligence & Pipeline Extras (`modules/`)

| Path | Size | Purpose |
|:---|:---:|:---|
| `modules/cls.rten` | 590 KB | Text-Line 180° Direction Classifier |
| `modules/orient.rten` | 6.4 MB | Document Page 90°/180°/270° Orientation Detector |
| `modules/uvdoc.rten` | 30 MB | 3D Document Dewarping & Unwarping |
| `modules/layout_cdla.rten` | 7.1 MB | Document Layout Analysis (CDLA 10-Class) |
| `modules/layout_publaynet.rten` | 7.1 MB | Document Layout Analysis (PubLayNet 5-Class) |

---

## Integrity Verification

SHA256 checksums are recorded in `SHA256SUMS`.
