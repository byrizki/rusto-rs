# RustO Benchmark Suites

This directory contains evaluation harnesses, ground-truth datasets, and reproducible benchmarks comparing RustO against industry-standard OCR engines and inference runtimes.

---

## 1. Benchmark Suites

| Benchmark Suite | Path | Focus | Key Reference |
|---|---|---|---|
| **OmniDocBench Comprehensive Evaluation** | [`omnidocbench/`](omnidocbench/) | Multi-page real-world document OCR, reading order, and layout analysis across 21 configurations | [OmniDocBench Report](omnidocbench/OMNIDOCBENCH_EVALUATION_REPORT.md) / [GitHub Issue #17](https://github.com/byrizki/rusto-rs/issues/17) |
| **Inference Engine Comparison** | [`engine_comparison/`](engine_comparison/) | Low-level runtime execution (RTen vs ONNX Runtime vs MNN) | [`ORT_RTEN_MNN_COMPARISON.md`](ORT_RTEN_MNN_COMPARISON.md) |
| **Concurrency & Throughput Suite** | [`concurrency/`](concurrency/) | End-to-end single-call latency, concurrent throughput (C=1 to C=8), and memory working set | [`concurrency/README.md`](concurrency/README.md) |
| **Base Pipeline Baseline** | [`BASE_PIPELINE_COMPARISON.md`](BASE_PIPELINE_COMPARISON.md) | Initial baseline benchmarks across model tiers | [`COMPARISON_RESULT.md`](COMPARISON_RESULT.md) |

---

## 2. OmniDocBench Benchmark Highlights

Evaluating across 30 diverse multi-column documents (academic literature, textbooks, newspapers, exam papers):

### Base OCR (Pure Detection + Recognition)

| Engine / Runtime | Binary Size | Model Weights | Mean Latency ↓ | Peak RSS ↓ | Accuracy (1 - NED) ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **PaddleOCR + PP-OCRv6 Tiny** | 120.1 MB | 6.1 MB | **628.7 ms** | 696.2 MB | 45.33% |
| **RapidOCR + PP-OCRv6 Tiny** | 22.7 MB | 6.1 MB | 1,030.1 ms | 395.8 MB | 44.84% |
| **RustO NN OCR (Pure-Rust MNN) + Indra Models** | **2.9 MB** | **5.4 MB** | **582.9 ms** | **188.0 MB** | **45.70%** |
| **RustO + Indra Models (RTen)** | **6.7 MB** | **5.4 MB** | 728.7 ms | **155.5 MB** | **45.69%** |
| **RustO NN OCR (Pure-Rust MNN) + PP-OCRv6 Tiny (FP32)** | **2.9 MB** | 5.9 MB | 1,127.7 ms | 307.9 MB | **46.53%** |
| **RustO NN + PP-OCRv6 Tiny (FP16)** | **2.9 MB** | **3.1 MB** | 1,142.8 ms | 307.5 MB | **46.57%** |
| **RustO NN + PP-OCRv6 Tiny (INT8)** | **2.9 MB** | **3.3 MB** | 1,821.8 ms | 377.5 MB | **46.74%** |
| **RustO + PP-OCRv6 Tiny (RTen FP32)** | **6.7 MB** | 6.1 MB | 996.2 ms | **193.4 MB** | **45.43%** |
| **RustO + PP-OCRv6 Tiny (RTen INT8)** | **6.7 MB** | **3.4 MB** | 1,113.5 ms | **177.2 MB** | **45.52%** |
| **RustO + PP-OCRv6 Tiny (Alibaba MNN C++)** | **5.5 MB** | 5.9 MB | 878.3 ms | **207.3 MB** | 39.60% |
| **Tesseract OCR (v5 / PSM 6)** | 4.2 MB | 3.9 MB | 1,755.5 ms | **174.4 MB** | 41.14% |
| **ocrs (RTen Original)** | 12.2 MB | 11.6 MB | 2,372.2 ms | 1,271.7 MB | 44.33% |
| **PaddleOCR + PP-OCRv6 Small** | 120.1 MB | 29.8 MB | 2,390.5 ms | 928.7 MB | 44.68% |
| **RapidOCR + PP-OCRv6 Small** | 22.7 MB | 29.8 MB | 3,522.0 ms | 508.3 MB | 44.55% |
| **RustO NN + PP-OCRv6 Small (INT8)** | **2.9 MB** | 16.1 MB | 6,970.7 ms | 766.2 MB | 45.14% |
| **RustO NN + PP-OCRv6 Medium (INT8)** | **2.9 MB** | 75.9 MB | 21,914.1 ms | 1,032.8 MB | 44.59% |

### PP-OCRv6 Precision & Quantization (FP32 vs FP16 vs INT8)

| Configuration | Precision | Model Weights | Mean Latency ↓ | P95 Latency ↓ | Peak RSS ↓ | Accuracy (1 - NED) ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **RustO NN + PP-OCRv6 Tiny** | **FP32** | 5.9 MB | 1,127.7 ms | 2,292.4 ms | 307.9 MB | 46.53% |
| **RustO NN + PP-OCRv6 Tiny** | **FP16** | **3.1 MB** | 1,142.8 ms | **2,210.3 ms** | 307.5 MB | **46.57%** |
| **RustO NN + PP-OCRv6 Tiny** | **INT8** | **3.3 MB** | 1,821.8 ms | 3,972.2 ms | 377.5 MB | **46.74%** |
| **RustO RTen + PP-OCRv6 Tiny** | **INT8** | **3.4 MB** | **1,113.5 ms** | 2,323.1 ms | **177.2 MB** | 45.52% |
| **RapidOCR Tiny (ONNX Runtime)** | **FP16** | 3.0 MB | 1,398.0 ms | 2,657.7 ms | 516.7 MB | 44.86% |
| **RapidOCR Tiny (ONNX Runtime)** | **INT8** | 3.3 MB | 1,992.8 ms | 2,862.9 ms | 400.8 MB | 44.75% |

### Full Pipeline (Native Document Understanding & Layout Recovery)

| Engine & Ecosystem | Pipeline Description | Latency ↓ | Peak RSS ↓ | Accuracy (1 - NED) ↑ |
|:---|:---|:---:|:---:|:---:|
| **RustO NN Full + Indra Models (Indra Order)** | Pure Rust (`det` + `orient` + `rec` + `order.onnx`) | **599.9 ms** | **184.6 MB** | **63.90%** |
| **Indra OCR (Official Python)** | Python ONNX (`det` + `orient` + `rec` + `order.onnx`) | 3,111.6 ms | 472.2 MB | 63.76% |
| **RustO NN + Pure-Rust Heuristic Order** | Pure Rust (`det` + `rec` + spatial heuristic order) | 1,634.8 ms | 454.3 MB | **62.72%** |
| **RustO RTen Full + Indra Models (Indra Order)** | Native Rust (`det` + `orient` + `rec` + `order.rten`) | **735.9 ms** | **161.5 MB** | **62.04%** |
| **RustO RTen + Pure-Rust Heuristic Order** | Native Rust (`det` + `rec` + spatial heuristic order) | 1,529.5 ms | **266.8 MB** | **60.98%** |
| **Tesseract OCR (v5 / PSM 3)** | C++ / Leptonica (`tesseract` full page) | 1,847.7 ms | **184.6 MB** | 60.29% |
| **RustO NN Full + PP-OCRv6 Tiny (CDLA)** | Pure Rust (`det` + `rec` + `cls` + `orient` + `layout_cdla`) | 2,332.4 ms | 503.8 MB | **57.91%** |
| **PaddleOCR + PP-Structure** | Python Paddle (`det` + `rec` + `layout` + `sorted_boxes`) | 20,676.5 ms | 929.6 MB | 57.18% |
| **RustO RTen Full + PP-OCRv6 Tiny (CDLA)** | Native Rust (`det` + `rec` + `cls` + `orient` + `layout_cdla`) | 2,079.7 ms | 351.6 MB | 56.46% |
| **RapidOCR + RapidLayout Tiny** | Python ONNX (`det` + `rec` + `layout` + `sorted_boxes`) | 1,561.7 ms | 453.5 MB | 53.85% |

### Standalone & Hybrid Model Mixing (Indra Order + PP-OCR)

| Configuration | Model Setup | Latency ↓ | Accuracy (1 - NED) ↑ | USA Today NED ↓ |
|:---|:---|:---:|:---:|:---:|
| **PP-OCRv6 Tiny (Base 1D)** | Baseline RapidOCR Tiny | 1,048.1 ms | 44.79% | 0.7815 |
| **Hybrid: PP-OCR Tiny + Indra Order** | PP-OCR Tiny Det + Rec + **Indra `order.rten`** | **1,045.7 ms** | **62.49% (+17.7%)** | **0.0417 (95.8%)** |
| **Hybrid: Indra Det + PP-OCR Rec + Order** | Indra Det + PP-OCR Tiny Rec + **Indra `order.rten`** | **755.3 ms** | **63.03% (+18.2%)** | **0.0389 (96.1%)** |
| **Indra Full Pipeline** | Indra Det + Orient + Rec + **Indra `order.rten`** | **536.3 ms** | **63.75% (+19.0%)** | **0.0348 (96.5%)** |

See [`omnidocbench/HYBRID_OCR_RESEARCH_REPORT.md`](omnidocbench/HYBRID_OCR_RESEARCH_REPORT.md) and [`omnidocbench/README.md`](omnidocbench/README.md) for full instructions and detailed findings.

