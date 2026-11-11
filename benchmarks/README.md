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
| **PaddleOCR + PP-OCRv6 Tiny** | 120.1 MB | 6.1 MB | **730.7 ms** | 694.8 MB | 45.33% |
| **RapidOCR + PP-OCRv6 Tiny** | 22.7 MB | 6.1 MB | 1,170.7 ms | 397.1 MB | 44.84% |
| **RustO NN OCR (Pure-Rust MNN) + Indra Models** | **2.1 MB** | **5.4 MB** | **1,827.2 ms** | **190.5 MB** | **45.70%** |
| **RustO NN OCR (Pure-Rust MNN) + PP-OCRv6 Tiny** | **2.1 MB** | 5.9 MB | 2,522.0 ms | **247.8 MB** | **45.44%** |
| **RustO + PP-OCRv6 Tiny (RTen)** | **4.3 MB** | 6.1 MB | 12,020.1 ms | **169.0 MB** | **45.40%** |
| **RustO + PP-OCRv6 Tiny (Alibaba MNN C++)** | **4.7 MB** | 5.9 MB | 2,032.2 ms | **168.1 MB** | 39.57% |
| **Tesseract OCR (v5 / PSM 6)** | 4.2 MB | 3.9 MB | 1,959.2 ms | **174.4 MB** | 41.14% |
| **ocrs (RTen Original)** | 12.2 MB | 11.6 MB | 2,372.2 ms | 1,271.7 MB | 44.33% |
| **PaddleOCR + PP-OCRv6 Small** | 120.1 MB | 29.8 MB | 2,390.5 ms | 928.7 MB | 44.68% |
| **RapidOCR + PP-OCRv6 Small** | 22.7 MB | 29.8 MB | 3,522.0 ms | 508.3 MB | 44.55% |

### Full Pipeline (Native Document Understanding & Layout Recovery)

| Engine & Ecosystem | Pipeline Description | Latency ↓ | Peak RSS ↓ | Accuracy (1 - NED) ↑ |
|:---|:---|:---:|:---:|:---:|
| **RustO NN Full + Indra Models (Indra Order)** | Pure Rust (`det` + `orient` + `rec` + `order.onnx`) | **1,780.2 ms** | **193.9 MB** | **63.90%** |
| **Indra OCR (Official Python)** | Python ONNX (`det` + `orient` + `rec` + `order.onnx`) | 3,111.6 ms | 472.2 MB | 63.76% |
| **Tesseract OCR (v5 / PSM 3)** | C++ / Leptonica (`tesseract` full page) | 1,985.8 ms | **184.6 MB** | 60.29% |
| **RustO NN + Pure-Rust Heuristic Order** | Pure Rust (`det` + `rec` + spatial heuristic order) | 3,263.2 ms | 375.6 MB | 57.06% |
| **PaddleOCR + PP-Structure** | Python Paddle (`det` + `rec` + `layout` + `sorted_boxes`) | 20,676.5 ms | 929.6 MB | 57.18% |
| **RustO RTen Full + PP-OCRv6 Tiny (PubLayNet)** | Native Rust (`det` + `rec` + `layout_publaynet.rten`) | 20,455.2 ms | 398.6 MB | 55.98% |
| **RustO NN Full + PP-OCRv6 Tiny (CDLA)** | Pure Rust (`det` + `rec` + `cls` + `orient` + `layout_cdla`) | 4,837.3 ms | 433.7 MB | 55.03% |
| **RapidOCR + RapidLayout Tiny** | Python ONNX (`det` + `rec` + `layout` + `sorted_boxes`) | 1,561.7 ms | 453.5 MB | 53.85% |

### Standalone & Hybrid Model Mixing (Indra Order + PP-OCR)

| Configuration | Model Setup | Latency ↓ | Accuracy (1 - NED) ↑ | USA Today NED ↓ |
|:---|:---|:---:|:---:|:---:|
| **PP-OCRv6 Tiny (Base 1D)** | Baseline RapidOCR Tiny | 1,048.1 ms | 44.79% | 0.7815 |
| **Hybrid: PP-OCR Tiny + Indra Order** | PP-OCR Tiny Det + Rec + **Indra `order.rten`** | **1,045.7 ms** | **62.49% (+17.7%)** | **0.0417 (95.8%)** |
| **Hybrid: Indra Det + PP-OCR Rec + Order** | Indra Det + PP-OCR Tiny Rec + **Indra `order.rten`** | **755.3 ms** | **63.03% (+18.2%)** | **0.0389 (96.1%)** |
| **Indra Full Pipeline** | Indra Det + Orient + Rec + **Indra `order.rten`** | **536.3 ms** | **63.75% (+19.0%)** | **0.0348 (96.5%)** |

See [`omnidocbench/HYBRID_OCR_RESEARCH_REPORT.md`](omnidocbench/HYBRID_OCR_RESEARCH_REPORT.md) and [`omnidocbench/README.md`](omnidocbench/README.md) for full instructions and detailed findings.

