# RustO Benchmark Suites

This directory contains evaluation harnesses, ground-truth datasets, and reproducible benchmarks comparing RustO against industry-standard OCR engines and inference runtimes.

---

## 1. Benchmark Suites

| Benchmark Suite | Path | Focus | Key Reference |
|---|---|---|---|
| **OmniDocBench Comprehensive Evaluation** | [`omnidocbench/`](omnidocbench/) | Multi-page real-world document OCR, reading order, and layout analysis across 21 configurations | [OmniDocBench Report](omnidocbench/OMNIDOCBENCH_EVALUATION_REPORT.md) / [GitHub Issue #17](https://github.com/byrizki/rusto-rs/issues/17) |
| **Inference Engine Comparison** | [`engine_comparison/`](engine_comparison/) | Low-level runtime execution (RTen vs ONNX Runtime vs MNN) | [`ORT_RTEN_MNN_COMPARISON.md`](ORT_RTEN_MNN_COMPARISON.md) |
| **Base Pipeline Baseline** | [`BASE_PIPELINE_COMPARISON.md`](BASE_PIPELINE_COMPARISON.md) | Initial baseline benchmarks across model tiers | [`COMPARISON_RESULT.md`](COMPARISON_RESULT.md) |

---

## 2. OmniDocBench Benchmark Highlights

Evaluating across 30 diverse multi-column documents (academic literature, textbooks, newspapers, exam papers):

### Base OCR (Pure Detection + Recognition)

| Engine / Runtime | Binary Size | Model Weights | Mean Latency ↓ | Peak RSS ↓ | Accuracy (1 - NED) ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **PaddleOCR + PP-OCRv6 Tiny** | 120.1 MB | 6.1 MB | **605.3 ms** | 659.3 MB | 45.33% |
| **RustO + PP-OCRv6 Tiny (RTen)** | **8.2 MB** | 6.1 MB | **976.0 ms** | **178.0 MB** | **45.41%** |
| **RapidOCR + PP-OCRv6 Tiny** | 22.7 MB | 6.1 MB | 1,011.5 ms | 392.5 MB | 44.84% |
| **RustO + Indra Models (RTen Base)** | **8.2 MB** | **5.4 MB** | **1,054.7 ms** | **196.2 MB** | **45.97%** |
| **RustO + Indra Models (Pure-Rust MNN Base)** | **8.2 MB** | **5.4 MB** | 1,141.3 ms | 285.9 MB | **45.97%** |
| **RustO + PP-OCRv6 Tiny (Pure-Rust MNN)** | **8.2 MB** | 5.9 MB | 1,304.2 ms | 253.1 MB | 43.21% |
| **ocrs (RTen Original)** | 12.2 MB | 11.6 MB | 2,059.1 ms | 1,900.4 MB | 44.33% |
| **PaddleOCR + PP-OCRv6 Small** | 120.1 MB | 29.8 MB | 2,390.5 ms | 928.7 MB | 44.68% |
| **RustO + PP-OCRv6 Small (RTen)** | **8.2 MB** | 29.8 MB | 3,135.0 ms | **249.4 MB** | 44.64% |
| **RapidOCR + PP-OCRv6 Small** | 22.7 MB | 29.8 MB | 3,522.0 ms | 508.3 MB | 44.55% |

### Full Pipeline (Native Document Understanding & Layout Recovery)

| Engine & Ecosystem | Pipeline Description | Latency ↓ | Peak RSS ↓ | Accuracy (1 - NED) ↑ |
|:---|:---|:---:|:---:|:---:|
| **RustO Full + Indra Models (RTen)** | Pure Rust (`det` + `orient` + `rec` + `order.rten`) | **1,113.6 ms** | **223.8 MB** | **46.00%** |
| **RustO Full + Indra Models (Pure-Rust MNN)** | Pure Rust (`det` + `orient` + `rec` + `order.rten`) | **1,205.2 ms** | **316.1 MB** | **46.00%** |
| **Indra OCR (Official Python)** | Python ONNX (`det` + `orient` + `rec` + `order.onnx`) | 1,415.4 ms | 492.0 MB | **63.76%** |
| **PaddleOCR + PP-Structure** | Python Paddle (`det` + `rec` + `layout` + `sorted_boxes`) | 5,571.4 ms | 10,624.3 MB | 57.18% |
| **RapidOCR + RapidLayout Tiny** | Python ONNX (`det` + `rec` + `layout` + `sorted_boxes`) | 1,177.5 ms | 449.5 MB | 53.85% |
| **RustO + PP-Structure Tiny** | Pure Rust (`det` + `rec` + `layout_cdla.mnn`) | **1,509.3 ms** | **296.9 MB** | 46.00% |

### Standalone & Hybrid Model Mixing (Indra Order + PP-OCR)

| Configuration | Model Setup | Latency ↓ | Accuracy (1 - NED) ↑ | USA Today NED ↓ |
|:---|:---|:---:|:---:|:---:|
| **PP-OCRv6 Tiny (Base 1D)** | Baseline RapidOCR Tiny | 1,048.1 ms | 44.79% | 0.7815 |
| **Hybrid: PP-OCR Tiny + Indra Order** | PP-OCR Tiny Det + Rec + **Indra `order.rten`** | **1,045.7 ms** | **62.49% (+17.7%)** | **0.0417 (95.8%)** |
| **Hybrid: Indra Det + PP-OCR Rec + Order** | Indra Det + PP-OCR Tiny Rec + **Indra `order.rten`** | **755.3 ms** | **63.03% (+18.2%)** | **0.0389 (96.1%)** |
| **Indra Full Pipeline** | Indra Det + Orient + Rec + **Indra `order.rten`** | **536.3 ms** | **63.75% (+19.0%)** | **0.0348 (96.5%)** |

See [`omnidocbench/HYBRID_OCR_RESEARCH_REPORT.md`](omnidocbench/HYBRID_OCR_RESEARCH_REPORT.md) and [`omnidocbench/README.md`](omnidocbench/README.md) for full instructions and detailed findings.

