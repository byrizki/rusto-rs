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
| **RustO + Indra Models (Base)** | **5.5 MB** | **5.4 MB** | **439.2 ms** | 474.0 MB | **45.46%** |
| **PaddleOCR + PP-OCRv6 Tiny** | 120.1 MB | 6.1 MB | 605.3 ms | 659.3 MB | 45.33% |
| **RapidOCR + PP-OCRv6 Tiny** | 22.7 MB | 6.1 MB | 1,011.5 ms | 392.5 MB | 44.84% |
| **RustO + PP-OCRv6 Tiny** | **5.5 MB** | 6.1 MB | **1,104.4 ms** | **253.9 MB** | 45.34% |
| **PaddleOCR + PP-OCRv6 Small** | 120.1 MB | 29.8 MB | 2,390.5 ms | 928.7 MB | 44.68% |
| **RapidOCR + PP-OCRv6 Small** | 22.7 MB | 29.8 MB | 3,522.0 ms | 508.3 MB | 44.55% |
| **RustO + PP-OCRv6 Small** | **5.5 MB** | 29.8 MB | 3,700.4 ms | 477.9 MB | 44.57% |

### Full Pipeline (Native Document Understanding & Layout Recovery)

| Engine & Ecosystem | Pipeline Description | Latency ↓ | Peak RSS ↓ | Accuracy (1 - NED) ↑ |
|:---|:---|:---:|:---:|:---:|
| **RustO + Indra Full** | Pure Rust (`det` + `orient` + `rec` + `order.onnx`) | **442.5 ms** | **485.1 MB** | **63.75%** |
| **Indra OCR (Official)** | Python ONNX (`det` + `orient` + `rec` + `order.onnx`) | 1,415.4 ms | 492.0 MB | 63.76% |
| **PaddleOCR + PP-Structure** | Python Paddle (`det` + `rec` + `layout` + `sorted_boxes`) | 5,571.4 ms | 10,624.3 MB | 57.18% |
| **RapidOCR + RapidLayout Tiny** | Python ONNX (`det` + `rec` + `layout` + `sorted_boxes`) | 1,177.5 ms | 449.5 MB | 53.85% |
| **RustO + PP-Structure Tiny** | Pure Rust (`det` + `rec` + `layout_cdla.onnx`) | **1,474.3 ms** | **341.8 MB** | **51.33%** |

### Standalone & Hybrid Model Mixing (Indra Order + PP-OCR)

| Configuration | Model Setup | Latency ↓ | Accuracy (1 - NED) ↑ | USA Today NED ↓ |
|:---|:---|:---:|:---:|:---:|
| **PP-OCRv6 Tiny (Base 1D)** | Baseline RapidOCR Tiny | 1,048.1 ms | 44.79% | 0.7815 |
| **Hybrid: PP-OCR Tiny + Indra Order** | PP-OCR Tiny Det + Rec + **Indra `order.rten`** | **1,045.7 ms** | **62.49% (+17.7%)** | **0.0417 (95.8%)** |
| **Hybrid: Indra Det + PP-OCR Rec + Order** | Indra Det + PP-OCR Tiny Rec + **Indra `order.rten`** | **755.3 ms** | **63.03% (+18.2%)** | **0.0389 (96.1%)** |
| **Indra Full Pipeline** | Indra Det + Orient + Rec + **Indra `order.rten`** | **536.3 ms** | **63.75% (+19.0%)** | **0.0348 (96.5%)** |

See [`omnidocbench/HYBRID_OCR_RESEARCH_REPORT.md`](omnidocbench/HYBRID_OCR_RESEARCH_REPORT.md) and [`omnidocbench/README.md`](omnidocbench/README.md) for full instructions and detailed findings.

