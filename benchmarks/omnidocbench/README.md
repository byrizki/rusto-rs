# OmniDocBench Evaluation Suite for RustO

This benchmark suite integrates the [OpenDataLab OmniDocBench](https://github.com/opendatalab/OmniDocBench) dataset and evaluation harness for evaluating RustO's OCR pipeline, aligned with the methodology in [Issue #17](https://github.com/byrizki/rusto-rs/issues/17).

---

## 1. Pinned Commits & Setup

To ensure strict reproducibility matching [Issue #17](https://github.com/byrizki/rusto-rs/issues/17):

| Component | Upstream Repository | Pinned SHA | Details |
|---|---|---|---|
| **Evaluator** | [`opendatalab/OmniDocBench` (GitHub)](https://github.com/opendatalab/OmniDocBench) | `f133a71e9e91c3621c7ce8994200a7b394a06eb3` | Evaluation code suite (`pdf_validation.py`, matching metrics) |
| **Dataset** | [`opendatalab/OmniDocBench` (Hugging Face)](https://huggingface.co/datasets/opendatalab/OmniDocBench) | `aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec` | Ground truth `OmniDocBench.json` & page images |

### Dataset Subsets
- **English-tagged pages (Default)**: 755 pages (721 with scored text, ~780 MB). This is the exact dataset evaluated in Issue #17.
- **Seeded Sample**: 101 seeded proportional document pages.
- **Full Benchmark**: 1,651 pages across all document types and languages (~1.35 GB).

---

## 2. Directory Layout

```
benchmarks/omnidocbench/
├── README.md                  # This documentation
├── evaluate_pipeline.py       # RustO evaluation runner and metric calculator
├── evaluator/                 # Pinned evaluator repository (ignored by git)
│   ├── configs/
│   ├── pdf_validation.py
│   └── src/
├── dataset/                   # Pinned dataset repository (ignored by git)
│   ├── OmniDocBench.json      # Ground truth annotations
│   └── images/                # Page images
└── predictions/               # Generated OCR prediction markdown files (ignored by git)
    ├── tiny/
    └── small/
```

---

## 3. Pulling Dataset and Evaluator

Use the centralized script `scripts/download_omnidocbench.sh` to download or update the suite. The script automatically caches downloaded assets locally, verifies existing files, and skips redundant fetches on re-runs:

```bash
# Pull the evaluator and the 755 English-tagged images (Issue #17 setup) [default]
./scripts/download_omnidocbench.sh --english-only

# Or pull all 1651 pages across all languages (~1.35 GB)
./scripts/download_omnidocbench.sh --all

# Or pull the 101 seeded sample pages
./scripts/download_omnidocbench.sh --sample-only

# Or specify a persistent external cache directory
./scripts/download_omnidocbench.sh --english-only --cache-dir ~/.cache/omnidocbench

# Or only pull evaluator without images
./scripts/download_omnidocbench.sh --evaluator-only
```

---

## 4. Running Pipeline Evaluation

Ensure `rusto-rs` is compiled in release mode first:

```bash
cargo build --release --bin rusto-rs
```

### Quick Single-Page Evaluation
Test against the multi-column newspaper sample from Issue #17 (`newspaper_USAToday-2025-01-08@magazinesclubnew_page_012.png`):

```bash
./benchmarks/omnidocbench/evaluate_pipeline.py \
  --tier tiny \
  --image newspaper_USAToday-2025-01-08@magazinesclubnew_page_012.png
```

### Run on a Subset
Evaluate on the first 10 English pages:

```bash
./benchmarks/omnidocbench/evaluate_pipeline.py \
  --tier tiny \
  --subset english \
  --limit 10
```

### Run Full English Evaluation (Issue #17 Benchmark)
Evaluate all 755 English pages with PP-OCRv6 Tiny:

```bash
./benchmarks/omnidocbench/evaluate_pipeline.py \
  --tier tiny \
  --subset english
```

Evaluate with PP-OCRv6 Small:

```bash
./benchmarks/omnidocbench/evaluate_pipeline.py \
  --tier small \
  --subset english
```

### Standalone & Hybrid Model Mixing Benchmark
Evaluate standalone mixing of Indra and PP-OCR sub-models (e.g. PP-OCR detection/recognition with Indra neural reading order):

```bash
python3 benchmarks/omnidocbench/benchmark_hybrid_mixing.py
```

### Metrics Reported
- **Mean Latency / Median / P95**: Wall-clock pipeline execution timing.
- **Mean Normalized Edit Distance**: `Levenshtein(gt, pred) / max(len(gt), len(pred))` on concatenated text blocks in reading order.
- **Reading Order Diagnostics**: Highlights multi-column layout interleaving vs paragraph retention.

---

## 5. Comprehensive Benchmark Results

The complete 21-pipeline benchmark on OmniDocBench evaluates engines across **Part I: Base OCR** (pure text detection and recognition with identical 1D spatial sorting) and **Part II: Full Pipeline** (native multi-stage document understanding with layout analysis and reading order recovery).

> For detailed architectural breakdowns, visual charts, and page-by-page diagnostics, see the [Full OmniDocBench Evaluation Report](OMNIDOCBENCH_EVALUATION_REPORT.md). Raw machine metrics are recorded in [`results/benchmark_metrics.json`](results/benchmark_metrics.json).

### Part I: Base OCR Benchmark (Pure Det + Rec)

| Engine / Runtime | Pipeline Target | Binary Size | Model Weights | Mean Latency ↓ | P95 Latency ↓ | Peak RSS ↓ | Mean NED ↓ | Accuracy (1-NED) ↑ | USA Today NED ↓ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PaddleOCR + PP-OCRv6 Tiny** | Python (Paddle ONNX) | 120.1 MB | 6.1 MB | **628.7 ms** | **1,371.1 ms** | 696.2 MB | 0.5467 | 45.33% | **0.7331** |
| **RapidOCR + PP-OCRv6 Tiny** | Python (ONNX Runtime) | 22.7 MB | 6.1 MB | 1,030.1 ms | 2,141.6 ms | 395.8 MB | 0.5516 | 44.84% | 0.7831 |
| **MNN OCR + PP-OCRv6 Tiny (Direct)** | C++ MNN Standalone | **3.7 MB** | 5.9 MB | 3,686.5 ms | 11,792.8 ms | 849.4 MB | 0.6084 | 39.16% | 0.7786 |
| **RTen OCR (Direct Rust RTen)** | Native Rust RTen | **3.3 MB** | 6.1 MB | 26,691.0 ms | 76,390.8 ms | 1,013.9 MB | 0.5495 | 45.05% | 0.7813 |
| **ocrs (RTen Original)** | Native Rust (ocrs) | 12.2 MB | 11.6 MB | 2,372.2 ms | 5,573.1 ms | 1,271.7 MB | 0.5567 | 44.33% | 0.7396 |
| **RustO NN OCR + PP-OCRv6 Tiny (FP32)** | Pure-Rust MNN (`rusto-nn`) | **2.9 MB** | 5.9 MB | 1,127.7 ms | 2,292.4 ms | 307.9 MB | **0.5347** | **46.53%** | **0.7326** |
| **RustO NN + PP-OCRv6 Tiny (FP16)** | Pure-Rust MNN (`rusto-nn`) | **2.9 MB** | **3.1 MB** | 1,142.8 ms | 2,210.3 ms | 307.5 MB | **0.5343** | **46.57%** | 0.7336 |
| **RustO NN + PP-OCRv6 Tiny (INT8)** | Pure-Rust MNN (`rusto-nn`) | **2.9 MB** | **3.3 MB** | 1,821.8 ms | 3,972.2 ms | 377.5 MB | **0.5326** | **46.74%** | **0.7314** |
| **RustO + PP-OCRv6 Tiny (RTen FP32)** | Native Rust (`rusto-rten`) | **6.7 MB** | 6.1 MB | 996.2 ms | 2,215.0 ms | **193.4 MB** | 0.5457 | 45.43% | 0.7317 |
| **RustO + PP-OCRv6 Tiny (RTen INT8)** | Native Rust (`rusto-rten`) | **6.7 MB** | **3.4 MB** | 1,113.5 ms | 2,323.1 ms | **177.2 MB** | 0.5448 | 45.52% | 0.7411 |
| **RustO + PP-OCRv6 Tiny (Alibaba MNN)** | Alibaba MNN C++ (`rusto-mnn`)| **5.5 MB** | 5.9 MB | 878.3 ms | 1,607.4 ms | **207.3 MB** | 0.6040 | 39.60% | 0.7332 |
| **RustO NN OCR + Indra Models** | Pure-Rust MNN (`rusto-nn`) | **2.9 MB** | **5.4 MB** | **582.9 ms** | **1,557.0 ms** | **188.0 MB** | **0.5430** | **45.70%** | 0.7840 |
| **RustO + Indra Models (RTen)** | Native Rust (`rusto-rten`) | **6.7 MB** | **5.4 MB** | 728.7 ms | 2,235.0 ms | **155.5 MB** | **0.5431** | **45.69%** | 0.7840 |
| **RustO + Indra Models (Alibaba MNN)** | Alibaba MNN C++ (`rusto-mnn`)| **4.7 MB** | **5.4 MB** | 1,470.1 ms | 4,001.5 ms | **245.5 MB** | 0.9380 | 6.20% | 0.9919 |
| **Tesseract OCR (v5 / PSM 6)** | C++ / Leptonica | 4.2 MB | 3.9 MB | 1,755.5 ms | 6,536.1 ms | **174.4 MB** | 0.5886 | 41.14% | 0.7669 |
| **RustO NN + PP-OCRv6 Small (INT8)** | Pure-Rust MNN (`rusto-nn`) | **2.9 MB** | 16.1 MB | 6,970.7 ms | 17,791.0 ms | 766.2 MB | 0.5486 | 45.14% | 0.7685 |
| **RustO NN + PP-OCRv6 Medium (INT8)** | Pure-Rust MNN (`rusto-nn`) | **2.9 MB** | 75.9 MB | 21,914.1 ms | 54,295.4 ms | 1,032.8 MB | 0.5541 | 44.59% | 0.7842 |
| **RapidOCR + PP-OCRv6 Tiny (FP16)** | Python (ONNX Runtime) | 22.7 MB | **3.0 MB** | 1,398.0 ms | 2,657.7 ms | 516.7 MB | 0.5514 | 44.86% | 0.7831 |
| **RapidOCR + PP-OCRv6 Tiny (INT8)** | Python (ONNX Runtime) | 22.7 MB | **3.3 MB** | 1,992.8 ms | 2,862.9 ms | 400.8 MB | 0.5525 | 44.75% | 0.7829 |

### Part II: Full Pipeline Benchmark (Native Document Understanding)

| Engine & Ecosystem | Native Pipeline Components | Binary Size | Model Weights | Mean Latency ↓ | P95 Latency ↓ | Peak RSS ↓ | Mean NED ↓ | Accuracy ↑ | USA Today NED ↓ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RustO NN Full + Indra Models (Indra Order)** | `det` + `orient` + `rec` + **`order`** | **2.9 MB** | 8.6 MB | **599.9 ms** | **1,430.2 ms** | **184.6 MB** | **0.3610** | **63.90%** | **0.0350** |
| **Indra OCR (Official Python)** | `det` + `orient` + `rec` + **`order`** | 22.7 MB | 8.6 MB | 3,111.6 ms | 5,067.0 ms | 472.2 MB | 0.3624 | 63.76% | **0.0348** |
| **RustO NN + Pure-Rust Heuristic Order** | `det` + `rec` + spatial heuristic order | **2.9 MB** | 5.9 MB | 1,634.8 ms | 3,081.0 ms | 454.3 MB | **0.3728** | **62.72%** | **0.0356** |
| **RustO RTen Full + Indra Models (Indra Order)** | `det` + `orient` + `rec` + **`order`** | **6.7 MB** | 8.6 MB | **735.9 ms** | 2,171.9 ms | **161.5 MB** | **0.3796** | **62.04%** | **0.0331** |
| **RustO RTen + Pure-Rust Heuristic Order** | `det` + `rec` + spatial heuristic order | **6.7 MB** | 6.1 MB | 1,529.5 ms | 2,971.4 ms | **266.8 MB** | 0.3902 | 60.98% | **0.0328** |
| **Tesseract OCR (v5 / PSM 3)** | C++ / Leptonica (PSM 3 full page) | 4.2 MB | 3.9 MB | 1,847.7 ms | 6,893.0 ms | **184.6 MB** | 0.3971 | 60.29% | **0.0317** |
| **RustO NN Full + PP-OCRv6 Tiny (CDLA)** | `det` + `rec` + `cls` + `orient` + **`layout_cdla`** | **2.9 MB** | 49.8 MB | 2,332.4 ms | 4,748.4 ms | 503.8 MB | 0.4209 | 57.91% | **0.0383** |
| **PaddleOCR + PP-Structure** | `det` + `rec` + **`layout`** + **`sorted_boxes`** | 120.1 MB | 23.0 MB | 20,676.5 ms | 47,844.8 ms | 929.6 MB | 0.4282 | 57.18% | 0.6638 |
| **RustO RTen Full + PP-OCRv6 Tiny (CDLA)** | `det` + `rec` + `cls` + `orient` + **`layout_cdla`** | **6.7 MB** | 20.2 MB | 2,079.7 ms | 4,383.2 ms | 351.6 MB | 0.4354 | 56.46% | **0.0322** |
| **RapidOCR + RapidLayout Tiny** | `det` + `rec` + **`layout`** + **`sorted_boxes`** | 22.7 MB | 13.1 MB | 1,561.7 ms | 2,950.1 ms | 453.5 MB | 0.4615 | 53.85% | 0.5735 |
| **RustO NN Full + PP-OCRv6 Tiny (PubLayNet)** | `det` + `rec` + **`layout_publaynet`** | **2.9 MB** | 19.7 MB | 2,336.3 ms | 4,774.9 ms | 503.5 MB | 0.4629 | 53.71% | 0.4783 |
| **RustO RTen Full + PP-OCRv6 Tiny (PubLayNet)** | `det` + `rec` + **`layout_publaynet`** | **6.7 MB** | 20.2 MB | 2,188.8 ms | 4,522.5 ms | 354.6 MB | 0.4753 | 52.47% | 0.4727 |

### Part III: Standalone & Hybrid Model Mixing Benchmark

Evaluates whether sub-models from Indra OCR can be extracted and run standalone or mixed with PP-OCR components (such as running PP-OCR detection and recognition with Indra's neural reading order transformer).

> For full architectural details, model contracts, and cross-model analysis, see the [Hybrid OCR Research Report](HYBRID_OCR_RESEARCH_REPORT.md). Raw machine metrics are recorded in [`results/hybrid_mixing_metrics.json`](results/hybrid_mixing_metrics.json).

| Pipeline Configuration | Mean Latency ↓ | P95 Latency ↓ | Mean NED ↓ | Accuracy (1-NED) ↑ | USA Today NED ↓ | Gain vs Base |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. PP-OCRv6 Tiny (Base Det+Rec, 1D sort)** | 1,048.1 ms | 2,003.2 ms | 0.5521 | 44.79% | 0.7815 | Baseline |
| **2. PP-OCRv6 Small (Base Det+Rec, 1D sort)** | 3,678.2 ms | 6,902.0 ms | 0.5564 | 44.36% | 0.7845 | -0.43% |
| **3. Indra OCR (Base Det+Rec, 1D sort)** | **447.8 ms** | **1,135.3 ms** | 0.5454 | 45.46% | 0.7831 | +0.67% |
| **4. Hybrid: PP-OCR Tiny (Det+Rec) + Indra Order** | **1,045.7 ms** | 2,053.2 ms | **0.3751** | **62.49%** | **0.0417** | **+17.70%** |
| **5. Hybrid: PP-OCR Small (Det+Rec) + Indra Order** | 3,629.7 ms | 7,130.1 ms | 0.3766 | 62.34% | 0.0380 | **+17.55%** |
| **6. Hybrid: Indra Det + PP-OCR Tiny Rec (1D sort)** | 793.9 ms | 2,093.7 ms | 0.5510 | 44.90% | 0.7832 | +0.11% |
| **7. Hybrid: Indra Det + PP-OCR Tiny Rec + Indra Order** | **755.3 ms** | 2,057.1 ms | **0.3697** | **63.03%** | **0.0389** | **+18.24%** |
| **8. Hybrid: PP-OCR Tiny Det + Indra Rec (1D sort)** | 1,853.7 ms | 6,744.2 ms | 0.5469 | 45.31% | 0.7830 | +0.52% |
| **9. Hybrid: PP-OCR Tiny Det + Indra Rec + Indra Order** | 1,547.9 ms | 5,008.3 ms | 0.3901 | 60.99% | 0.0347 | **+16.20%** |
| **10. Indra Native Full (Det+Orient+Rec+Order)** | **536.3 ms** | **1,273.5 ms** | **0.3625** | **63.75%** | **0.0348** | **+18.96%** |

### Key Conclusions

1. **Pure Rust Runtime Efficiency**: The standalone Pure-Rust `rusto` binary is only **2.9 MB** (down from 8.2 MB), running Pure-Rust MNN (`rusto-nn`) in sub-second latency (**582.9 ms** on Indra Base) and consuming **155.5 MB (`rusto-rten`) to 307.9 MB (`rusto-nn`) RAM** in Base OCR, outperforming RapidOCR (395.8 MB) and PaddleOCR (696.2 MB) on memory footprint while achieving strong raw accuracy (**46.53%** Tiny FP32 / **46.74%** Tiny INT8).
2. **Layout Integration Gains**: Pure-Rust layout and reading order recovery (`order.onnx` / `order.rten` or pure-Rust heuristic order) jumps document transcription accuracy from ~45%–47% to **63.90% (NED 0.3610)** at **599.9 ms** mean latency while consuming only **184.6 MB peak RSS**, strictly outperforming the official Python Indra pipeline (**63.76% / 472.2 MB**) with 5.19x lower latency and 61% less memory. Pure-Rust heuristic reading order reaches **62.72%** accuracy (`rusto`) / **60.98%** (`rusto-rten`) with zero extra layout model weights.
3. **FP16 & INT8 Precision Tradeoffs**: **PP-OCRv6 Tiny FP16 (`rusto-nn`)** halves weights to **3.1 MB** at near-identical latency (**1,142.8 ms**) and **46.57%** accuracy, while **PP-OCRv6 Tiny INT8 (`rusto-rten`)** runs in **1,113.5 ms** at just **177.2 MB** RAM (**45.52%**). On `rusto-nn`, Tiny INT8 compresses weights to **3.3 MB** and hits **46.74%** accuracy (**1,821.8 ms** due to CPU dequantization overhead, mirroring RapidOCR INT8 at **1,992.8 ms**). Across all tiers, PP-OCRv6 Small/Medium INT8 deliver no English accuracy gain over Tiny while scaling latency 6x–19x, confirming Tiny (FP32 or FP16) as the optimal production tier.
4. **Neural Reading Order**: Dedicated sequence prediction transformer (`order.onnx` / `order.rten`) delivers 63.90% document accuracy by resolving multi-column ordering on complex documents, demonstrating that reading order sequencing is the primary driver of document-level NED.

