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
| **PaddleOCR + PP-OCRv6 Tiny** | Target 4 (Base) | 120.1 MB | 6.1 MB | **605.3 ms** | **1,322.0 ms** | 659.3 MB | 0.5467 | 45.33% | **0.7331** |
| **RustO + PP-OCRv6 Tiny (RTen)** | Target 1 (RTen) | **8.3 MB** | 6.1 MB | **976.0 ms** | **2,005.0 ms** | **178.0 MB** | **0.5459** | **45.41%** | **0.7319** |
| **RapidOCR + PP-OCRv6 Tiny** | Target 3 (Base) | 22.7 MB | 6.1 MB | 1,011.5 ms | 2,233.1 ms | 392.5 MB | 0.5516 | 44.84% | 0.7831 |
| **RustO + Indra Models (RTen Base)** | Target 7 (RTen) | **8.3 MB** | **5.4 MB** | **1,054.7 ms** | 2,823.1 ms | **196.2 MB** | **0.5403** | **45.97%** | 0.7839 |
| **RustO + Indra Models (Pure MNN Base)** | Target 7 (MNN) | **4.5 MB** | **5.4 MB** | 1,141.3 ms | 3,255.9 ms | 285.9 MB | **0.5403** | **45.97%** | 0.7839 |
| **RustO + PP-OCRv6 Tiny (Pure MNN)** | Target 1 (MNN) | **4.5 MB** | 5.9 MB | 1,304.2 ms | 2,772.0 ms | 253.1 MB | 0.5679 | 43.21% | 0.9643 |
| **ocrs (RTen Original)** | Target 5 (Base) | 12.2 MB | 11.6 MB | 2,059.1 ms | 4,507.6 ms | 1,900.4 MB | 0.5567 | 44.33% | 0.7396 |
| **PaddleOCR + PP-OCRv6 Small** | Target 4 (Base) | 120.1 MB | 29.8 MB | 2,390.5 ms | 5,271.0 ms | 928.7 MB | 0.5532 | 44.68% | 0.7474 |
| **RustO + PP-OCRv6 Small (Base)** | Target 1 (Base) | **4.5 MB** | 29.8 MB | 2,971.0 ms | 7,854.3 ms | **243.6 MB** | 0.5536 | 44.64% | 0.7656 |
| **RustO + PP-OCRv6 Small (RTen)** | Target 1 (RTen) | **8.3 MB** | 29.8 MB | 3,135.0 ms | 7,728.6 ms | **249.4 MB** | 0.5536 | 44.64% | 0.7656 |
| **RapidOCR + PP-OCRv6 Small** | Target 3 (Base) | 22.7 MB | 29.8 MB | 3,522.0 ms | 6,863.2 ms | 508.3 MB | 0.5545 | 44.55% | 0.7839 |

### Part II: Full Pipeline Benchmark (Native Document Understanding)

| Engine & Ecosystem | Native Pipeline Components | Binary Size | Model Weights | Mean Latency ↓ | P95 Latency ↓ | Peak RSS ↓ | Mean NED ↓ | Accuracy ↑ | USA Today NED ↓ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RustO Full + Indra Models (RTen)** | `det` + `orient` + `rec` + **`order`** | **8.3 MB** | 8.6 MB | **1,113.6 ms** | **2,766.3 ms** | **223.8 MB** | **0.5403** | **46.0%** | **0.7839** |
| **RustO Full + Indra Models (Pure MNN)** | `det` + `orient` + `rec` + **`order`** | **4.5 MB** | 8.6 MB | **1,205.2 ms** | 2,853.5 ms | 316.1 MB | **0.5403** | **46.0%** | **0.7839** |
| **Indra OCR (Official)** | `det` + `orient` + `rec` + **`order`** | 22.7 MB | 8.6 MB | 1,415.4 ms | 2,454.6 ms | 492.0 MB | **0.3624** | **63.76%** | **0.0348** |
| **PaddleOCR + PP-Structure** | `det` + `rec` + **`layout`** + **`sorted_boxes`** | 120.1 MB | 23.0 MB | 5,571.4 ms | 13,002.7 ms | 10,624.3 MB | 0.4282 | **57.18%** | 0.6638 |
| **RapidOCR + RapidLayout Tiny** | `det` + `rec` + **`layout`** + **`sorted_boxes`** | 22.7 MB | 13.1 MB | 1,177.5 ms | 2,380.2 ms | 449.5 MB | 0.4615 | **53.85%** | 0.5735 |
| **RapidOCR + RapidLayout Small** | `det` + `rec` + **`layout`** + **`sorted_boxes`** | 22.7 MB | 36.9 MB | 3,626.6 ms | 7,071.5 ms | 557.5 MB | 0.4649 | 53.51% | 0.5730 |
| **RustO + PP-Structure Tiny** | `det` + `rec` + **`layout_cdla`** | **4.5 MB** | 12.8 MB | 1,509.3 ms | 3,100.2 ms | **296.9 MB** | 0.5404 | 46.0% | 0.9641 |
| **RustO + PP-Structure Small** | `det` + `rec` + **`layout_cdla`** | **4.5 MB** | 36.9 MB | 3,153.4 ms | 7,413.9 ms | **314.7 MB** | 0.4999 | 50.0% | 0.7610 |
| **RustO Full Pipeline Tiny (MNN)** | `det` + `rec` + `cls` + `orient` + **`layout_cdla`** | **4.5 MB** | 49.8 MB | 1,910.5 ms | 3,897.6 ms | **312.7 MB** | 0.5404 | 46.0% | 0.9641 |
| **RustO Full Pipeline Small** | `det` + `rec` + `cls` + `orient` + **`layout_cdla`** | **4.5 MB** | 74.1 MB | 3,893.4 ms | 9,461.9 ms | **342.1 MB** | 0.4999 | 50.0% | 0.7610 |
| **PaddleOCR Full Tiny** | `det` + `rec` + `cls` | 120.1 MB | 6.6 MB | 747.1 ms | 1,640.8 ms | 669.7 MB | 0.5466 | 45.34% | 0.7335 |
| **PaddleOCR Full Small** | `det` + `rec` + `cls` | 120.1 MB | 30.3 MB | 2,664.6 ms | 5,838.2 ms | 942.3 MB | 0.5539 | 44.61% | 0.7464 |
| **RapidOCR Full Tiny** | `det` + `rec` + `cls` | 22.7 MB | 6.6 MB | 1,146.3 ms | 2,174.3 ms | 392.6 MB | 0.5516 | 44.84% | 0.7831 |
| **RapidOCR Full Small** | `det` + `rec` + `cls` | 22.7 MB | 30.3 MB | 3,995.8 ms | 9,790.6 ms | 506.9 MB | 0.5545 | 44.55% | 0.7839 |

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

1. **Pure Rust Runtime Efficiency**: Running on pure Rust execution engines, RustO clocks in at sub-second latency (**976.0 ms** for Tiny RTen) and consumes only **178.0 MB to 253.1 MB RAM** in Base OCR, outperforming RapidOCR ONNX Runtime on memory and latency while matching recognition accuracy.
2. **Layout Integration Gains**: Adding PP-Structure PicoDet layout analysis (`layout_cdla.mnn` / `layout_cdla.onnx`) directly improves document reading order accuracy while consuming only **296.9 MB to 314.7 MB peak RSS** (vs 10.6 GB for PaddleStructure).
3. **Issue #17 Confirmation**: Across all ecosystems, PP-OCRv6 Small models deliver **no accuracy benefit** over Tiny models while increasing latency by 2.5x to 4x. Tiny is confirmed as the optimal production configuration.
4. **Neural Reading Order**: Indra OCR's dedicated sequence prediction transformer (`order.onnx`) delivers 63.8% document accuracy by resolving multi-column ordering, demonstrating that reading order sequencing is the primary driver of document-level NED.

