# RustO Concurrency, Latency & Throughput Benchmark Suite

This directory contains evaluation scripts, workload configurations, and empirical benchmark results measuring end-to-end single-call latency, concurrent throughput ($QPS$), thread scaling, and resident memory footprints ($RSS$) for `rusto` (`rusto-nn` pure-Rust engine) compared against `rusto-rten` (RTen ONNX engine).

---

## 1. Directory Structure

- [`benchmark_suite.py`](benchmark_suite.py) — Full evaluation runner executing:
  - **Part 1**: Single-call latency benchmarks across PP-OCRv6 Tiny and Indra OCR models on both `chronam-ad-page.png` and `USAToday 960` document samples.
  - **Part 2**: Multi-threaded concurrent throughput benchmark measuring $QPS$, mean latency, and wall-clock times at concurrency levels $C=1$, $C=2$, $C=4$, and $C=8$.
- [`pool_vs_rayon.py`](pool_vs_rayon.py) — Comparison harness tracking user CPU time, system CPU time, peak RSS (via `/usr/bin/time -v`), and concurrent throughput.
- [`thread_sweep.py`](thread_sweep.py) — Scaling test sweeping thread counts ($1$ to $16$) under various concurrency loads.
- [`results/`](results/) — Captured evaluation outputs:
  - [`results/benchmark_results.json`](results/benchmark_results.json): Structured JSON results from the latest `benchmark_suite.py` run.
  - OCR prediction JSON dumps for validation and text alignment checks (`chronam_indra.json`, `chronam_ppocr.json`, `usatoday_indra.json`, `usatoday_ppocr.json`, etc.).

---

## 2. Running Benchmarks

### Prerequisites

Ensure the release binaries are built:

```bash
cargo build --release --bin rusto
cargo build --release -p rusto-rten --bin rusto-rten
```

### Run Latency & Concurrency Suite

```bash
python3 benchmarks/concurrency/benchmark_suite.py
```

Results will be printed to stdout and saved directly into [`results/benchmark_results.json`](results/benchmark_results.json).

### Run Thread Scaling & Concurrency Test

```bash
python3 benchmarks/concurrency/thread_sweep.py
```

---

## 3. Latest Benchmark Results

### A. Single-Call Latency (`chronam-ad-page.png` & `USAToday 960`)

| Pipeline & Model | `rusto-nn` Latency | `rusto-rten` Latency | Difference |
| :--- | :--- | :--- | :--- |
| **Chronam: PP-OCRv6 Tiny (Plain)** | **1741.9 ms** | 1783.8 ms | `rusto-nn` is **41.9 ms FASTER (2.4%)** |
| **Chronam: PP-OCRv6 Tiny (CDLA Layout)** | 1881.1 ms | 1868.2 ms | Comparable (0.7%) |
| **Chronam: Indra OCR (Plain)** | **678.4 ms** | 1049.6 ms | `rusto-nn` is **371.2 ms FASTER (35.4%)** |
| **Chronam: Indra OCR (Neural Order)** | **751.3 ms** | 1078.2 ms | `rusto-nn` is **326.9 ms FASTER (30.3%)** |
| **USAToday 960: Indra OCR (Plain)** | **1673.6 ms** | 2902.8 ms | `rusto-nn` is **1229.3 ms FASTER (42.3%)** |
| **USAToday 960: Indra OCR (Neural Order)** | **1541.6 ms** | 2738.4 ms | `rusto-nn` is **1196.8 ms FASTER (43.7%)** |

### B. Concurrent Request Throughput ($QPS$) (`chronam-ad-page.png`, 8 Requests)

| Concurrency ($C$) | `rusto-nn` QPS | `rusto-nn` Mean Latency | `rusto-rten` QPS | `rusto-rten` Mean Latency |
| :---: | :---: | :---: | :---: | :---: |
| **$C = 1$** | **0.56 req/s** | **1774.3 ms** | 0.53 req/s | 1873.9 ms |
| **$C = 2$** | 0.54 req/s | 3585.3 ms | 0.71 req/s | 2813.7 ms |
| **$C = 4$** | 0.62 req/s | 6043.4 ms | 0.78 req/s | 5043.7 ms |
| **$C = 8$** | 0.58 req/s | 12604.6 ms | 0.65 req/s | 11753.9 ms |

### C. Peak Memory Footprint (HWM) & Minor Page Faults

| Configuration | Peak RSS (HWM) | Minor Page Faults | Settled Post-Run RSS |
| :--- | :---: | :---: | :---: |
| **`rusto-nn` (Previous Baseline)** | 604 MB | ~160,000 | 584 MB |
| **`rusto-nn` (Optimized, Serial Rec)** | **223 MB** | **90,524** | **115 MB** |
| **`rusto-nn` (Optimized, 4 Workers)** | 418 MB | 125,000 | 385 MB |
| **`rusto-rten` (Reference)** | ~288 MB | ~80,000 | 260 MB |

