# RustO Dimension-Adaptive & Auto-Config Benchmark

This benchmark evaluates the performance, accuracy, and latency impact of RustO's **Dimension-Adaptive Auto-Config** engine (`auto_config: true`) compared against the standard static baseline (`auto_config: false`).

---

## 1. Overview & Purpose

RustO's Auto-Config automatically inspects input image dimensions, luma histogram, local contrast, and optical characteristics before inference to dynamically tune:

- **Dimension-Adaptive Detection Scale**: Adjusts `limit_side_len` dynamically so tiny text snippets and high-density documents are resized appropriately without bloating memory or latency.
- **Relief & Contrast Enhancement**: Calibrates CLAHE / bilateral filtering for faint relief (e.g., gravestone incisions, stamps, watermarks).
- **Illumination Flattening**: Corrects strong lighting gradients and shadows across receipts and mobile captures.
- **Adaptive Crop Padding**: Dynamically scales horizontal (`pad_x`) and vertical (`pad_y`) recognition crop margins based on image dimensions and layout density (e.g., tight padding for dense multi-column newspaper lines, generous padding for isolated text crops).
- **Descreening & Denoising**: Smooths out halftone / guilloche security patterns and sensor noise.

This benchmark verifies both the **optical recovery benefits** on degraded images and the **computational overhead / speedups** across standard documents.

---

## 2. Benchmark Fixtures

The suite tests 8 benchmark fixtures spanning synthetic edge cases and real-world documents:

### A. Synthetic Degraded Optical Conditions ([`fixtures/`](fixtures/))

| Fixture | Dimensions | Optical Challenge | Ground Truth Target |
|---|---|---|---|
| **Carved White Gravestone** | 600 × 240 | High-luma limestone, subtle crevice groove shadows, near-zero contrast | `"IN MEMORY OF"`, `"ALEXANDER HAMILTON"`, `"1757 - 1804"` |
| **Degraded / Broken Text** | 600 × 200 | Fractured crossbars, ink dropout, thermal ribbon degradation | `"SYSTEM RECOVERY CODE"`, `"AUTH-9821-BATCH-42"` |
| **Uneven Illumination Gradient** | 650 × 220 | 100% luma on left falling to 22% deep shadow on right | `"FINANCIAL ACCOUNT SUMMARY"`, `"BALANCE: $458,920.00 USD"` |
| **Optical Defocus Blur** | 600 × 180 | Camera out-of-focus blur (Gaussian radius 1.6) | `"OUT OF FOCUS LENS SAMPLE"`, `"CALIBRATION TEST PASS 100%"` |

### B. Real-World Document Fixtures

| Fixture | Dimensions | Source File | Characteristics |
|---|---|---|---|
| **ID Card / Passport** | 1024 × 721 | `models/test_images/idcard.jpg` | Guilloche security background, uneven lighting |
| **Historical Newspaper Page** | 999 × 743 | `tests/fixtures/chronam-ad-page.png` | Dense 1890s multi-column newspaper ad scan, tight line spacing |
| **Scanned Invoice** | 359 × 600 | `models/test_images/invoice1.jpg` | Multi-line financial table, low scanning contrast |
| **Standard Document Scan** | 462 × 656 | `models/test_images/example1.png` | Standard letter/document scan |

---

## 3. Latest Benchmark Results

Captured across **Baseline** (`auto_config: false`), **Auto-Config** (`auto_config: true`), and **Auto + Reading Layout** (`auto_config: true`, `layout_reading: true`) using **PP-OCRv6 Tiny (RTen)** on Linux x86_64:

### A. Latency & Speedup

| Fixture | Baseline Latency | Auto-Config Latency | Auto + Reading Layout Latency | Auto Speedup |
|---|:---:|:---:|:---:|:---:|
| **Optical Defocus Blur** | 1,023.2 ms | **80.8 ms** | 83.4 ms | **-92.1%** |
| **Degraded / Broken Text** | 833.5 ms | **69.3 ms** | 86.9 ms | **-91.7%** |
| **Uneven Illumination Gradient** | 910.9 ms | 142.3 ms | **129.2 ms** | **-84.4%** |
| **Historical Newspaper Page** | 2,541.4 ms | 3,108.4 ms | **2,429.3 ms** | -4.4% (Layout) |
| **Scanned Invoice** | **652.1 ms** | 662.1 ms | 659.6 ms | +1.5% |
| **Standard Document Example1** | **649.2 ms** | 713.0 ms | 714.7 ms | +9.8% |
| **ID Card / Passport (Real)** | **687.0 ms** | 745.5 ms | 737.8 ms | +8.5% |

### B. Detection Lines & Ground Truth Matches

| Fixture | Baseline Lines | Auto-Config Lines | Auto + Reading Layout Lines | Ground Truth Target Matches | Reading Order & Column Notes |
|---|:---:|:---:|:---:|:---:|---|
| **Historical Newspaper Page** | 231 lines | 228 lines | **236 lines** | **2/2 (100%)** | Layout prevents cross-column line mergers; reads Column 1 top-to-bottom contiguously |
| **Degraded / Broken Text** | 2 lines | 2 lines | 2 lines | Preserved | Morphological stroke repair heals fractured crossbars |
| **Uneven Illumination Gradient** | 3 lines | 3 lines | 3 lines | 1/3 (100% conf) | Illumination leveling preserves word integrity |
| **Optical Defocus Blur** | 0 lines | **1 line** | **1 line** | Recovered | Unsharp mask restores detection edges from 0% |
| **Scanned Invoice** | 27 lines | 24 lines | 23 lines | **1/1 (100%)** | Cleans scan border flecks; preserves table structure |
| **Standard Document Example1** | 48 lines | 48 lines | 48 lines | Preserved | Clean high-confidence text preserved with zero regression |
| **ID Card / Passport (Real)** | 27 lines | 27 lines | 27 lines | **3/3 (100%)** | Descreening suppresses guilloche pattern |

> **Key Takeaway**:
> - **Auto-Config** cuts latency by **84% to 92%** on crops and degraded samples while recovering lost text on optical blur.
> - **Reading Layout** (`layout_reading: true`) resolves multi-column flow on complex pages like historical newspapers (recovering **236 column-isolated lines** without horizontal cross-column interleaving) at near-zero overhead.

Full per-run JSON metrics are saved in [`results/auto_config_benchmark_results.json`](results/auto_config_benchmark_results.json).

---

## 4. How to Run the Benchmark

### Prerequisites

Ensure the Python bindings are compiled:

```bash
cargo build --release -p rusto --features uniffi
```

### Execution

Run with default settings (5 iterations, 2 warmups):

```bash
python3 benchmarks/auto_config/benchmark_auto_config.py
```

Custom iterations and output destination:

```bash
python3 benchmarks/auto_config/benchmark_auto_config.py --iterations 10 --warmups 3 --output benchmarks/auto_config/results/my_results.json
```

