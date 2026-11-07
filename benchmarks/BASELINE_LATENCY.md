# Baseline Latency Report: Published RTen (crates.io 0.27.0)

**Date**: 2026-11-03  
**Target Test Image**: `tests/fixtures/chronam-ad-page.png` (Dense newspaper page, 150+ lines)  
**Hardware**: 8-Core CPU Linux x86_64  
**Profile**: `release` (`opt-level = 3`, LTO enabled)  
**Warmups**: 2 | **Timed Iterations**: 10  

---

## 1. PP-OCRv6 Tiny (Published RTen Baseline)

| Stage / Step | Duration (ms) | % of Total | Details |
| :--- | :---: | :---: | :--- |
| **1. Image Load & PNG Decode** | 3.59 ms | 0.2% | `image::load_from_memory` |
| **2. Global Resize & Padding** | 0.30 ms | 0.0% | Aspect bounds & vertical pad |
| **3. Detection Stage (Overall)** | 184.93 ms | 9.5% | Total DBNet detection |
| &nbsp;&nbsp;├─ 3a. Det Preprocess | 33.60 ms | 1.7% | Tensor packing & normalization |
| &nbsp;&nbsp;├─ 3b. Det RTen Inference | 124.38 ms | 6.4% | DBNet forward pass |
| &nbsp;&nbsp;└─ 3c. Det Postprocess | 26.95 ms | 1.4% | Binarization & polygon unclip |
| **4. Coordinate Un-mapping** | 0.00 ms | 0.0% | Inverse scale & padding |
| **5. Text Box Cropping** | 22.21 ms | 1.1% | 282 affine perspective warps |
| **6. Recognition Stage (Overall)** | **1,729.60 ms** | **89.1%** | **47 batches of recognition** |
| **7. Reading Order / Layout Sort** | 0.13 ms | 0.0% | Spatial sort |
| **TOTAL PIPELINE LATENCY** | **1,940.77 ms** | **100.0%** | Full pipeline end-to-end |

* **Detected Boxes**: 282
* **Recognition Batches ($N=6$)**: 47
* **Time per Box Recognition**: **6.133 ms**

---

## 2. Indra OCR (Published RTen Baseline)

| Stage / Step | Duration (ms) | % of Total | Details |
| :--- | :---: | :---: | :--- |
| **1. Image Load & PNG Decode** | 3.62 ms | 0.3% | `image::load_from_memory` |
| **2. Global Resize & Padding** | 0.34 ms | 0.0% | Aspect bounds & vertical pad |
| **3. Detection Stage (Overall)** | 64.58 ms | 5.5% | LineNet detection |
| **4. Coordinate Un-mapping** | 0.00 ms | 0.0% | Inverse scale & padding |
| **5. Text Box Cropping** | 36.99 ms | 3.1% | 179 affine perspective warps |
| **6. Recognition Stage (Overall)** | **1,071.06 ms** | **91.0%** | **30 batches of recognition** |
| **7. Reading Order / Layout Sort** | 0.08 ms | 0.0% | Spatial sort |
| **TOTAL PIPELINE LATENCY** | **1,176.68 ms** | **100.0%** | Full pipeline end-to-end |

* **Detected Boxes**: 179
* **Recognition Batches ($N=6$)**: 30
* **Time per Box Recognition**: **5.984 ms**

