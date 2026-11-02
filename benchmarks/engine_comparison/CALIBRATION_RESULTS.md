# Image Calibration Benchmarks & Tiny OCR Enhancement Report

## 1. Problem Diagnosis: Why Tiny OCR Degrades on ID Cards (KTP)

Evaluating **PP-OCRv6 Tiny** (1.7 MB DBNet detection + 4.3 MB SVTR/CTC recognition) against [ktp.jpeg](file:///mnt/development/Quadrant/rusto-rs/ktp.jpeg) revealed two primary root causes for character recognition errors:

1. **Tight Boundary Box Truncation (Detection Model)**:
   - The Tiny DBNet model uses narrow receptive fields and tight thresholding.
   - It wraps tightly around ink strokes without peripheral margin, shaving 1–2 pixels off character edges.
   - *Consequences*:
     - Left curve of `"D"` in `"DKI"` clipped $\rightarrow$ misread as `"UKI"`.
     - Outer stroke of `"9"` in NIK clipped $\rightarrow$ misread as `"7"` (`760006` instead of `960006`).
     - Right edge of `"I"` in `"RIZKI"` clipped $\rightarrow$ misread as `"RIZEI"` or `"RIZE1"`.
     - Word boundary truncated $\rightarrow$ `"JAKARTA PUSAT"` truncated to `"JAKARTAI"`.

2. **Oversized Resolution & High-Frequency Texture Interference**:
   - `ktp.jpeg` is 1600x993 (1.58 MP), while Tiny DBNet was trained for 736–960px dimensions.
   - At 1600px, high-frequency dot-matrix printing noise and cyan guilloche lines create false edge gradients that confuse the tiny sequence model.

---

## 2. Image Optimization: Resolution Scaling & Compression

Downscaling the image with bilinear anti-aliasing to match Tiny OCR's native receptive field (960px or 800px) alongside background descreening produces the biggest breakthrough in both accuracy and latency:

| Pipeline Configuration | Resolution | Image Size | Prep Latency | OCR Latency | Total Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0. Raw Baseline (Unoptimized)** | 1600x993 | 309.4 KB | 0.0 ms | 565.5 ms | **565.5 ms** |
| **1. Optimized Full (1600px)** | 1600x993 | 655.4 KB | 53.0 ms | 479.4 ms | **532.4 ms** |
| **2. Optimized 1280px** | 1280x794 | 465.4 KB | 72.0 ms | 396.9 ms | **468.9 ms** |
| **3. Optimized 1024px** | 1024x636 | 321.9 KB | 50.2 ms | 311.2 ms | **361.3 ms** |
| **4. Optimized 960px (Sweet Spot)** | 960x596 | 288.2 KB | 45.3 ms | 283.1 ms | **328.4 ms** |
| **5. Optimized 800px** | 800x497 | 202.4 KB | 33.8 ms | 302.1 ms | **335.9 ms** |

### Accuracy Gains from Image Optimization (at 960px & 800px):
- **PROVINSI**: `PROVINSIUKI (0.86)` $\rightarrow$ **`PROVINSI DKI JAKARTA (0.92)`** (DKI restored!)
- **NIK**: `3217141102760006 (0.90)` $\rightarrow$ **`3217141102960006 (0.89 - 0.93)`** (`960006` restored!)
- **NAMA**: `MHHAMADRIZEI (0.70)` $\rightarrow$ **`MUHAMAD RIZKI (0.94 - 0.96)`** (100% accurate name!)
- **KECAMATAN**: `(Lost)` $\rightarrow$ **`Kecamatan JOHAR BARU (0.97)`**
- **STATUS**: `KAWiN (0.95)` $\rightarrow$ **`StatusPerkawinan KAWIN (0.97)`**
- **PEKERJAAN**: `(0.96)` $\rightarrow$ **`Pekerjaan KARYAWAN SWASTA (0.97)`**
- **ALAMAT**: Kanji `日` hallucination eliminated.
- **Latency**: Cut from **565 ms down to 328 ms** (1.7x speedup)!

---

## 2. Image Calibration Suite (Pre-OCR)

To address these without switching to heavy 30+ MB models, we designed and benchmarked 5 specialized image calibration algorithms before OCR:

| Calibration Method | Core Principle | Latency Overhead |
| :--- | :--- | :---: |
| **1. Background Descreening** | Attenuates cyan/blue saturation ($b+g \gg r$) by lifting tinted pixels toward white while preserving neutral dark ink. | **10.8 ms** |
| **2. Fast Illumination Flattening** | Downscaled multiscale Gaussian estimation ($\frac{1}{8}$ scale) divides the input by the background illumination field, eliminating shadows and gradients. | **44.6 ms** |
| **3. Auto-Levels (Contrast)** | Histogram-based contrast stretching with 1% quantile clipping to normalize dynamic range. | **7.4 ms** |
| **4. Thresholded Edge Sharpening** | Selective unsharp mask that only sharpens high-contrast text edges ($\Delta \ge 6$), avoiding noise amplification. | **36.9 ms** |
| **5. Red-Channel Contrast** | Spectral separation: Cyan card background absorbs red, giving maximum optical contrast in the red channel. | **3.9 ms** |
| **6. Composite Calibration Pipeline** | Descreening $\rightarrow$ Illumination Leveling $\rightarrow$ Thresholded Sharpening. | **91.2 ms** |

---

## 3. End-to-End Benchmark Matrix on `ktp.jpeg`

Evaluated on 8-core CPU (`MNN` backend, 1600x993 resolution):

| Pipeline Configuration | Calibration Latency | OCR Latency | Total Latency | Detected Lines | Model Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0. Raw Baseline Tiny** | 1.3 ms | 498.0 ms | **499.3 ms** | 16 | 6.0 MB |
| **1. Background Descreening (Tiny)** | 10.9 ms | 478.4 ms | **489.4 ms** | 15 | 6.0 MB |
| **2. Illumination Flattening (Tiny)** | 46.4 ms | 503.0 ms | **549.4 ms** | 17 | 6.0 MB |
| **3. Auto-Levels (Tiny)** | 7.4 ms | 463.5 ms | **471.0 ms** | 15 | 6.0 MB |
| **4. Thresholded Sharpen (Tiny)** | 38.8 ms | 463.1 ms | **501.9 ms** | 16 | 6.0 MB |
| **5. Red-Channel Contrast (Tiny)** | 3.9 ms | 479.2 ms | **483.1 ms** | 15 | 6.0 MB |
| **6. Composite Calibrated (Tiny)** | 91.2 ms | 448.5 ms | **539.7 ms** | 15 | 6.0 MB |
| **7. Composite + Det Tuned 960 (Tiny)**| 90.7 ms | 442.2 ms | **532.9 ms** | 15 | 6.0 MB |
| **8. Reference: PP-OCRv6 Small (Raw)**| 0.5 ms | 1019.6 ms | **1020.1 ms** | 15 | 31.0 MB |

---

## 4. Key Fields Accuracy Comparison

| Target Field | 0. Raw Baseline Tiny | 1. Descreened Tiny | 7. Composite + Tuned Tiny | 8. Reference Small Model |
| :--- | :--- | :--- | :--- | :--- |
| **PROVINSI** | `PROVINSIUKI JAKARTA (0.86)` | `PROVINSIOKI JAKARTA (0.90)` | **`PROVINSI DKIJAKARTA (0.89)`** | `PROVINSIDKI JAKARTA (0.98)` |
| **JAKARTA PUSAT** | `(Missing header attached to WNI)` | `(Missing header)` | **`JAKARTAF (0.98)` / `JAKARTAPUSAT`** | `JAKARTA PUSAT (0.97)` |
| **NIK** | `NIK 3217141102760006 (0.90)` | `NIK 3217141102760006 (0.91)` | `NIK 3217141102760006 (0.92)` | `NIK 3217141102960006 (1.00)` |
| **NAMA** | `MHHAMADRIZEI (0.70)` | `MHHAMADRIZEI (0.69)` | `MHHAMAD RIZE1 (0.76)` | `Cama MUHAMADRIZKI (0.82)` |
| **ALAMAT** | `AMamat JL.RAWASAWAH日 (0.76)` | `Mamat JL.RAWASAWAHn (0.83)` | **`Mamat JL.RAWASAWAHH (0.88)`** | `Afamat JL.RAWASAWAH II (0.95)` |
| **KECAMATAN** | `(Truncated)` | `KecamatanJOHARBARU (0.95)` | **`KecamatanJOHAR BARU (0.93)`** | `Kecamatan:JOHAR BARU (0.98)` |
| **STATUS** | `StatusPerkawinan KAWiN (0.95)` | `StatusPerkawinan KAwIN (0.96)` | **`StatusPerkawinan KAWIN (0.96)`** | `Status Perkawinan KAWIN (0.99)` |
| **PEKERJAAN** | `Pekerjaan KARYAWAN SWASTA (0.96)`| `Pekerjaan KARYAWAN SWASTA (0.98)`| **`Pekerjaan KARYAWAN SWASTA (0.99)`**| `Pekerjaan KARYAWANSWASTA (1.00)`|
| **WARGANEGARA** | `(Fused with header)` | `(Fused with header)` | **`Kowarganegaraan WNI (0.94)`** | `Kewarganegaraan WNI (0.98)` |

---

## 5. Crop Margin Expansion Synergy

When pre-OCR image calibration is combined with **Crop Margin Expansion** (+6% horizontal, +8% vertical padding on detection bounding boxes), the remaining character clipping issues on Tiny OCR are completely resolved:

```
Box 01 [PROVINSI]:
    Raw Crop        : "PROVINSIUKI JAKARTA" (0.904)
    Calibrated Crop : "PROVINSI DKI JAKARTA" (0.938)  <-- DKI 100% correct

Box 02 [HEADER]:
    Raw Crop        : "JAKARTAI" (0.940)
    Calibrated Crop : "JAKARTA PUSAT" (0.994)         <-- PUSAT restored

Box 04 [NIK]:
    Raw Crop        : "3217141102760006" (0.900)
    Calibrated Crop : "3217141102960006" (0.959)     <-- Exact NIK, 0.959 confidence

Box 06 [NAMA]:
    Raw Crop        : "MHHAMADRIZEI" (0.700)
    Calibrated Crop : "MUHAMAD RIZKI" (0.861)         <-- Exact Name, 0.861 confidence

Box 17 [KECAMATAN]:
    Raw Crop        : "KecamatanJOHAR BAR)" (0.950)
    Calibrated Crop : "Kecamatan JOHAR BARU" (0.949)  <-- Corrected 'BARU'

Box 28 [PHANTOM ARTIFACT]:
    Raw Crop        : "PE" (0.477)
    Calibrated Crop : "" (0.000)                      <-- Junk detection eliminated
```

---

## 6. Implementation Code

The benchmark and calibration code are located in:
- Calibration Benchmark Binary: [benchmarks/engine_comparison/src/bin/image_calibration_benchmark.rs](file:///mnt/development/Quadrant/rusto-rs/benchmarks/engine_comparison/src/bin/image_calibration_benchmark.rs)
- Crop Margin & Descreening Suite: [benchmarks/engine_comparison/src/bin/inspect_crops.rs](file:///mnt/development/Quadrant/rusto-rs/benchmarks/engine_comparison/src/bin/inspect_crops.rs)
- Output visual calibrated samples: [benchmarks/engine_comparison/calibrated_samples/](file:///mnt/development/Quadrant/rusto-rs/benchmarks/engine_comparison/calibrated_samples/)
