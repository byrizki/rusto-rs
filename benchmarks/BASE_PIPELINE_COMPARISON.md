# RustO Base Pipeline End-to-End Latency Benchmark

> **Benchmark Target**: `ktp.jpeg` (1600x993 pixels)
> **Model Architecture**: PP-OCRv6 Tiny (`det` ~1.8 MB, `rec` ~4.3 MB)
> **Execution Pipeline**: Native RustO base pipeline (DetPreProcess -> Det Inference -> DBPostProcess -> 4-point Perspective Crop -> RecPreProcess -> Rec Inference -> CTC Decode)
> **Iterations**: 2 warmup runs, 5 timed benchmark runs

---

## 1. Executive Summary & Total Pipeline Latency

| Inference Engine | Total Pipeline Latency (ms) | Min (ms) | Max (ms) | Neural Infer Only (ms) | Image & Postproc (ms) | Text Lines Detected |
|---|---|---|---|---|---|---|
| **EngineSession (Native rusto)** | **705.13 ms** | 668.13 ms | 752.47 ms | 520.98 ms | 184.16 ms | 27/16 lines |
| **Rust ort (ONNX)** | **447.83 ms** | 444.41 ms | 450.71 ms | 282.79 ms | 165.04 ms | 27/16 lines |
| **RTen (.onnx)** | **689.94 ms** | 654.44 ms | 745.16 ms | 496.37 ms | 193.57 ms | 27/16 lines |
| **RTen (.rten)** | **678.08 ms** | 592.31 ms | 777.92 ms | 481.39 ms | 196.68 ms | 27/16 lines |

---

## 2. Detailed Stage-by-Stage Breakdown (ms)

Every millisecond from raw image preprocessing til final transcription decoded:

| Pipeline Stage | Description | EngineSession (Native) | Rust ort | RTen (.onnx) | RTen (.rten) |
|---|---|---|---|---|---|
| **1. Det Preprocess** | Bounds resize, vertical pad, 736 min-side resize, ImageNet BGR norm | 98.13 ms | 89.89 ms | 99.44 ms | 102.40 ms |
| **2. Det Inference** | Neural detection model inference on `[1, 3, 736, 1184]` | 342.32 ms | 173.63 ms | 335.67 ms | 312.17 ms |
| **3. Det Postprocess** | DBNet binary thresholding (0.3), contour extraction, unclip (2.0), box sorting | 25.21 ms | 17.52 ms | 30.39 ms | 26.33 ms |
| **4. Crop & Unwarp** | 4-point perspective warp & 90° orientation rotation for all 16 boxes | 25.77 ms | 26.30 ms | 28.77 ms | 28.99 ms |
| **5. Rec Preprocess** | Aspect ratio sorting, height-48 scaling, zero padding, batch construction | 17.77 ms | 17.86 ms | 18.34 ms | 21.25 ms |
| **6. Rec Inference** | Neural text recognition across all 16 crops (batches of 6) | 178.65 ms | 109.17 ms | 160.70 ms | 169.22 ms |
| **7. CTC Decode** | Greedy token decoding & vocabulary mapping to UTF-8 characters | 17.26 ms | 13.45 ms | 16.63 ms | 17.70 ms |
| **TOTAL PIPELINE** | **Preprocessing Til Finished** | **705.13 ms** | **447.83 ms** | **689.94 ms** | **678.08 ms** |

---

## 3. Transcription Parity Verification on ktp.jpeg

| Line # | EngineSession (Native rusto) | Rust ort | RTen (.onnx) | RTen (.rten) |
|---|---|---|---|---|
| Line 01 | PROVINSIUKIJAKARTA (0.87) | PROVINSIUKIJAKARTA (0.87) | PROVINSIUKIJAKARTA (0.87) | PROVINSIUKIJAKARTA (0.87) |
| Line 02 | JAKARTAI (0.97) | JAKARTAI (0.97) | JAKARTAI (0.97) | JAKARTAI (0.97) |
| Line 03 | NIK (0.99) | NIK (0.99) | NIK (0.99) | NIK (0.99) |
| Line 04 | 3217141102760006 (0.91) | 3217141102760006 (0.91) | 3217141102760006 (0.91) | 3217141102760006 (0.91) |
| Line 05 | MHHAMADR1/EI (0.71) | MHHAMADR1/EI (0.71) | MHHAMADR1/EI (0.71) | MHHAMADR1/EI (0.71) |
| Line 06 | TompwtgLahw JAAAR1A11021006 (0.71) | TompwtgLahw JAAAR1A11021006 (0.71) | TompwtgLahw JAAAR1A11021006 (0.71) | TompwtgLahw JAAAR1A11021006 (0.71) |
| Line 07 | henis ketamnm (0.73) | henis ketamnm (0.73) | henis ketamnm (0.73) | henis ketamnm (0.73) |
| Line 08 | AKLARI (0.75) | AKLARI (0.75) | AKLARI (0.75) | AKLARI (0.75) |
| Line 09 | Gol Dsrah (0.79) | Gol Dsrah (0.79) | Gol Dsrah (0.79) | Gol Dsrah (0.79) |
| Line 10 | AMamat (0.79) | AMamat (0.79) | AMamat (0.79) | AMamat (0.79) |
| Line 11 | JL.RAWASAWAHn (0.84) | JL.RAWASAWAHn (0.84) | JL.RAWASAWAHn (0.84) | JL.RAWASAWAHn (0.84) |
| Line 12 | RT/RW (0.84) | RT/RW (0.84) | RT/RW (0.84) | RT/RW (0.84) |
| Line 13 | 006/006 (0.89) | 006/006 (0.89) | 006/006 (0.89) | 006/006 (0.89) |
| Line 14 | Kel/Desa (0.99) | Kel/Desa (0.99) | Kel/Desa (0.99) | Kel/Desa (0.99) |
| Line 15 | KAMPUNGRAWA (0.98) | KAMPUNGRAWA (0.98) | KAMPUNGRAWA (0.98) | KAMPUNGRAWA (0.98) |
| Line 16 | Kecamatan:JOHARBARJ (0.93) | Kecamatan:JOHARBARJ (0.93) | Kecamatan:JOHARBARJ (0.93) | Kecamatan:JOHARBARJ (0.93) |
| Line 17 | Agama (0.99) | Agama (0.99) | Agama (0.99) | Agama (0.99) |
| Line 18 | ISLAM (1.00) | ISLAM (1.00) | ISLAM (1.00) | ISLAM (1.00) |
| Line 19 | StatusPerkawinan KAWIN (0.93) | StatusPerkawinan KAWIN (0.93) | StatusPerkawinan KAWIN (0.93) | StatusPerkawinan KAWIN (0.93) |
| Line 20 | Pekerjaan (0.94) | Pekerjaan (0.94) | Pekerjaan (0.94) | Pekerjaan (0.94) |
| Line 21 | KARYAWANSWASTA (1.00) | KARYAWANSWASTA (1.00) | KARYAWANSWASTA (1.00) | KARYAWANSWASTA (1.00) |
| Line 22 | JAKARTA PUSAT (0.93) | JAKARTA PUSAT (0.93) | JAKARTA PUSAT (0.93) | JAKARTA PUSAT (0.93) |
| Line 23 | Kowarganegatasn WNl (0.90) | Kowarganegatasn WNl (0.90) | Kowarganegatasn WNl (0.90) | Kowarganegatasn WNl (0.90) |
| Line 24 | 29-10-2024 (0.98) | 29-10-2024 (0.98) | 29-10-2024 (0.98) | 29-10-2024 (0.98) |
| Line 25 | BerlakuHingga (0.96) | BerlakuHingga (0.96) | BerlakuHingga (0.96) | BerlakuHingga (0.96) |
| Line 26 | SEUMURHIDUP (0.97) | SEUMURHIDUP (0.97) | SEUMURHIDUP (0.97) | SEUMURHIDUP (0.97) |
| Line 27 | PY (0.56) | PY (0.56) | PY (0.56) | PY (0.56) |

