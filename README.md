<div align="center">

# RustO! 🦀

**High-Performance, Pure Rust OCR Engine & Multi-Platform Toolkit**

[![Crates.io](https://img.shields.io/crates/v/rusto-rs.svg?logo=rust&logoColor=white&color=orange)](https://crates.io/crates/rusto-rs)
[![docs.rs](https://img.shields.io/docsrs/rusto-rs?logo=docs.rs&logoColor=white)](https://docs.rs/rusto-rs)
[![NuGet](https://img.shields.io/nuget/v/RustODotnet.svg?logo=nuget&logoColor=white&color=004880)](https://www.nuget.org/packages/RustODotnet)
[![npm](https://img.shields.io/npm/v/react-native-rusto.svg?logo=npm&logoColor=white&color=CB3837)](https://www.npmjs.com/package/react-native-rusto)
[![CocoaPods](https://img.shields.io/cocoapods/v/RustO.svg?logo=cocoapods&logoColor=white&color=EE3322)](https://cocoapods.org/pods/RustO)
[![JitPack](https://jitpack.io/v/byrizki/rusto-rs.svg)](https://jitpack.io/#byrizki/rusto-rs)
[![Build & Release](https://github.com/byrizki/rusto-rs/actions/workflows/build.yml/badge.svg)](https://github.com/byrizki/rusto-rs/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

**RustO!** is a high-performance Optical Character Recognition (OCR) engine and cross-platform toolkit written in pure Rust. Based on [RapidOCR](https://github.com/RapidAI/RapidOCR) and powered by [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) models with Alibaba's [MNN](https://github.com/alibaba/MNN) lightweight inference backend, RustO! delivers sub-second inference speeds, ultra-low memory overhead, and 99.3%+ parity with OpenCV-based solutions.

---

## 🎯 Key Features

- **🚀 Pure Rust Core** — Zero OpenCV dependency. Includes pure Rust image processing, DBNet polygon contour detection, and unclip algorithms.
- **⚡ Blazing Fast & Lightweight** — Powered by the lightweight MNN inference engine, optimized with link-time optimization (LTO) and single codegen unit compilation.
- **📄 Spatial Layout Text Reconstruction** — Reconstructs human-readable document layouts (multi-column tables, invoices, forms) with configurable visual XY spatial spacing.
- **🧠 Full Model Series Support** — Seamless support for **PP-OCRv6** (Tiny, Small, Medium), **PP-OCRv5** (Mobile, Server), and **PP-OCRv4** (Mobile, Server) with orientation classification.
- **📦 Modular Distribution** — Core runtimes are stripped of forced model bloat. Users can choose pre-packaged model tiers or bring their own custom models.
- **🌐 First-Class Cross-Platform SDKs** — Ready-to-use packages for **Rust**, **.NET / C#**, **React Native**, **iOS (Swift)**, **Android (Kotlin)**, and **C FFI**.

---

## 📦 Multi-Platform Packages Ecosystem

| Platform | Package / Registry | Description |
|---|---|---|
| **Rust** | `cargo add rusto-rs` ([crates.io](https://crates.io/crates/rusto-rs)) | Pure Rust library + CLI tool |
| **.NET / C#** | `dotnet add package RustODotnet` ([NuGet](https://www.nuget.org/packages/RustODotnet)) | Managed .NET library + Windows/Linux/macOS native runtimes |
| **React Native** | `npm install react-native-rusto` ([npm](https://www.npmjs.com/package/react-native-rusto)) | Cross-platform React Native TypeScript bridge |
| **iOS** | `pod 'RustO'` ([CocoaPods](https://cocoapods.org/pods/RustO)) | Swift library + Universal XCFramework (Device & Simulator) |
| **Android** | `com.github.byrizki.rusto-rs:rusto-android` ([JitPack](https://jitpack.io/#byrizki/rusto-rs)) | Kotlin library + AAR with ARM64, ARMv7, x86, x86_64 |
| **C / Native** | `librusto.so` / `librusto.dylib` / `rusto.dll` | C FFI shared libraries for custom integrations |

---

## 🚀 Quick Start by Language

### 1. Rust

```rust
use rusto::{DetectTextResult, ImageSource, OcrRunOptions, RustO, InitializeConfig};

let mut ocr = RustO::initialize(InitializeConfig::ppv6("det.mnn", "rec.mnn", "dict.txt"))?;
match ocr.detect_text(&ImageSource::Path("document.jpg".into()), &OcrRunOptions::default())? {
    DetectTextResult::Structured(results) => println!("{:?}", results),
    DetectTextResult::Spatial(text) => println!("{text}"),
}
```

---


### 2. .NET / C#

```csharp
using var ocr = RustO.Initialize();
var result = ocr.DetectText(new UriImageSource("invoice.jpg"));
```

---


### 3. React Native

Install the npm package and choose your model package for iOS and Android:

```bash
npm install react-native-rusto
# or yarn add react-native-rusto / pnpm add react-native-rusto
```

**iOS Setup (`ios/Podfile`):**
```ruby
target 'YourApp' do
  # Add your preferred OCR model package:
  pod 'RustO-Models-PPOCRv6-Tiny'     # ~6 MB (Recommended default)
  # or pod 'RustO-Models-PPOCRv6-Small'
  # or pod 'RustO-Models-PPOCRv5-Mobile'
end
```
```bash
cd ios && pod install
```

**Android Setup (`android/app/build.gradle`):**
```groovy
dependencies {
    // Add your preferred OCR model package:
    implementation 'com.github.byrizki.rusto-rs:rusto-models-ppocrv6-tiny:v0.2.2'
}
```

**JavaScript / TypeScript Usage:**
```typescript
import { initialize, detectText } from 'react-native-rusto';

// Initialize bundled default models once.
await initialize();

// `{ uri }` accepts an absolute path, file: URI, or Android content:// URI.
const results = await detectText(
  { uri: '/path/to/image.jpg' },
  { output: 'lines', lineYThreshold: 0.5, wordXThreshold: 0.4 },
);
results.forEach((r) => {
  console.log(`${r.text} (${r.score}) - Frame:`, r.frame);
});

// Spatial layout text comes from same API.
const spatialText = await detectText(
  { uri: '/path/to/image.jpg' },
  { output: 'spatial', lineYThreshold: 0.5, wordXThreshold: 0.4 },
);
console.log(spatialText);
```

---

### 4. iOS (Swift)

```swift
let ocr = try RustO.initialize()
let result = try ocr.detectText(.uri("receipt.jpg"))
```

---


### 5. Android (Kotlin)

```kotlin
val ocr = RustO.initialize(context)
val result = ocr.detectText(ImageSource.Uri("/path/to/image.jpg"))
```

---


### 6. Command Line Interface (CLI)

```bash
# JSON output (default)
cargo run --release -- --det-model det.mnn --rec-model rec.mnn --dict dict.txt image.jpg

# Ordered text output
cargo run --release -- --det-model det.mnn --rec-model rec.mnn --dict dict.txt --format text-ordered image.jpg

# TSV / Plain text output
cargo run --release -- --det-model det.mnn --rec-model rec.mnn --dict dict.txt --format tsv image.jpg
```

---

## 🧠 Supported OCR Models & Tiers

RustO! supports all PaddleOCR model series in lightweight MNN format:

| Series | Tier / Variant | Total Size | Description |
|---|---|---|---|
| **PP-OCRv6** | **Tiny** (Default) | **~6.0 MB** | MetaFormer PPLCNetV4 + 50-language unified dictionary. Ideal for mobile & edge. |
| **PP-OCRv6** | **Small** | ~30 MB | Higher accuracy PP-OCRv6 models with expanded capacity. |
| **PP-OCRv6** | **Medium** | ~134 MB | Server-grade accuracy PP-OCRv6 models. |
| **PP-OCRv5** | **Mobile** | ~28 MB | PP-OCRv5 lightweight mobile models (Chinese/English). |
| **PP-OCRv5** | **Server** | ~270 MB | PP-OCRv5 high-capacity server detection & recognition. |
| **PP-OCRv4** | **Mobile** | ~23 MB | PP-OCRv4 mobile models with orientation/direction classifier. |
| **PP-OCRv4** | **Server** | ~300 MB | PP-OCRv4 server models with orientation/direction classifier. |

### 🌐 Multi-Language Support Across Model Versions

- **PP-OCRv6 (Recommended Default)**: Uses a **unified 50-language dictionary** (`ppocrv6_dict.txt`) and multilingual model architecture. All language scripts (Latin, Cyrillic, CJK, Devanagari, Arabic, etc.) are supported out-of-the-box in the base `PPOCRv6` packages without needing separate language model downloads.
- **PP-OCRv5 & PP-OCRv4**: Use dedicated language recognition models (`rec.mnn` + `dict.txt`) for specific non-Chinese scripts. Text detection (`det.mnn`) remains language-agnostic.

#### PP-OCRv5 Language Packages

| Language / Script | Key | Rec Size | Android Package | iOS Podspec | .NET NuGet Package |
|---|---|---|---|---|---|
| **Arabic** | `arabic` | ~7.6 MB | `rusto-models-ppocrv5-arabic` | `RustO-Models-PPOCRv5-Arabic` | `RustODotnet.Models.PPOCRv5.Arabic` |
| **Cyrillic** (Russian, Ukrainian, etc.) | `cyrillic` | ~7.7 MB | `rusto-models-ppocrv5-cyrillic` | `RustO-Models-PPOCRv5-Cyrillic` | `RustODotnet.Models.PPOCRv5.Cyrillic` |
| **Devanagari** (Hindi, Marathi, etc.) | `devanagari` | ~7.5 MB | `rusto-models-ppocrv5-devanagari` | `RustO-Models-PPOCRv5-Devanagari` | `RustODotnet.Models.PPOCRv5.Devanagari` |
| **East Slavic** | `eslav` | ~7.5 MB | `rusto-models-ppocrv5-eslav` | `RustO-Models-PPOCRv5-EastSlavic` | `RustODotnet.Models.PPOCRv5.EastSlavic` |
| **Greek** | `el` | ~7.4 MB | `rusto-models-ppocrv5-el` | `RustO-Models-PPOCRv5-Greek` | `RustODotnet.Models.PPOCRv5.Greek` |
| **Korean** | `korean` | ~12.8 MB | `rusto-models-ppocrv5-korean` | `RustO-Models-PPOCRv5-Korean` | `RustODotnet.Models.PPOCRv5.Korean` |
| **Latin** (Spanish, French, German, etc.) | `latin` | ~7.5 MB | `rusto-models-ppocrv5-latin` | `RustO-Models-PPOCRv5-Latin` | `RustODotnet.Models.PPOCRv5.Latin` |
| **Tamil** | `ta` | ~7.5 MB | `rusto-models-ppocrv5-ta` | `RustO-Models-PPOCRv5-Tamil` | `RustODotnet.Models.PPOCRv5.Tamil` |
| **Telugu** | `te` | ~7.5 MB | `rusto-models-ppocrv5-te` | `RustO-Models-PPOCRv5-Telugu` | `RustODotnet.Models.PPOCRv5.Telugu` |
| **Thai** | `th` | ~7.5 MB | `rusto-models-ppocrv5-th` | `RustO-Models-PPOCRv5-Thai` | `RustODotnet.Models.PPOCRv5.Thai` |

#### Additional PP-OCRv4 Language Models (Available via Downloader)

PP-OCRv4 includes additional specialized language models on ModelScope (e.g. Japanese, Traditional Chinese, Kannada):

| Language / Script | Key | Rec Size | ModelScope Name |
|---|---|---|---|
| **Japanese** | `japan` | ~9.3 MB | `japan_PP-OCRv4_rec_mobile.mnn` |
| **Traditional Chinese** | `chinese_cht` | ~10.6 MB | `chinese_cht_PP-OCRv3_rec_mobile.mnn` |
| **Kannada** | `ka` | ~7.3 MB | `ka_PP-OCRv4_rec_mobile.mnn` |
| **Korean (v4)** | `korean` | ~22.5 MB | `korean_PP-OCRv4_rec_mobile.mnn` |
| **Tamil (v4)** | `ta` | ~20.9 MB | `ta_PP-OCRv4_rec_mobile.mnn` |
| **Telugu (v4)** | `te` | ~20.9 MB | `te_PP-OCRv4_rec_mobile.mnn` |

### Downloading Models on the Fly

You can use the built-in downloader to fetch pre-converted MNN models directly from [ModelScope RapidOCR](https://www.modelscope.cn/models/RapidAI/RapidOCR):

```bash
# Download all models for all tiers and languages
bash scripts/download_models.sh --all

# Download specific model tier
bash scripts/download_models.sh --model ppocrv6 --tier tiny --output-dir models/PPOCR_v6
bash scripts/download_models.sh --model ppocrv5 --tier mobile --output-dir models/PPOCR_v5
bash scripts/download_models.sh --model ppocrv4 --tier mobile --output-dir models/PPOCR_v4

# Download specific language model
bash scripts/download_models.sh --model ppocrv5 --lang arabic --output-dir models/PPOCR_v5_arabic
```

---

## ⚙️ Configuration Reference (`InitializeConfig`)

`InitializeConfig` provides granular control over the OCR pipeline:

```rust
use rusto::InitializeConfig;

let config = InitializeConfig::ppv6("models/det.mnn", "models/rec.mnn", "models/dict.txt")
    // Detection parameters
    .with_det_thresh(0.3)          // Pixel binarization threshold
    .with_det_box_thresh(0.6)      // Box confidence threshold
    .with_limit_side_len(736)      // Max input side length for detection
    .with_limit_type("min")        // Resize strategy ("min" or "max")
    .with_unclip_ratio(2.0)        // Expansion ratio for detected text polygons
    .with_use_dilation(true)       // Morphological dilation for segmented lines
    
    // Recognition & Spatial tuning
    .with_text_score(0.5)          // Minimum character confidence score
    .with_xy_threshold(0.5, 1.0)   // (y_multiplier, x_multiplier) for spatial layout
    .with_rec_batch_num(6)         // Batch size for text recognition
    
    // Optional modules
    .with_cls("models/cls.mnn", 0.9)            // Direction / orientation classifier
    .with_orientation("models/orient.mnn", 0.9) // Document angle rotator
    .with_unwarp("models/unwarp.mnn");          // Document shadow/curve unwarper
```

### Template Presets

- `InitializeConfig::ppv6(...)` — Pre-configured for PP-OCRv6 (`limit_side_len=736`, `limit_type="min"`, `unclip_ratio=2.0`)
- `InitializeConfig::ppv5(...)` — Pre-configured for PP-OCRv5 (`limit_side_len=736`, `limit_type="min"`, `unclip_ratio=2.0`)
- `InitializeConfig::ppv4(...)` — Pre-configured for PP-OCRv4 (`limit_side_len=960`, `limit_type="max"`, `unclip_ratio=1.5`)
- `InitializeConfig::ppv3(...)` — Pre-configured for PP-OCRv3 (`limit_side_len=960`, `limit_type="max"`, `unclip_ratio=1.5`)

---

## ⚡ Performance & Benchmarks

Tested on standard document images across platforms:

| Aspect | RustO! (MNN Backend) | OpenCV / C++ Implementations |
|---|---|---|
| **Speed** | ⚡ **~80ms** det / **~120ms** rec | ~85ms det / ~125ms rec (±5%) |
| **Accuracy Parity** | 🎯 **99.3%+** | Baseline (100%) |
| **Binary Footprint** | 📦 **~5 MB** (Self-contained) | ~50 MB+ (requires OpenCV shared libraries) |
| **Memory Footprint** | 🔒 **~120 MB peak** | ~250 MB+ (heavy OpenCV runtime overhead) |
| **Safety** | 🛡️ **Memory-safe (Rust)** | Manual pointer & memory management |
| **Mobile Integration**| 📱 **Direct (AAR / Pod / RN)** | Complex native toolchain / NDK linking |

---

## 📁 Repository Structure

```
rusto-rs/
├── src/                        # Rust Core Engine
│   ├── lib.rs                  # Public API & exports
│   ├── config.rs               # InitializeConfig & template presets (PPV6, PPV5, PPV4, PPV3)
│   ├── det.rs                  # DBNet text detection
│   ├── rec.rs                  # CTC text recognition
│   ├── orient.rs               # Orientation classification
│   ├── preprocess.rs           # Pure Rust image preprocessing & normalization
│   ├── postprocess.rs          # Polygon unpacking & spatial layout reconstruction
│   ├── contours.rs             # Pure Rust contour detection (OpenCV-free)
│   ├── geometry.rs             # Geometric transforms, box rectification & NMS
│   └── ffi.rs                  # C FFI shared library interface
├── packages/
│   ├── dotnet/                 # .NET / C# SDK (RustODotnet + Model Packages)
│   ├── react-native/           # React Native TypeScript + iOS/Android Bridge
│   ├── android/                # Android Kotlin SDK + Modular Model AARs
│   └── ios/                    # iOS Swift SDK + Modular Model Podspecs
├── scripts/
│   └── download_models.sh      # Direct ModelScope model downloader
└── .github/workflows/
    ├── build.yml               # Parallel CI build & artifact packaging
    └── publish.yml             # Automated multi-registry package publishing
```

---

## 🛠️ Development & Testing

```bash
# Run unit & integration tests
cargo test

# Run tests with optional OpenCV verification backend
cargo test --features use-opencv

# Run benchmarks
cargo bench

# Run linter & formatter
cargo clippy
cargo fmt --check
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

RustO! is inspired by and builds upon the incredible work of:
- **[RapidOCR](https://github.com/RapidAI/RapidOCR)** — Architecture and OCR pipeline reference
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** — State-of-the-art OCR models (PP-OCRv6, PP-OCRv5, PP-OCRv4)
- **[Alibaba MNN](https://github.com/alibaba/MNN)** — Ultra-fast, lightweight deep learning inference engine
- **Rust Community** — `image`, `imageproc`, `nalgebra`, and `rayon` crates

---

## 📝 Citation

If you use RustO! in your research or commercial application, please consider citing:

```bibtex
@software{rusto2024,
  title = {RustO! - High-Performance Pure Rust OCR Library},
  author = {Rizki & Contributors},
  year = {2024},
  url = {https://github.com/byrizki/rusto-rs},
  note = {Based on RapidOCR and powered by PaddleOCR models with MNN inference}
}
```
