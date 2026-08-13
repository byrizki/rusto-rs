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

Add RustO! to your `Cargo.toml`:

```toml
[dependencies]
rusto-rs = "0.1"
```

```rust
use rusto::{RustO, RustOConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize with PP-OCRv6 preset
    let config = RustOConfig::ppv6("models/det.mnn", "models/rec.mnn", "models/dict.txt")
        .with_text_score(0.5)
        .with_xy_threshold(0.5, 1.0); // Configure spatial text spacing
    
    let mut ocr = RustO::new(config)?;
    let output = ocr.run("document.jpg")?;
    
    // 1. Structured text results with bounding boxes & frames
    for res in output.to_text_results() {
        println!("Text: '{}' (Confidence: {:.2})", res.text, res.score);
        println!("  Frame: [left={:.1}, top={:.1}, w={:.1}, h={:.1}]", 
            res.frame.left, res.frame.top, res.frame.width, res.frame.height);
    }
    
    // 2. Spatial layout text (visual document representation)
    let spatial_text = output.to_spatial_text(None, None);
    println!("Spatial Document:\n{}", spatial_text);
    
    Ok(())
}
```

---

### 2. .NET / C#

Install the core runtime and your preferred model package:

```bash
# Core managed runtime + cross-platform desktop native runtimes
dotnet add package RustODotnet

# Choose an OCR model tier (models copy automatically to output models/ dir):
dotnet add package RustODotnet.Models.PPOCRv6.Tiny    # ~6 MB (Recommended default)
# or dotnet add package RustODotnet.Models.PPOCRv6.Small   # ~30 MB
# or dotnet add package RustODotnet.Models.PPOCRv6.Medium  # ~134 MB
# or dotnet add package RustODotnet.Models.PPOCRv5.Mobile  # ~28 MB
# or dotnet add package RustODotnet.Models.PPOCRv4.Mobile  # ~23 MB
```

```csharp
using System;
using RustODotnet;

// 1. Basic OCR (automatically discovers models in models/ folder)
using var ocr = new RustO();
var results = ocr.RecognizeFile("invoice.jpg");

foreach (var res in results)
{
    Console.WriteLine($"Text: '{res.Text}' (Confidence: {res.Score:P1})");
    Console.WriteLine($"  Frame: X={res.Frame.Left}, Y={res.Frame.Top}, W={res.Frame.Width}, H={res.Frame.Height}");
}

// 2. Spatial layout formatted output (preserves columns, tables, paragraphs)
string spatialText = ocr.RecognizeFileToSpatialText("invoice.jpg");
Console.WriteLine(spatialText);
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
    implementation 'com.github.byrizki.rusto-rs:rusto-models-ppocrv6-tiny:v0.2.0'
}
```

**JavaScript / TypeScript Usage:**
```typescript
import { initialize, detectText, detectTextToSpatialText } from 'react-native-rusto';

// Initialize with bundled default models (no parameters needed!)
await initialize();

// Detect text with bounding frames
const results = await detectText('/path/to/image.jpg');
results.forEach((r) => {
  console.log(`${r.text} (${r.score}) - Frame:`, r.frame);
});

// Or extract visual spatial layout text
const spatialText = await detectTextToSpatialText('/path/to/image.jpg', 0.5, 1.0);
console.log(spatialText);
```

---

### 4. iOS (Swift)

Add to your `Podfile`:

```ruby
target 'YourApp' do
  pod 'RustO'
  pod 'RustO-Models-PPOCRv6-Tiny' # Pre-packaged models
end
```

```swift
import RustO

// Initialize with automatic model discovery
let ocr = try RustO()

let results = try ocr.recognizeFile("receipt.jpg")
for res in results {
    print("\(res.text) (\(res.score)): frame=\(res.frame.left),\(res.frame.top),\(res.frame.width)x\(res.frame.height)")
}

// Spatial formatted layout output
let spatialText = try ocr.recognizeFileToSpatialText("receipt.jpg")
print(spatialText)
```

---

### 5. Android (Kotlin)

Add JitPack repository to `settings.gradle`:

```groovy
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
        maven { url 'https://jitpack.io' }
    }
}
```

Add dependencies to `app/build.gradle`:

```groovy
dependencies {
    implementation 'com.github.byrizki.rusto-rs:rusto-android:v0.2.0'
    implementation 'com.github.byrizki.rusto-rs:rusto-models-ppocrv6-tiny:v0.2.0'
}
```

```kotlin
import com.byrizki.rusto.RustO

// Initialize engine from Android context assets
val ocr = RustO.create(context)

val results = ocr.recognizeFile("/path/to/image.jpg")
for (res in results) {
    println("${res.text} (score: ${res.score}) at [${res.frame.left}, ${res.frame.top}]")
}

val spatialText = ocr.recognizeFileToSpatialText("/path/to/image.jpg")
println(spatialText)
```

---

### 6. Command Line Interface (CLI)

```bash
# JSON output (default)
cargo run --release -- --det-model det.mnn --rec-model rec.mnn --dict dict.txt image.jpg

# Spatial formatted text output
cargo run --release -- --det-model det.mnn --rec-model rec.mnn --dict dict.txt --format spatial image.jpg

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

## ⚙️ Configuration Reference (`RustOConfig`)

`RustOConfig` provides granular control over the OCR pipeline:

```rust
use rusto::RustOConfig;

let config = RustOConfig::ppv6("models/det.mnn", "models/rec.mnn", "models/dict.txt")
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

- `RustOConfig::ppv6(...)` — Pre-configured for PP-OCRv6 (`limit_side_len=736`, `limit_type="min"`, `unclip_ratio=2.0`)
- `RustOConfig::ppv5(...)` — Pre-configured for PP-OCRv5 (`limit_side_len=736`, `limit_type="min"`, `unclip_ratio=2.0`)
- `RustOConfig::ppv4(...)` — Pre-configured for PP-OCRv4 (`limit_side_len=960`, `limit_type="max"`, `unclip_ratio=1.5`)
- `RustOConfig::ppv3(...)` — Pre-configured for PP-OCRv3 (`limit_side_len=960`, `limit_type="max"`, `unclip_ratio=1.5`)

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
│   ├── config.rs               # RustOConfig & template presets (PPV6, PPV5, PPV4, PPV3)
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
