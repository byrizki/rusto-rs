<div align="center">

# RustO! 🦀

**High-Performance, Pure Rust OCR Engine & Multi-Platform Toolkit**

[![Crates.io](https://img.shields.io/crates/v/rusto-rs.svg?logo=rust&logoColor=white&color=orange)](https://crates.io/crates/rusto-rs)
[![docs.rs](https://img.shields.io/docsrs/rusto-rs?logo=docs.rs&logoColor=white)](https://docs.rs/rusto-rs)
[![NuGet](https://img.shields.io/nuget/v/RustODotnet.svg?logo=nuget&logoColor=white&color=004880)](https://www.nuget.org/packages/RustODotnet)
[![npm](https://img.shields.io/npm/v/@rustors/react-native.svg?logo=npm&logoColor=white&color=CB3837)](https://www.npmjs.com/package/@rustors/react-native)
[![CocoaPods](https://img.shields.io/cocoapods/v/RustO.svg?logo=cocoapods&logoColor=white&color=EE3322)](https://cocoapods.org/pods/RustO)
[![JitPack](https://jitpack.io/v/byrizki/rusto-rs.svg)](https://jitpack.io/#byrizki/rusto-rs)
[![Build & Release](https://github.com/byrizki/rusto-rs/actions/workflows/build.yml/badge.svg)](https://github.com/byrizki/rusto-rs/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

**RustO!** is a high-performance Optical Character Recognition (OCR) engine and cross-platform toolkit written in pure Rust. Based on [RapidOCR](https://github.com/RapidAI/RapidOCR) and powered by [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) models with the pure-Rust [RTen](https://github.com/robertknight/rten) inference engine (supporting both `.rten` FlatBuffers and standard `.onnx`), RustO! delivers sub-second inference speeds, ultra-low memory overhead, zero C++ / CMake build dependencies, and 99.3%+ parity with OpenCV-based solutions.

---

## 🎯 Key Features

- **🚀 100% Pure Rust Core** — Zero OpenCV and zero C++ dependencies. Powered by pure-Rust RTen inference, image processing, DBNet polygon contour detection, and unclip algorithms.
- **⚡ Fast & Lightweight** — Pure-Rust vector-accelerated inference with SIMD optimizations (AVX, NEON), link-time optimization (LTO), and single codegen unit compilation.
- **📄 Spatial Layout Text Reconstruction** — Reconstructs human-readable document layouts (multi-column tables, invoices, forms) with configurable visual XY spatial spacing.
- **🧠 Full Model Series Support** — Support for **PP-OCRv6** (Tiny, Small, Medium), **PP-OCRv5** (Mobile, Server), and **PP-OCRv4** (Mobile, Server) with orientation classification.
- **📦 Modular Distribution** — Core runtimes are stripped of forced model bloat. Users can choose pre-packaged model tiers or bring their own custom models.
- **🌐 Universal Multi-Platform SDKs** — Ready-to-use packages for **Rust**, **.NET / C#**, **React Native**, **iOS (Swift)**, **Android (Kotlin)**, and **C FFI**.

---

## 📦 Multi-Platform Packages Ecosystem

| Platform | Package / Registry | Description |
|---|---|---|
| **Rust** | `cargo add rusto-rs` ([crates.io](https://crates.io/crates/rusto-rs)) | Pure Rust library + CLI tool |
| **Web (WASM)** | `npm install @rustors/web` ([npm](https://www.npmjs.com/package/@rustors/web)) | Browser-native pure Rust RTen WASM OCR engine |
| **React** | `npm install @rustors/react` ([npm](https://www.npmjs.com/package/@rustors/react)) | React hooks & components for browser OCR |
| **.NET / C#** | `dotnet add package RustODotnet` ([NuGet](https://www.nuget.org/packages/RustODotnet)) | Managed .NET library + Windows/Linux/macOS native runtimes |
| **React Native** | `npm install @rustors/react-native` ([npm](https://www.npmjs.com/package/@rustors/react-native)) | Cross-platform React Native TypeScript bridge |
| **iOS** | `pod 'RustO'` ([CocoaPods](https://cocoapods.org/pods/RustO)) | Swift library + Universal XCFramework (Device & Simulator) |
| **Android** | `com.github.byrizki.rusto-rs:rusto-android` ([JitPack](https://jitpack.io/#byrizki/rusto-rs)) | Kotlin library + AAR with ARM64, ARMv7, x86, x86_64 |
| **C / Native** | `librusto.so` / `librusto.dylib` / `rusto.dll` | C FFI shared libraries for custom integrations |

---

## ⚠️ Notable Breaking Changes in v0.3.0

Upgrading from **v0.2.x** to **v0.3.0** introduces significant architectural upgrades:

### 1. Pure-Rust RTen Engine (MNN Removed)
- **Zero C++ / CMake toolchain**: In v0.2, RustO relied on Alibaba MNN which required CMake, C++ compilers, and platform-specific native runtime bindings. In v0.3, the core inference engine is 100% pure-Rust powered by [RTen](https://github.com/robertknight/rten).
- **New model formats (`.rten` & `.onnx`)**: Proprietary `.mnn` models are deprecated and replaced with high-performance `.rten` (FlatBuffers) or standard `.onnx` models.
- **Model filenames**: Update model references from `det.mnn` / `rec.mnn` to `det.rten` / `rec.rten` (or `det.onnx` / `rec.onnx`).

### 2. Session Abstraction (`EngineSession` replaces `MnnSession`)
- **`MnnSession` is deprecated**: Use the generic `EngineSession` wrapper.
- **Pluggable custom inference backends**: `EngineSession` implements `InferenceSession` and can wrap either built-in RTen or custom runtime engines (e.g. ONNX Runtime `ort`, Tract, Candle) via `EngineSession::from_custom` or `RustO::with_custom_engines(config, det_session, rec_session)`.
- **Constructors**: Use `EngineSession::from_det_config(&config.det)` and `EngineSession::from_rec_config(&config.rec)`.

### 3. Decoupled Initialization vs Per-Request Runtime Options
- **`InitializeConfig`**: Used solely for static model loading, dictionaries, and hardware thread configuration.
- **`OcrRunOptions`**: Configures per-image inference (score thresholds, dimension limits, output shape).
- **Flat options hierarchy**: `detection` and `postprocess` are now direct sibling fields on `OcrRunOptions` (the legacy nested `preprocessing` block has been removed).
- **Calibration & Optimization**: ID card and document preprocessing options (`CalibrationOptions`, `OptimizationOptions`) are passed per request via `OcrRunOptions.calibration` and `OcrRunOptions.optimization`.

### 4. Ecosystem & NPM Package Namespace Migration (`@rustors/*`)
- **Unified `@rustors` scope**: All npm packages have migrated from legacy 0.2.x standalone package names to the official `@rustors` scope:
  - `react-native-rusto` → [`@rustors/react-native`](https://www.npmjs.com/package/@rustors/react-native)
  - `rusto-web` → [`@rustors/web`](https://www.npmjs.com/package/@rustors/web)
  - `react-rusto` → [`@rustors/react`](https://www.npmjs.com/package/@rustors/react)
  - `rusto-models-*` → `@rustors/model-*` (e.g. `@rustors/model-ppocrv6-tiny`)
- **No legacy fallbacks**: In line with the v0.3.0 breaking release, legacy package names and fallbacks are deprecated and removed.
- Prepackaged model distributions for Android (AAR), iOS (CocoaPods), .NET (NuGet), and React Native (npm) have been upgraded to v0.3.0 with `.rten` / `.onnx` models.

---

## 🚀 Quick Start by Language

### 1. Rust

```rust
use rusto::{DetectTextResult, DetectionRunOptions, ImageSource, InitializeConfig, OcrRunOptions, PostprocessRunOptions, RustO};

let mut ocr = RustO::initialize(InitializeConfig::ppv6("det.rten", "rec.rten", "dict.txt"))?;
match ocr.detect_text(&ImageSource::Path("document.jpg".into()), &OcrRunOptions::default())? {
    DetectTextResult::Structured(results) => println!("{:?}", results),
    DetectTextResult::Spatial(text) => println!("{text}"),
}
```

---


### 2. .NET / C#

```csharp
using RustODotnet;

using var ocr = RustO.Initialize(InitializeConfig.Ppv6());
var result = ocr.DetectText(
    new UriImageSource("invoice.jpg"),
    new OcrRunOptions { Output = OutputGranularity.Words });

if (result is StructuredDetectTextResult structured)
    foreach (var item in structured.Items)
        Console.WriteLine($"{item.Text} ({item.Score:F2})");
// Total (0.98)
// $12.50 (0.96)
```

---

### 3. Web (WASM / Browser)

```bash
npm install @rustors/web @rustors/model-ppocrv6-tiny
# or with INT8 quantized model for smaller bundle size:
# npm install @rustors/web @rustors/model-ppocrv6-tiny-int8
```

```typescript
import { initialize, detectText } from '@rustors/web';
import { getModelUrls } from '@rustors/model-ppocrv6-tiny';

// 1. Initialize engine with model URLs (hosted locally or via CDN)
const urls = getModelUrls('/models/ppocrv6-tiny');
await initialize({
  preset: 'ppv6',
  models: {
    detection: urls.detection,
    recognition: urls.recognition,
    dictionary: urls.dictionary,
  },
});

// 2. Run OCR on an Image, Canvas, File, Blob, or URL
const results = await detectText(document.getElementById('receipt') as HTMLImageElement);
for (const item of results) {
  console.log(`${item.text} (${item.score})`);
}
```

---

### 4. React

```bash
npm install @rustors/react @rustors/web @rustors/model-ppocrv6-tiny
```

```tsx
import React, { useState } from 'react';
import { RustoProvider, useOcr, OcrOverlay, OcrDropzone } from '@rustors/react';
import { getModelUrls } from '@rustors/model-ppocrv6-tiny';

const urls = getModelUrls('/models/ppocrv6-tiny');

export function App() {
  return (
    <RustoProvider
      config={{
        preset: 'ppv6',
        models: {
          detection: urls.detection,
          recognition: urls.recognition,
          dictionary: urls.dictionary,
        },
      }}
    >
      <OcrScanner />
    </RustoProvider>
  );
}

function OcrScanner() {
  const { detect, results, isProcessing } = useOcr();
  const [imageSrc, setImageSrc] = useState<string | null>(null);

  return (
    <div>
      <OcrDropzone
        onFileSelect={async (file) => {
          setImageSrc(URL.createObjectURL(file));
          await detect(file);
        }}
      />
      {isProcessing && <p>Scanning document...</p>}
      {imageSrc && results.length > 0 && (
        <OcrOverlay imageSrc={imageSrc} results={results} />
      )}
    </div>
  );
}
```

---

### 5. React Native

Install the npm package and choose your model package for iOS and Android:

```bash
npm install @rustors/react-native
# or yarn add @rustors/react-native / pnpm add @rustors/react-native
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
    implementation 'com.github.byrizki.rusto-rs:rusto-models-ppocrv6-tiny:v0.3.0'
}
```

**JavaScript / TypeScript Usage:**
```typescript
import { initialize, detectText } from '@rustors/react-native';

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

### 6. iOS (Swift)

```swift
let ocr = try RustO.initialize(config: .ppv6())
let result = try ocr.detectText(.uri("receipt.jpg"), options: .init(output: .words))

if case .structured(let items) = result {
    items.forEach { print("\($0.text) (\($0.score))") }
}
```

---

### 7. Android (Kotlin)

```kotlin
RustO.initialize(context).use { ocr ->
    when (val result = ocr.detectText(
        ImageSource.Uri("/path/to/image.jpg"),
        OcrRunOptions(output = OutputGranularity.WORDS),
    )) {
        is DetectTextResult.Structured -> result.items.forEach { println("${it.text} (${it.score})") }
        is DetectTextResult.Spatial -> println(result.text)
    }
}
```

---

### 8. Command Line Interface (CLI)

```bash
# JSON output (default)
cargo run --release -- --det-model det.rten --rec-model rec.rten --dict dict.txt image.jpg

# Ordered text output
cargo run --release -- --det-model det.rten --rec-model rec.rten --dict dict.txt --format text-ordered image.jpg

# TSV / Plain text output
cargo run --release -- --det-model det.rten --rec-model rec.rten --dict dict.txt --format tsv image.jpg
```

---

## 🧠 Supported OCR Models & Tiers

RustO! supports all PaddleOCR model generations in lightweight RTen format (`.rten` FlatBuffers and `.onnx`), including standard FP32 and quantized INT8 models:

| Series | Tier / Variant | Total Size | Description |
|---|---|---|---|
| **PP-OCRv6** | **Tiny (INT8)** | **~3.4 MB** | Quantized INT8 weights. Minimal memory & size for web, mobile, and IoT. |
| **PP-OCRv6** | **Tiny** (Default) | **~6.0 MB** | MetaFormer PPLCNetV4 + 50-language unified dictionary. Ideal for mobile & edge. |
| **PP-OCRv6** | **Small (INT8)** | **~16.2 MB** | Quantized INT8 weights with high accuracy and reduced memory. |
| **PP-OCRv6** | **Small** | ~30 MB | Higher accuracy PP-OCRv6 models with expanded capacity. |
| **PP-OCRv6** | **Medium (INT8)** | **~75.7 MB** | Quantized INT8 weights for server-grade accuracy at half the footprint. |
| **PP-OCRv6** | **Medium** | ~134 MB | Server-grade accuracy PP-OCRv6 models. |
| **PP-OCRv5** | **Mobile** | ~28 MB | PP-OCRv5 lightweight mobile models (Chinese/English). |
| **PP-OCRv5** | **Server** | ~270 MB | PP-OCRv5 high-capacity server detection & recognition. |
| **PP-OCRv4** | **Mobile** | ~23 MB | PP-OCRv4 mobile models with orientation/direction classifier. |
| **PP-OCRv4** | **Server** | ~300 MB | PP-OCRv4 server models with orientation/direction classifier. |

### 🌐 Prepackaged Models & Multilingual Support

- **PP-OCRv6 (Unified Default)**: Recognizes 50+ languages simultaneously (Latin, Cyrillic, CJK, Devanagari, Arabic, etc.) out-of-the-box using a unified dictionary (`ppocrv6_dict.txt`). No separate language model downloads needed.
- **PP-OCRv6 INT8 Packages**: Ready-to-use quantized model packages are available on [npm](https://www.npmjs.com/search?q=%40rustors%2Fmodel-ppocrv6) (`@rustors/model-ppocrv6-*-int8`) and [NuGet](https://www.nuget.org/packages?q=RustODotnet.Models.PPOCRv6) (`RustODotnet.Models.PPOCRv6.*.Int8`).
- **PP-OCRv5 & PP-OCRv4 Language Packs**: Dedicated recognition models for Arabic, Cyrillic, Devanagari, Greek, Japanese, Korean, Latin, Tamil, Telugu, Thai, and Traditional Chinese across Android (AAR), iOS (Pods), and .NET (NuGet).

👉 **[View the Complete Multilingual OCR Guide & Package Matrix](https://byrizki.github.io/rusto-rs/en/advance-guide/multilingual-ocr)** ([local docs](./docs/content/en/03.advance-guide/02.multilingual-ocr.md))

---

## ⚙️ Configuration & Runtime Options

RustO separates static initialization from per-request execution:

1. **`InitializeConfig`** — Configures model sessions, vocabulary dictionary, and hardware threads.
2. **`OcrRunOptions`** — Configures per-image options: output shape (`lines`, `words`, `spatial`), score threshold, resize bounds, and postprocessing.

```rust
use rusto::{InitializeConfig, OcrRunOptions, OutputGranularity, RustO};

// 1. Initialize engine once
let mut ocr = RustO::initialize(InitializeConfig::ppv6(
    "models/det.rten",
    "models/rec.rten",
    "models/dict.txt",
))?;

// 2. Configure per request without mutating engine state
let result = ocr.detect_text(
    &ImageSource::Path("document.jpg".into()),
    &OcrRunOptions {
        output: OutputGranularity::Spatial,
        text_score: Some(0.55),
        max_side_len: Some(1600.0),
        ..Default::default()
    },
)?;
```

👉 **[Read the Full API Reference & Option Specifications](https://byrizki.github.io/rusto-rs/en/api-reference/ocr-run-options)** ([local docs](./docs/content/en/04.api-reference/))

---

## 📖 Documentation & Advanced Guides

Explore deep-dive guides and comprehensive references on the [RustO! Documentation Site](https://byrizki.github.io/rusto-rs/):

| Topic | Description | Link |
|---|---|---|
| **Models & Tiers Architecture** | Model formats (`.rten`, `.onnx`), INT8 quantization, and pluggable `InferenceSession` engines | [Read Guide](https://byrizki.github.io/rusto-rs/en/advance-guide/models-and-tiers) ([docs](./docs/content/en/03.advance-guide/01.models-and-tiers.md)) |
| **Multilingual OCR** | 50+ languages support, scripts matrix, and dedicated PP-OCRv5/v4 language packages | [Read Guide](https://byrizki.github.io/rusto-rs/en/advance-guide/multilingual-ocr) ([docs](./docs/content/en/03.advance-guide/02.multilingual-ocr.md)) |
| **Spatial Layout Reconstruction** | 2D XY spatial layout reconstruction for invoices, receipts, and multi-column tables | [Read Guide](https://byrizki.github.io/rusto-rs/en/advance-guide/spatial-layout-reconstruction) ([docs](./docs/content/en/03.advance-guide/03.spatial-layout-reconstruction.md)) |
| **Image Preprocessing & Tuning** | DBNet binarization thresholds, polygon unclipping, morphological dilation, and dimension limits | [Read Guide](https://byrizki.github.io/rusto-rs/en/advance-guide/image-preprocessing-and-tuning) ([docs](./docs/content/en/03.advance-guide/04.image-preprocessing-and-tuning.md)) |
| **Orientation & Classification** | 180° line direction classifier and 0°/90°/180°/270° document page rotation | [Read Guide](https://byrizki.github.io/rusto-rs/en/advance-guide/classification-and-orientation) ([docs](./docs/content/en/03.advance-guide/05.classification-and-orientation.md)) |
| **Performance & Benchmarks** | Latency, memory footprint, and parity benchmarks across x86-64, ARM64, and mobile | [Read Guide](https://byrizki.github.io/rusto-rs/en/advance-guide/performance-and-benchmarks) ([docs](./docs/content/en/03.advance-guide/06.performance-and-benchmarks.md)) |
| **Troubleshooting Guide** | Resolving model paths, Android JNI linking, React Native pods, and high-resolution camera OOM | [Read Guide](https://byrizki.github.io/rusto-rs/en/advance-guide/troubleshooting) ([docs](./docs/content/en/03.advance-guide/07.troubleshooting.md)) |
| **API Reference** | Specifications for `InitializeConfig`, `OcrRunOptions`, and `TextResult` across all platforms | [API Reference](https://byrizki.github.io/rusto-rs/en/api-reference/initialize-config) ([docs](./docs/content/en/04.api-reference/)) |

---

## ⚡ Performance & Benchmarks

Tested on standard document scans (A4 invoice / ID card, ~1200–1600px, 15–25 text lines) across platforms:

| Capability / Metric | RustO! (Pure Rust Core) | OpenCV / C++ Stacks |
|---|:---:|:---:|
| **CLI Binary Size (Self-Contained)** | ✅ **~5.4 MB** | ❌ ~50 MB+ (OpenCV + ONNX) |
| **C FFI Shared Library Size** | ✅ **~304 KB** | ❌ ~15–30 MB |
| **Android Native Size (Per-ABI)** | ✅ **~1.8 MB** | ❌ ~20 MB+ (libopencv.so + ONNX) |
| **WebAssembly (WASM) Bundle** | ✅ **~848 KB** | ❌ ~30 MB+ (Emscripten) |
| **Zero External C++ Dependencies** | ✅ Pure Rust (`cargo build`) | ❌ Requires CMake, OpenCV & NDK |
| **Peak Memory Footprint (RAM)** | ✅ **~85–120 MB** | ❌ ~250 MB+ runtime overhead |
| **Inference Latency (A4 Scan)** | ✅ **~80 ms** det / **~120 ms** rec | ✅ ~85 ms det / ~125 ms rec |
| **PaddleOCR Accuracy Parity** | ✅ **99.3%+** parity | ✅ Baseline (100%) |
| **Memory & Thread Safety** | ✅ Guaranteed by Rust borrow checker | ❌ Manual pointers & thread locks |
| **Ready Multi-Platform Packages** | ✅ Cargo, npm, NuGet, CocoaPods, AAR | ⚠️ Complex manual native compilation |

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

RustO! is inspired by and builds upon the work of:
- **[RapidOCR](https://github.com/RapidAI/RapidOCR)** — Architecture and OCR pipeline reference
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** — Pretrained OCR models (PP-OCRv6, PP-OCRv5, PP-OCRv4)
- **[RTen](https://github.com/robertknight/rten)** — Pure Rust deep learning inference engine powering RustO! v0.3+
- **[Alibaba MNN](https://github.com/alibaba/MNN)** — High-performance inference engine that powered RustO! in version 0.2.x
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
  note = {Based on RapidOCR and powered by PaddleOCR models with pure Rust RTen inference}
}
```
