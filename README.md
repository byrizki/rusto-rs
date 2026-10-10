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

**RustO!** is a pure-Rust Optical Character Recognition (OCR) engine and multi-platform toolkit. Powered by PaddleOCR models and the [RTen](https://github.com/robertknight/rten) inference engine (`.rten` & `.onnx`), RustO! delivers fast, memory-safe text extraction with zero C++ or OpenCV dependencies.

---

## 🎯 Key Features

- **Pure Rust Engine** — Zero C++ or OpenCV runtime dependencies. Build directly with `cargo build`.
- **Fast & Lightweight** — SIMD-accelerated inference, ~5.4 MB CLI binary, and ~85–120 MB peak RAM.
- **PP-OCR Model Tiers** — PP-OCRv6 (50+ languages unified), PP-OCRv5, and PP-OCRv4 (FP32 & INT8).
- **Document Layouts** — Outputs bounding boxes, line/word tokens, or formatted 2D spatial text layouts.
- **Cross-Platform SDKs** — First-class libraries for Rust, Web (WASM), React, React Native, iOS, Android, and .NET.

---

## 📦 Packages

| Platform | Package | Description |
|---|---|---|
| **Rust** | [`cargo add rusto-rs`](https://crates.io/crates/rusto-rs) | Pure Rust library and CLI |
| **Web (WASM)** | [`@rustors/web`](https://www.npmjs.com/package/@rustors/web) | Browser-native WASM engine |
| **React** | [`@rustors/react`](https://www.npmjs.com/package/@rustors/react) | React hooks and components |
| **React Native** | [`@rustors/react-native`](https://www.npmjs.com/package/@rustors/react-native) | iOS and Android native bridge |
| **iOS** | [`pod 'RustO'`](https://cocoapods.org/pods/RustO) | Swift library & XCFramework |
| **Android** | [`rusto-android`](https://jitpack.io/#byrizki/rusto-rs) | Kotlin library & AAR |
| **.NET / C#** | [`RustODotnet`](https://www.nuget.org/packages/RustODotnet) | Managed .NET library |
| **C / Native** | `librusto.so` / `.dylib` / `.dll` | C FFI shared libraries |

---

## ⚠️ Notable Breaking Changes in v0.3.0

Upgrading from **v0.2.x** to **v0.3.0** transitions the core engine to pure Rust with zero C++ toolchain dependencies:

| Aspect | v0.2.x | v0.3.0 (Current) |
|---|---|---|
| **Inference Engine** | Alibaba MNN (requires C++ / CMake) | **Pure Rust RTen** (zero native toolchain) |
| **Model Formats** | Proprietary `.mnn` models | High-performance **`.rten`** (FlatBuffers) or **`.onnx`** |
| **Session API** | `MnnSession` | **`EngineSession`** (pluggable `InferenceSession`) |
| **Configuration** | Nested `preprocessing` block | Decoupled **`InitializeConfig`** & **`OcrRunOptions`** |
| **NPM Scope** | Standalone names (`react-rusto`, etc.) | Unified **`@rustors/*`** scope |

👉 **[Read the Complete v0.2 → v0.3 Migration Guide](https://byrizki.github.io/rusto-rs/en/advance-guide/migration-guide)** for code comparisons, API changes, and step-by-step instructions.

---

## 🚀 Quick Start by Language

### 1. Rust

```rust
use rusto::{ImageSource, InitializeConfig, OcrRunOptions, RustO};

let mut ocr = RustO::initialize(InitializeConfig::ppv6("det.rten", "rec.rten", "dict.txt"))?;
let result = ocr.detect_text(&ImageSource::Path("document.jpg".into()), &OcrRunOptions::default())?;
println!("{result:?}");
```

---

### 2. .NET / C#

```csharp
using RustODotnet;

using var ocr = RustO.Initialize(InitializeConfig.Ppv6());
var result = ocr.DetectText(new UriImageSource("invoice.jpg"), new OcrRunOptions());

if (result is StructuredDetectTextResult structured) {
    foreach (var item in structured.Items) {
        Console.WriteLine($"{item.Text} ({item.Score:F2})");
    }
}
```

---

### 3. Web (WASM / Browser)

```bash
npm install @rustors/web @rustors/model-ppocrv6-tiny
```

```typescript
import { initialize, detectText } from '@rustors/web';
import { getModelUrls } from '@rustors/model-ppocrv6-tiny';

const urls = getModelUrls('/models/ppocrv6-tiny');
await initialize({
  preset: 'ppv6',
  models: { detection: urls.detection, recognition: urls.recognition, dictionary: urls.dictionary },
});

const results = await detectText(document.querySelector('img')!);
results.forEach((r) => console.log(`${r.text} (${r.score})`));
```

---

### 4. React

```bash
npm install @rustors/react @rustors/web @rustors/model-ppocrv6-tiny
```

```tsx
import { RustoProvider, useOcr } from '@rustors/react';
import { getModelUrls } from '@rustors/model-ppocrv6-tiny';

const urls = getModelUrls('/models/ppocrv6-tiny');

function OcrScanner() {
  const { detect, results, isProcessing } = useOcr();

  return (
    <div>
      <input type="file" onChange={(e) => e.target.files?.[0] && detect(e.target.files[0])} />
      {isProcessing && <p>Scanning...</p>}
      {results.map((r, i) => <p key={i}>{r.text} ({r.score.toFixed(2)})</p>)}
    </div>
  );
}

export function App() {
  return (
    <RustoProvider config={{ preset: 'ppv6', models: urls }}>
      <OcrScanner />
    </RustoProvider>
  );
}
```

---

### 5. React Native

```bash
npm install @rustors/react-native
```

```typescript
import { initialize, detectText } from '@rustors/react-native';

await initialize();
const results = await detectText({ uri: 'file:///path/to/image.jpg' });
results.forEach((r) => console.log(`${r.text} (${r.score})`));
```

---

### 6. iOS (Swift)

```swift
let ocr = try RustO.initialize(config: .ppv6())
let result = try ocr.detectText(.uri("receipt.jpg"))

if case .structured(let items) = result {
    items.forEach { print("\($0.text) (\($0.score))") }
}
```

---

### 7. Android (Kotlin)

```kotlin
RustO.initialize(context).use { ocr ->
    val result = ocr.detectText(ImageSource.Uri("/path/to/image.jpg"))
    if (result is DetectTextResult.Structured) {
        result.items.forEach { println("${it.text} (${it.score})") }
    }
}
```

---

### 8. Command Line Interface (CLI)

```bash
# JSON output
cargo run --release -- --det-model det.rten --rec-model rec.rten --dict dict.txt image.jpg

# Formatted text output
cargo run --release -- --det-model det.rten --rec-model rec.rten --dict dict.txt --format text-ordered image.jpg
```

---

## 🧠 Supported OCR Models & Tiers

RustO! supports PP-OCRv6 (recognizing 50+ languages simultaneously with a unified dictionary) alongside legacy v5/v4 models:

| Tier | Standard (FP32) | Quantized (INT8) | Best For |
|---|:---:|:---:|---|
| **PP-OCRv6 Tiny** (Default) | ~6.0 MB | **~3.4 MB** | Web (WASM), mobile, IoT & embedded edge |
| **PP-OCRv6 Small** | ~30 MB | **~16.2 MB** | Balanced accuracy for mobile & desktop |
| **PP-OCRv6 Medium** | ~134 MB | **~75.7 MB** | Server-grade accuracy workloads |
| **PP-OCRv5 / v4** | ~23–28 MB (Mobile) | — | Legacy pipelines & single-language packs |

Prepackaged models are available on [npm](https://www.npmjs.com/search?q=%40rustors%2Fmodel-ppocrv6) (`@rustors/model-ppocrv6-*`) and [NuGet](https://www.nuget.org/packages?q=RustODotnet.Models.PPOCRv6) (`RustODotnet.Models.PPOCRv6.*`).

👉 **[View Complete Multilingual OCR Guide & Model Matrix](https://byrizki.github.io/rusto-rs/en/advance-guide/multilingual-ocr)**

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

👉 **[Read the Full API Reference & Option Specifications](https://byrizki.github.io/rusto-rs/en/api-reference/ocr-run-options)**

---

## 📖 Documentation & Advanced Guides

Explore deep-dive guides and comprehensive references on the [RustO! Documentation Site](https://byrizki.github.io/rusto-rs/):

| Topic | Description | Link |
|---|---|---|
| **Models & Tiers Architecture** | Model formats (`.rten`, `.onnx`), INT8 quantization, and pluggable `InferenceSession` engines | [Read Guide](https://byrizki.github.io/rusto-rs/en/advance-guide/models-and-tiers) |
| **Multilingual OCR** | 50+ languages support, scripts matrix, and dedicated PP-OCRv5/v4 language packages | [Read Guide](https://byrizki.github.io/rusto-rs/en/advance-guide/multilingual-ocr) |
| **Spatial Layout Reconstruction** | 2D XY spatial layout reconstruction for invoices, receipts, and multi-column tables | [Read Guide](https://byrizki.github.io/rusto-rs/en/advance-guide/spatial-layout-reconstruction) |
| **Image Preprocessing & Tuning** | DBNet binarization thresholds, polygon unclipping, morphological dilation, and dimension limits | [Read Guide](https://byrizki.github.io/rusto-rs/en/advance-guide/image-preprocessing-and-tuning) |
| **Orientation & Classification** | 180° line direction classifier and 0°/90°/180°/270° document page rotation | [Read Guide](https://byrizki.github.io/rusto-rs/en/advance-guide/classification-and-orientation) |
| **Performance & Benchmarks** | Latency, memory footprint, and parity benchmarks across x86-64, ARM64, and mobile | [Read Guide](https://byrizki.github.io/rusto-rs/en/advance-guide/performance-and-benchmarks) |
| **Troubleshooting Guide** | Resolving model paths, Android JNI linking, React Native pods, and high-resolution camera OOM | [Read Guide](https://byrizki.github.io/rusto-rs/en/advance-guide/troubleshooting) |
| **v0.2 → v0.3 Migration Guide** | Upgrading to pure-Rust RTen, .rten model formats, EngineSession, and @rustors scope | [Read Guide](https://byrizki.github.io/rusto-rs/en/advance-guide/migration-guide) |
| **API Reference** | Specifications for `InitializeConfig`, `OcrRunOptions`, and `TextResult` across all platforms | [API Reference](https://byrizki.github.io/rusto-rs/en/api-reference/initialize-config) |

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
