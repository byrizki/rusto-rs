<div align="center">

# RustO! 🦀

**High-Performance, Pure Rust OCR Engine & Multi-Platform Toolkit**

[![Crates.io](https://img.shields.io/crates/v/rusto-rs.svg?logo=rust&logoColor=white&color=orange)](https://crates.io/crates/rusto-rs)
[![docs.rs](https://img.shields.io/docsrs/rusto-rs?logo=docs.rs&logoColor=white)](https://docs.rs/rusto-rs)
[![PyPI](https://img.shields.io/pypi/v/rusto.svg?logo=pypi&logoColor=white&color=blue)](https://pypi.org/project/rusto/)
[![npm (node)](https://img.shields.io/npm/v/@rustors/node.svg?logo=node.js&logoColor=white&color=339933&label=npm%20node)](https://www.npmjs.com/package/@rustors/node)
[![npm (react-native)](https://img.shields.io/npm/v/@rustors/react-native.svg?logo=react&logoColor=white&color=61DAFB&label=npm%20rn)](https://www.npmjs.com/package/@rustors/react-native)
[![pub.dev (dart)](https://img.shields.io/pub/v/rusto.svg?logo=dart&logoColor=white&color=0175C2&label=pub%20dart)](https://pub.dev/packages/rusto)
[![pub.dev (flutter)](https://img.shields.io/pub/v/rusto_flutter.svg?logo=flutter&logoColor=white&color=02569B&label=pub%20flutter)](https://pub.dev/packages/rusto_flutter)
[![NuGet](https://img.shields.io/nuget/v/RustODotnet.svg?logo=nuget&logoColor=white&color=004880)](https://www.nuget.org/packages/RustODotnet)
[![CocoaPods](https://img.shields.io/cocoapods/v/RustO.svg?logo=cocoapods&logoColor=white&color=EE3322)](https://cocoapods.org/pods/RustO)
[![JitPack](https://jitpack.io/v/byrizki/rusto-rs.svg)](https://jitpack.io/#byrizki/rusto-rs)
[![Documentation](https://img.shields.io/badge/docs-byrizki.com%2Frusto--rs-blue.svg)](https://byrizki.com/rusto-rs/)
[![Library CI](https://github.com/byrizki/rusto-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/byrizki/rusto-rs/actions/workflows/ci.yml)
[![Docs CI](https://github.com/byrizki/rusto-rs/actions/workflows/docs-ci.yml/badge.svg)](https://github.com/byrizki/rusto-rs/actions/workflows/docs-ci.yml)
[![Build & Release](https://github.com/byrizki/rusto-rs/actions/workflows/build.yml/badge.svg)](https://github.com/byrizki/rusto-rs/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Known Vulnerabilities](https://snyk.io/test/github/byrizki/rusto-rs/badge.svg)](https://snyk.io/test/github/byrizki/rusto-rs)

</div>

**RustO!** is a pure-Rust Optical Character Recognition (OCR) engine and multi-platform toolkit. Powered by [DataLab Indra OCR](https://github.com/datalab-to/indra-ocr) and PaddleOCR models running on RustO's pure-Rust MNN neural inference runtime (`rusto-nn`) with pluggable [RTen](https://github.com/robertknight/rten) and ONNX support, RustO! delivers fast, memory-safe text extraction and document layout understanding with zero C++ or OpenCV dependencies.

---

## 🎯 Key Features

- **Pure Rust MNN Engine (`rusto-nn`)** — Zero C++ or OpenCV runtime dependencies. SIMD vectorization (AVX-512, AVX2, NEON, WASM SIMD128) with pluggable RTen, ONNX, and hardware backends (WebGPU, CUDA). Build directly with `cargo build`.
- **Flagship Indra OCR + PP-OCRv6** — DataLab Indra OCR (~5.4 MB, **45.98%** OmniDocBench accuracy) by default alongside PP-OCRv6 (50+ languages unified), PP-OCRv5, and PP-OCRv4 (FP32 & INT8).
- **End-to-End Document Pipeline** — Built-in 3D UVDoc unwarping, orientation detection, PicoDet layout region classification (`regionId` / `layoutType`), and neural graph reading order (`indra_order.rten`).
- **Unified Fluent Builder API** — Consistent `.withAutoUnwarp()`, `.withLayoutReading()`, `.asLines()` / `.asWords()` / `.asSpatial()` / `.asTsv()`, and single `.run()` terminal execution across every SDK.
- **12 Cross-Platform SDKs + CLI** — First-class libraries for Rust, Node.js, Python, Go, Java, .NET, Dart, Flutter, Web (WASM), React, React Native, iOS, Android, and C FFI, plus the `rusto` CLI.

---

## 📦 Packages

| Platform | Package | Description |
|---|---|---|
| **Rust** | [`cargo add rusto-rs`](https://crates.io/crates/rusto-rs) | Pure Rust library and CLI |
| **Node.js (N-API)** | [`@rustors/node`](https://www.npmjs.com/package/@rustors/node) | Native Node.js N-API addon |
| **Python** | [`pip install rusto`](https://pypi.org/project/rusto) | UniFFI Python library & wheel |
| **Golang** | [`packages/go`](https://pkg.go.dev/github.com/byrizki/rusto-rs/packages/go) | Cgo Go module with embedded rpath |
| **Java (JVM)** | [`rusto-java`](https://jitpack.io/#byrizki/rusto-rs) (`com.github.byrizki.rusto-rs:rusto-java`) | Java & Kotlin JNA library via JitPack |
| **Dart** | [`rusto`](https://pub.dev/packages/rusto) | Pure Dart FFI library |
| **Flutter** | [`rusto_flutter`](https://pub.dev/packages/rusto_flutter) | Flutter plugin with Android & iOS bridges |
| **Web (WASM)** | [`@rustors/web`](https://www.npmjs.com/package/@rustors/web) | Browser-native WASM engine |
| **React** | [`@rustors/react`](https://www.npmjs.com/package/@rustors/react) | React hooks and components |
| **React Native** | [`@rustors/react-native`](https://www.npmjs.com/package/@rustors/react-native) | iOS and Android native bridge |
| **iOS** | `Package.swift` / [`pod 'RustO'`](https://cocoapods.org/pods/RustO) | Swift Package Manager & CocoaPods |
| **Android** | [`rusto-android`](https://jitpack.io/#byrizki/rusto-rs) | Kotlin library & AAR |
| **.NET / C#** | [`RustODotnet`](https://www.nuget.org/packages/RustODotnet) | Managed .NET library |
| **C / Native** | `librusto.so` / `.dylib` / `.dll` | C FFI shared libraries |

---

## ⚠️ Notable Breaking Changes in v0.4.0

Upgrading from **v0.3.x** to **v0.4.0** introduces a unified fluent builder API, 6-phase pipeline options, pipelined document layout/unwarping, and Indra OCR as the default flagship engine:

| Aspect | v0.3.x | v0.4.0 (Current) |
|---|---|---|
| **Default Model** | PP-OCRv6 Tiny (~6 MB) | **DataLab Indra OCR** (~5.4 MB, **45.98%** OmniDoc accuracy) |
| **Model Assets** | Bundled / platform model packages | **Decoupled standalone models on Hugging Face Hub** with zero-config auto-download |
| **API Paradigm** | Monolithic `detectText(source, options)` | **Unified fluent builder chaining** ending in `.run()` / `.Run()` |
| **Output Granularity** | Separate `*Spatial` methods | **`.asLines()` / `.asWords()` / `.asSpatial()`** (`as_*` / `As*`) |
| **Layout & Unwarp** | Disjoint separate engine methods | **Pipelined directly** via `.withLayoutReading()`, `.withAutoUnwarp()` |
| **Run Options** | Ad-hoc nested options object | **6-Phase structured pipeline** + 4 boolean toggles |
| **CLI Binary** | `rusto-rs` | **`rusto`** (`rusto <IMAGE>`, `rusto ocr`, `rusto download`) |

👉 **[Read the Complete v0.3 → v0.4 Migration Guide](https://byrizki.com/rusto-rs/en/advance-guide/migration-guide)** for code comparisons, API changes, and step-by-step instructions.

---

## 🚀 Quick Start by Language

### 1. Rust

```rust
use rusto::{DetectTextBuilderExt, ImageSource, InitializeConfig, RustO};

let mut ocr = RustO::initialize(InitializeConfig::default())?;
let result = ocr
    .detect_text_builder(ImageSource::Path("document.jpg".into()))
    .with_auto_unwarp(true)
    .with_layout_reading(true)
    .as_lines()
    .run()?;
println!("{result:?}");
```

---

### 2. .NET / C#

```csharp
using RustODotnet;

using var ocr = RustO.Initialize();
var result = ocr.DetectText("invoice.jpg")
    .WithAutoUnwarp()
    .WithLayoutReading()
    .AsLines()
    .Run();

if (result is StructuredDetectTextResult structured) {
    foreach (var item in structured.Items) {
        Console.WriteLine($"{item.Text} ({item.Score:F2})");
    }
}
```

---

### 3. Web (WASM / Browser)

```bash
npm install @rustors/web
```

```typescript
import { initialize, detectText } from '@rustors/web';

await initialize({
  preset: 'indra',
  models: {
    detection: '/models/indraocr/det.rten',
    recognition: '/models/indraocr/rec.rten',
    dictionary: '/models/indraocr/dict.txt',
  },
});

const results = await detectText(document.querySelector('img')!)
  .withAutoUnwarp()
  .withLayoutReading()
  .asLines()
  .run();

results.forEach((r) => console.log(`${r.text} (${r.score})`));
```

---

### 4. React

```bash
npm install @rustors/react @rustors/web
```

```tsx
import { RustoProvider, useOcr } from '@rustors/react';

const models = {
  detection: '/models/indraocr/det.rten',
  recognition: '/models/indraocr/rec.rten',
  dictionary: '/models/indraocr/dict.txt',
};

function OcrScanner() {
  const { detect, results, isProcessing } = useOcr();

  return (
    <div>
      <input
        type="file"
        onChange={(e) =>
          e.target.files?.[0] &&
          detect(e.target.files[0], { autoUnwarp: true, layoutReading: true })
        }
      />
      {isProcessing && <p>Scanning...</p>}
      {results.map((r, i) => (
        <p key={i}>
          {r.text} ({r.score.toFixed(2)})
        </p>
      ))}
    </div>
  );
}

export function App() {
  return (
    <RustoProvider config={{ preset: 'indra', models }}>
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
const results = await detectText({ uri: 'file:///path/to/image.jpg' })
  .withAutoUnwarp()
  .withLayoutReading()
  .asLines()
  .run();

results.forEach((r) => console.log(`${r.text} (${r.score})`));
```

---

### 6. iOS (Swift)

```swift
let ocr = try RustO.initialize()
let result = try ocr.detectText(image: "receipt.jpg")
    .withAutoUnwarp()
    .withLayoutReading()
    .asLines()
    .run()

if case .structured(let items) = result {
    items.forEach { print("\($0.text) (\($0.score))") }
}
```

---

### 7. Android (Kotlin)

```kotlin
RustO.initialize(context).use { ocr ->
    val result = ocr.detectText("/path/to/image.jpg")
        .withAutoUnwarp()
        .withLayoutReading()
        .asLines()
        .run()

    if (result is DetectTextResult.Structured) {
        result.items.forEach { println("${it.text} (${it.score})") }
    }
}
```

---

### 8. Node.js (N-API)

```bash
npm install @rustors/node
```

```javascript
const { RustO } = require('@rustors/node');

const engine = RustO.initialize();
const results = engine
  .detectText('invoice.png')
  .withAutoUnwarp()
  .withLayoutReading()
  .asLines()
  .run();

results.forEach((r) => console.log(`${r.text} (${(r.score * 100).toFixed(1)}%)`));
engine.close();
```

---

### 9. Python (UniFFI)

```bash
pip install rusto
```

```python
from rusto import RustO

with RustO.initialize() as engine:
    result = (
        engine.detect_text("invoice.png")
        .with_auto_unwarp()
        .with_layout_reading()
        .as_lines()
        .run()
    )

    if result.is_STRUCTURED():
        for item in result.results:
            print(f"{item.text} ({item.score:.2f})")
```

---

### 10. Golang (Cgo)

```bash
go get github.com/byrizki/rusto-rs/packages/go
```

```go
package main

import (
    "fmt"
    "github.com/byrizki/rusto-rs/packages/go"
)

func main() {
    engine, err := rusto.Initialize()
    if err != nil { panic(err) }
    defer engine.Close()

    res, err := engine.DetectText("invoice.png").
        WithAutoUnwarp(true).
        WithLayoutReading(true).
        AsLines().
        Run()
    if err != nil { panic(err) }

    for _, r := range res.Results {
        fmt.Printf("[%0.2f] %s\n", r.Score, r.Text)
    }
}
```

---

### 11. Java (JVM / JNA)

```java
import com.byrizki.rusto.RustO;
import com.byrizki.rusto.TextResult;

public class App {
    public static void main(String[] args) {
        try (RustO engine = RustO.initialize()) {
            RustO.DetectTextResult res = engine.detectText("invoice.png")
                .withAutoUnwarp()
                .withLayoutReading()
                .asLines()
                .run();
            for (TextResult r : res.results()) {
                System.out.printf("[%0.2f] %s%n", r.score(), r.text());
            }
        }
    }
}
```

---

### 12. Dart & Flutter

```dart
import 'package:rusto/rusto.dart';

final engine = RustO.initialize();
final result = engine
    .detectText('invoice.png')
    .withAutoUnwarp()
    .withLayoutReading()
    .asLines()
    .run();

if (result is DetectTextResultStructured) {
  for (final line in result.results) {
    print('${line.text} (${line.score})');
  }
}
engine.close();
```

---

### 13. Command Line Interface (CLI)

```bash
# Zero-config OCR (auto-downloads Indra OCR models on first run)
rusto image.jpg

# Multi-column layout reading order & 2D spatial text reconstruction
rusto --enable-unwarp --enable-layout --format spatial image.jpg

# Pre-download standalone RTen models for offline environments
rusto download --model indraocr
```

---

## 🧠 Supported OCR Models & Tiers

RustO! supports DataLab Indra OCR as its default flagship engine alongside PP-OCRv6 (recognizing 50+ languages simultaneously) and legacy v5/v4 models:

| Model / Tier | Standard (FP32) | Quantized (INT8) | Best For |
|---|:---:|:---:|---|
| **Indra OCR** (Default) | **~5.5 MB** | — | **Flagship default**: 2.5x faster (439 ms OmniDocBench), superior Latin/ID |
| **PP-OCRv6 Tiny** | ~6.0 MB | **~3.4 MB** | Lightweight edge, WASM, and IoT |
| **PP-OCRv6 Small** | ~30 MB | **~16.2 MB** | Balanced accuracy for mobile & desktop |
| **PP-OCRv6 Medium** | ~134 MB | **~75.7 MB** | Server-grade accuracy workloads |
| **PP-OCRv5 / v4** | ~23–28 MB (Mobile) | — | Legacy pipelines & single-language packs |

Pre-trained models are distributed as standalone assets on [Hugging Face Hub (`byrizki/rten-models`)](https://huggingface.co/byrizki/rten-models) with native CORS support and downloaded automatically on first run by the CLI and native SDKs.

👉 **[View Complete Multilingual OCR Guide & Model Matrix](https://byrizki.com/rusto-rs/en/advance-guide/multilingual-ocr)**

---

## ⚙️ Configuration & Runtime Options

RustO separates static initialization from per-request execution:

1. **`InitializeConfig`** — Configures model sessions, vocabulary dictionary, and hardware threads.
2. **`DetectTextBuilder` / `OcrRunOptions`** — Configures per-image options via fluent chaining: 4 toggles (`autoUnwarp`, `layoutReading`, `orientationDetection`, `lineClassification`) and 6 phases (`preDetection`, `detection`, `postDetection`, `preRecognition`, `recognition`, `postRecognition`).

```rust
use rusto::{DetectTextBuilderExt, ImageSource, InitializeConfig, RustO};

// 1. Initialize engine once (or RustO::initialize(InitializeConfig::default())? for auto-download)
let mut ocr = RustO::initialize(InitializeConfig::indra(
    "models/det.rten",
    "models/rec.rten",
    "models/dict.txt",
))?;

// 2. Configure per request via fluent builder chaining
let result = ocr
    .detect_text_builder(ImageSource::Path("document.jpg".into()))
    .with_auto_unwarp(true)
    .with_layout_reading(true)
    .as_spatial()
    .run()?;
```

👉 **[Read the Full API Reference & Option Specifications](https://byrizki.com/rusto-rs/en/api-reference/ocr-run-options)**

---

## 📖 Documentation & Advanced Guides

Explore deep-dive guides and comprehensive references on the [RustO! Documentation Site](https://byrizki.com/rusto-rs/):

| Topic | Description | Link |
|---|---|---|
| **Models & Tiers Architecture** | Model formats (`.rten`, `.onnx`), INT8 quantization, and pluggable `InferenceSession` engines | [Read Guide](https://byrizki.com/rusto-rs/en/advance-guide/models-and-tiers) |
| **Multilingual OCR** | 50+ languages support, scripts matrix, and dedicated PP-OCRv5/v4 language packages | [Read Guide](https://byrizki.com/rusto-rs/en/advance-guide/multilingual-ocr) |
| **Spatial Layout Reconstruction** | 2D XY spatial layout reconstruction for invoices, receipts, and multi-column tables | [Read Guide](https://byrizki.com/rusto-rs/en/advance-guide/spatial-layout-reconstruction) |
| **Image Preprocessing & Tuning** | DBNet binarization thresholds, polygon unclipping, morphological dilation, and dimension limits | [Read Guide](https://byrizki.com/rusto-rs/en/advance-guide/image-preprocessing-and-tuning) |
| **Multi-Column, Unwarp & Orientation** | Multi-column layout, neural reading order, 3D UVDoc unwarping, and orientation | [Read Guide](https://byrizki.com/rusto-rs/en/advance-guide/multi-column-unwarp-orientation) |
| **Decoded RGB Input** | Zero-copy direct OCR from uncompressed interleaved RGB8 pixel buffers | [Read Guide](https://byrizki.com/rusto-rs/en/advance-guide/use-decoded-rgb) |
| **Performance & Benchmarks** | Latency, memory footprint, and parity benchmarks across x86-64, ARM64, and mobile | [Read Guide](https://byrizki.com/rusto-rs/en/advance-guide/performance-and-benchmarks) |
| **Troubleshooting Guide** | Resolving model paths, Android JNI linking, React Native pods, and high-resolution camera OOM | [Read Guide](https://byrizki.com/rusto-rs/en/advance-guide/troubleshooting) |
| **v0.3 → v0.4 Migration Guide** | Upgrading to fluent builder API, 6-phase pipeline, Indra OCR default, and decoupled .rten models | [Read Guide](https://byrizki.com/rusto-rs/en/advance-guide/migration-guide) |
| **API Reference** | Specifications for `InitializeConfig`, `OcrRunOptions`, and `TextResult` across all platforms | [API Reference](https://byrizki.com/rusto-rs/en/api-reference/initialize-config) |

---

## ⚡ Performance & Benchmarks

Empirically evaluated against official PaddleOCR and RapidOCR (C++ ONNX Runtime) using the standard **OmniDocBench** dataset and high-resolution document scans (1600×993 px, 27 text lines).

> **Benchmark Hardware**: 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz (4 cores / 8 threads), 32 GB RAM, Linux x86_64.

| Capability / Metric | RustO! (Pure Rust Core) | OpenCV / C++ Stacks (RapidOCR / PaddleOCR) | Benchmark Reference |
|---|:---:|:---:|---|
| **CLI Binary Size (Self-Contained)** | ✅ **8.3 MB** | ❌ ~55 MB+ (libopencv + onnxruntime) | Production Linux x86-64 release |
| **C FFI Shared Library (`librusto.so`)** | ✅ **304 KB** | ❌ ~15–30 MB | Stripped release `.so` |
| **Android Native Size (Per-ABI)** | ✅ **~1.8 MB** | ❌ ~20 MB+ (`libopencv.so` + ONNX) | Android release AAR |
| **WebAssembly (WASM) Bundle** | ✅ **517 KB** (gzipped) | ❌ 3.78 MB (gzipped ONNX Runtime WASM) | Web client bundle size |
| **Zero External C++ Dependencies** | ✅ Pure Rust (`cargo build`) | ❌ Requires CMake, OpenCV & NDK | Toolchain build dependencies |
| **Peak Memory (OmniDocBench)** | ✅ **177.8 MB - 252.8 MB** | ❌ 392.5 MB (RapidOCR) / 659.3 MB (PaddleOCR) | OmniDocBench full dataset peak RSS |
| **Document Pipeline (1600×993 px)** | ⚡ **678.08 ms** (Det: 312 ms, Rec: 169 ms, 27 lines) | ⚡ 447.83 ms (C++ ONNX Runtime) | Base pipeline end-to-end benchmark |
| **OmniDocBench Mean Latency** | ⚡ **1,166.8 ms** (Indra) / **1,172.5 ms** (Tiny) | ⚡ 605.3 ms (PaddleOCR) / 1,011.5 ms (RapidOCR) | OmniDocBench full evaluation |
| **Transcription Parity (High-Res Scan)** | 🎯 **100% token identical** across all 27 lines | 🎯 Baseline reference | Character-for-character comparison |
| **OmniDocBench Baseline Accuracy (1 - NED)** | 🎯 **45.40%** (Tiny) / **45.98%** (Indra) | 🎯 45.33% (PaddleOCR Tiny) / 44.84% (RapidOCR) | Standard 1D reading order |
| **Full Pipeline Accuracy (Reading Order)** | 🎯 **63.75%** (RustO Indra Full) | 🎯 63.76% (Indra Official) / 57.18% (Paddle Structure)| Graph neural reading order |
| **Memory & Thread Safety** | ✅ Guaranteed by Rust borrow checker | ❌ Manual pointers & thread locks | Data-race free concurrency |

---

## 📁 Repository Structure

```
rusto-rs/
├── crates/
│   ├── rusto-nn/               # Pure-Rust neural inference runtime & hardware backends
│   ├── rusto-core/             # Core OCR pipeline (DBNet, CTC, UVDoc unwarp, PicoDet layout, reading order)
│   ├── rusto-rten/             # RTen (.rten) session backend
│   ├── rusto-mnn/              # Optional MNN C++ session backend
│   ├── rusto-node/             # Native Node.js N-API addon crate
│   └── rten-wasm/              # Browser WebAssembly OCR & layout runtime
├── src/                        # Workspace root crate, UniFFI/C FFI bridges & `rusto` CLI
├── include/                    # Canonical C header (`rusto.h`)
├── packages/
│   ├── node/                   # @rustors/node (Node.js N-API SDK)
│   ├── python/                 # rusto (Python UniFFI SDK)
│   ├── go/                     # Go Cgo module
│   ├── java/                   # rusto-java (JVM JNA SDK)
│   ├── dotnet/                 # RustODotnet (.NET / C# SDK)
│   ├── dart/                   # rusto (Pure Dart FFI SDK)
│   ├── flutter/                # rusto_flutter (Flutter plugin)
│   ├── web/                    # @rustors/web (Browser WASM SDK)
│   ├── react/                  # @rustors/react (React hooks & components)
│   ├── react-native/           # @rustors/react-native (iOS & Android bridge)
│   ├── android/                # rusto-android (Kotlin Android AAR)
│   └── ios/                    # RustO (Swift Package Manager & CocoaPods)
├── docs/                       # Bilingual documentation site & live WASM demo (EN / ID)
├── scripts/                    # Version bump, UniFFI codegen & package verification scripts
└── .github/workflows/          # CI, cross-platform artifact build & automated registry publishing
```

---

## 🛠️ Development & Testing

```bash
# Run core library unit tests across the workspace
cargo test --workspace --lib

# Build the `rusto` CLI in release mode
cargo build --release --bin rusto

# Run linter & formatter
cargo clippy --workspace
cargo fmt --check
```

---

## 🛡️ Security & Snyk Partnership

RustO! is a community-driven open source project protected and monitored with the support of the **[Snyk Open Source Program](https://snyk.io/open-source/)**. Snyk provides automated developer security tooling to detect, track, and remediate vulnerabilities across our Rust engine, dependencies, container environments, and multi-platform packages.

- **Vulnerability Status**: [![Known Vulnerabilities](https://snyk.io/test/github/byrizki/rusto-rs/badge.svg)](https://snyk.io/test/github/byrizki/rusto-rs)
- **Reporting Vulnerabilities**: If you discover a security issue or vulnerability, please refer to our [Security Policy](SECURITY.md) for responsible disclosure steps.
- **Co-branding Grant**: We hereby grant [Snyk](https://snyk.io/) permission and a non-exclusive license to use, reproduce, and display the RustO! project logo and related project content on Snyk's website and marketing materials in accordance with the Snyk Open Source Program.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

RustO! is inspired by and builds upon the work of:
- **[DataLab](https://github.com/datalab-to)** — Creators of **[Indra OCR](https://github.com/datalab-to/indra-ocr)** models and the **[OmniDocBench](https://github.com/datalab-to/omnidocbench)** evaluation dataset
- **[RapidOCR](https://github.com/RapidAI/RapidOCR)** — Architecture and OCR pipeline reference
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** — Pretrained OCR models (PP-OCRv6, PP-OCRv5, PP-OCRv4)
- **[Alibaba MNN](https://github.com/alibaba/MNN)** — Pioneered efficient mobile deep learning formats, executed natively in pure Rust via `rusto-nn`
- **[RTen](https://github.com/robertknight/rten)** — Pure Rust deep learning inference engine supported in RustO!
- **[Snyk](https://snyk.io)** — Developer security and vulnerability monitoring through the Snyk Open Source Program
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
  note = {Based on RapidOCR and powered by PaddleOCR models with pure Rust MNN (rusto-nn) inference}
}
```
