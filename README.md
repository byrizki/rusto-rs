<div align="center">

# RustO!

Pure-Rust OCR engine, CLI, and multi-platform bindings powered by PaddleOCR and Indra OCR models.

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

**RustO!** runs text detection, recognition, orientation correction, document unwarping, and layout reading without OpenCV or Python dependencies. Inference runs on [`rusto-nn`](crates/rusto-nn) (a pure-Rust MNN runtime with optional WebGPU and CUDA backends) or pluggable [`rusto-rten`](crates/rusto-rten) (`.rten`) and [`rusto-mnn`](crates/rusto-mnn) (optional C++ MNN) sessions.

By default, the CLI and SDKs download and run **PP-OCRv6 Tiny FP32** (`ppocrv6-tiny`, Apache 2.0) on first use. DataLab **Indra OCR** (`indraocr`) and additional PP-OCRv6/v5/v4 tiers are available as opt-in models.

---

## Overview

- **Pure Rust runtime (`rusto-nn`)** — Executes `.mnn`, `.onnx`, and `.rten` models directly via `cargo build` with AVX-512, AVX2, NEON, and WASM SIMD128 kernels. No OpenCV or C++ runtime required unless you opt into `mnn-cpp`.
- **Default `ppocrv6-tiny` (FP32, Apache 2.0)** — ~5.9–6.1 MB base weights covering Simplified Chinese, English, and common Latin scripts (`6,904` characters). Supports FP16 (`ppocrv6-tiny-fp16`, ~3.1 MB on `rusto-nn`) and INT8 (`ppocrv6-tiny-int8`, ~3.3–3.4 MB) alongside larger Small/Medium tiers and optional Indra OCR (`indraocr`).
- **Document pipeline stages** — Optional UVDoc 3D unwarping, whole-page orientation (`0°/90°/180°/270°`), line inversion (`180°`), PicoDet layout detection (`cdla` / `publaynet`), pure-Rust heuristic reading order, and neural reading order (`indra_order.rten`).
- **Consistent builder API** — Configure calls with `.withAutoUnwarp()`, `.withLayoutReading()`, `.asLines()` / `.asWords()` / `.asSpatial()`, and `.run()` across Rust, CLI, and foreign-language bindings.

---

## Packages

| Platform | Package | Notes |
|---|---|---|
| Rust | [`rusto-rs`](https://crates.io/crates/rusto-rs) | Core crate and `rusto` CLI (`2.9 MB` release binary) |
| CLI | `cargo install rusto-rs` | `rusto <image>`, `rusto ocr`, `rusto download` |
| Node.js | [`@rustors/node`](https://www.npmjs.com/package/@rustors/node) | Native N-API addon |
| Python | [`rusto`](https://pypi.org/project/rusto) | UniFFI bindings and pre-built wheels |
| Go | [`packages/go`](https://pkg.go.dev/github.com/byrizki/rusto-rs/packages/go) | Cgo module |
| Java / JVM | [`rusto-java`](https://jitpack.io/#byrizki/rusto-rs) | JNA bindings via JitPack |
| .NET / C# | [`RustODotnet`](https://www.nuget.org/packages/RustODotnet) | Managed bindings via NuGet |
| Dart | [`rusto`](https://pub.dev/packages/rusto) | Pure Dart FFI library |
| Flutter | [`rusto_flutter`](https://pub.dev/packages/rusto_flutter) | Flutter plugin for Android and iOS |
| Web (WASM) | [`@rustors/web`](https://www.npmjs.com/package/@rustors/web) | Browser WASM + Web Worker engine |
| React | [`@rustors/react`](https://www.npmjs.com/package/@rustors/react) | React provider, hooks, and overlays |
| React Native | [`@rustors/react-native`](https://www.npmjs.com/package/@rustors/react-native) | iOS and Android native bridge |
| iOS | `RustO` ([CocoaPods](https://cocoapods.org/pods/RustO) / SPM) | Swift bindings |
| Android | [`rusto-android`](https://jitpack.io/#byrizki/rusto-rs) | Kotlin bindings and AAR |
| C / Native | `librusto.so` / `.dylib` / `.dll` | C FFI (`include/rusto.h`) |

---

## Quick Start

### CLI

```bash
# Runs with ppocrv6-tiny (FP32) by default, downloading weights to ~/.cache/rusto/models on first run
rusto receipt.jpg

# Output reconstructed 2D spatial layout with unwarping and layout analysis
rusto ocr document.jpg --enable-unwarp --enable-layout --format spatial

# Pre-download default or optional models for offline use
rusto download
rusto download --model ppocrv6-tiny-fp16
rusto download --model indraocr
```

### Rust

```rust
use rusto::{DetectTextBuilderExt, ImageSource, InitializeConfig, RustO};

rusto::init();
let mut ocr = RustO::initialize(InitializeConfig::default())?;
let result = ocr
    .detect_text_builder(ImageSource::Path("document.jpg".into()))
    .with_auto_unwarp(true)
    .with_layout_reading(true)
    .as_lines()
    .run()?;
println!("{result:?}");
```

### Node.js

```bash
npm install @rustors/node
```

```ts
import { RustO } from '@rustors/node';

const engine = RustO.initialize();
try {
  const lines = engine
    .detectText('invoice.png')
    .withAutoUnwarp()
    .withLayoutReading()
    .asLines()
    .run();
  for (const r of lines) {
    console.log(`${r.text} (${(r.score * 100).toFixed(1)}%)`);
  }
} finally {
  engine.close();
}
```

### Python

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

### Go

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
    if err != nil {
        panic(err)
    }
    defer engine.Close()

    res, err := engine.DetectText("invoice.png").
        WithAutoUnwarp(true).
        WithLayoutReading(true).
        AsLines().
        Run()
    if err != nil {
        panic(err)
    }

    for _, r := range res.Results {
        fmt.Printf("[%.2f] %s\n", r.Score, r.Text)
    }
}
```

### .NET / C#

```bash
dotnet add package RustODotnet
```

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

### Java (JNA)

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
                System.out.printf("[%.2f] %s%n", r.score(), r.text());
            }
        }
    }
}
```

### Web (WASM) & React

```bash
npm install @rustors/web
```

```ts
import { initialize, detectText, close } from '@rustors/web';

const HF_BASE = 'https://huggingface.co/byrizki/rusto-models/resolve/main';

await initialize({
  preset: 'ppv6',
  models: {
    detection: `${HF_BASE}/ppocrv6-tiny/det.rten`,
    recognition: `${HF_BASE}/ppocrv6-tiny/rec.rten`,
    dictionary: `${HF_BASE}/ppocrv6-tiny/dict.txt`,
  },
});

const results = await detectText(document.querySelector('img')!)
  .withAutoUnwarp()
  .withLayoutReading()
  .asLines()
  .run();

results.forEach((r) => console.log(`${r.text} (${r.score.toFixed(2)})`));
close();
```

For React apps (`npm install @rustors/react @rustors/web`), wrap your tree with `<RustoProvider config={{ preset: 'ppv6', models }}>` and call `useOcr()`. See the [React guide](https://byrizki.com/rusto-rs/en/how-to/use-react) for full components.

### Mobile (React Native, iOS Swift, Android Kotlin, Dart/Flutter)

<details>
<summary>React Native, iOS, Android, and Dart/Flutter snippets</summary>

**React Native** (`npm install @rustors/react-native`):

```ts
import { initialize, detectText } from '@rustors/react-native';

await initialize();
const results = await detectText({ uri: 'file:///path/to/image.jpg' })
  .withAutoUnwarp()
  .withLayoutReading()
  .asLines()
  .run();
```

**iOS (Swift)**:

```swift
let ocr = try RustO.initialize()
let result = try ocr.detectText(image: "receipt.jpg")
    .withAutoUnwarp()
    .withLayoutReading()
    .asLines()
    .run()
```

**Android (Kotlin)**:

```kotlin
RustO.initialize(context).use { ocr ->
    val result = ocr.detectText("/path/to/image.jpg")
        .withAutoUnwarp()
        .withLayoutReading()
        .asLines()
        .run()
}
```

**Dart / Flutter** (`dart pub add rusto` / `flutter pub add rusto_flutter`):

```dart
import 'package:rusto/rusto.dart';

final engine = RustO.initialize();
final result = engine
    .detectText('invoice.png')
    .withAutoUnwarp()
    .withLayoutReading()
    .asLines()
    .run();
engine.close();
```

</details>

---

## Models & Licensing

Pre-converted models are hosted on [Hugging Face Hub (`byrizki/rusto-models`)](https://huggingface.co/byrizki/rusto-models) with CORS enabled and cached locally on first run.

| Model ID | Weight Size | Precisions | License (Weights) | Notes |
|---|---|---|---|---|
| **`ppocrv6-tiny`** *(Default)* | ~5.9–6.1 MB | FP32 | Apache 2.0 | Default balanced accuracy & permissive license across 6,904 characters |
| **`ppocrv6-tiny-fp16`** | ~3.1 MB | FP16 (`.mnn` / `.onnx`) | Apache 2.0 | Half-precision weights for `rusto-nn` and `rusto-webgpu` |
| **`ppocrv6-tiny-int8`** | ~3.3–3.4 MB | INT8 (`.mnn` / `.rten`) | Apache 2.0 | Compact 8-bit weights for constrained environments |
| **`ppocrv6-small`** / `-int8` | ~30.5 MB / ~16.1 MB | FP32 / INT8 | Apache 2.0 | 18,708-character vocabulary, higher density recall |
| **`ppocrv6-medium-int8`** | ~75.9 MB | INT8 | Apache 2.0 | Server-grade capacity |
| **`indraocr`** *(Optional)* | ~5.4 MB | FP32 | Modified OpenRAIL-M | Fast Latin & Indonesian regional model (`datalab-to/indra-ocr`) |
| **`ppocrv5` / `ppocrv4`** | ~21–113 MB | FP32 | Apache 2.0 | Legacy pipelines and regional packs (`latin`, `ar`, `cyrillic`, `ja`, `ko`, etc.) |

### Indra OCR License Note

Upstream [`datalab-to/indra-ocr`](https://github.com/datalab-to/indra-ocr) code is Apache 2.0, while the **model weights** (`indraocr` detection/recognition/orientation and `indra_order` reading order) are licensed under a **modified OpenRAIL-M** license (see the [upstream weight license](https://huggingface.co/datalab-to/indra-ocr/blob/main/LICENSE)):

- Free for research, personal use, and organizations or startups with under \$20M in total funding and revenue.
- Cannot be used in products competing with Datalab's OCR API; commercial licensing above those thresholds is available at [datalab.to/pricing](https://www.datalab.to/pricing).

Because those weight terms are non-permissive for larger commercial API deployments, RustO! uses **PP-OCRv6 Tiny FP32 (`ppocrv6-tiny`, Apache 2.0)** as the default model. Use `--model indraocr` / `{ model: 'indraocr' }` when your use case complies with Datalab's OpenRAIL-M weight license.

---

## Configuration

`InitializeConfig` sets model paths/presets once per engine instance; `DetectTextBuilder` / `OcrRunOptions` configures per-image options:

```rust
use rusto::download::parse_config_with_auto_download;
use rusto::{DetectTextBuilderExt, ImageSource, InitializeConfig, RustO};

rusto::init();

// Auto-download a specific model preset
let config = parse_config_with_auto_download(r#"{"model": "ppocrv6-tiny-fp16"}"#)?;
let mut ocr = RustO::initialize(config)?;

// Or point directly to local .mnn / .onnx / .rten files
let local = InitializeConfig::ppv6("models/det.rten", "models/rec.rten", "models/dict.txt")
    .with_unwarp("models/uvdoc.rten")
    .with_layout("models/layout_cdla.rten");
let mut local_ocr = RustO::initialize(local)?;

let result = local_ocr
    .detect_text_builder(ImageSource::Path("document.jpg".into()))
    .with_auto_unwarp(true)
    .with_layout_reading(true)
    .as_spatial()
    .run()?;
```

---

## Output Types

### `TextResult` — per-line OCR result

```rust
pub struct TextResult {
    pub text: String,          // recognized text
    pub score: f32,            // confidence [0.0 – 1.0]
    pub box_points: Frame,     // four corner points of the text box
}

/// Quad bounding box: top-left, top-right, bottom-right, bottom-left
pub type Frame = [(f32, f32); 4];
```

`DetectTextResult::Structured(Vec<TextResult>)` is returned by default; use `.as_spatial()` to get `DetectTextResult::Spatial(String)` for 2-D reconstructed layout.

### `BlockResult` — hierarchical document blocks

Block-level output groups `TextResult` lines into their enclosing layout regions (title, paragraph, figure caption, etc.) when a layout model is active, or falls back to a single `Unassigned` block.

```rust
pub struct BlockResult {
    pub layout_type: LayoutType,    // region class: Title, Text, Figure, Table, …
    pub lines: Vec<TextResult>,     // lines within this block
}

impl BlockResult {
    /// All text in this block joined with newlines
    pub fn text(&self) -> String { … }
}

pub enum LayoutType {
    Title, Text, Figure, FigureCaption,
    Table, TableCaption, TableFootnote,
    Header, Footer, Reference, Equation,
    Unassigned,
}
```

**Usage:**

```rust
// via builder (recommended)
let blocks = ocr
    .detect_text_builder(ImageSource::Path("document.pdf".into()))
    .with_layout_reading(true)
    .run_blocks()?;                     // -> Vec<BlockResult>

for block in &blocks {
    println!("[{:?}] {}", block.layout_type, block.text());
}

// or from an existing DetectTextResult
let result = ocr.detect_text(&source, &options)?;
let blocks = result.to_blocks();
```

---

## Recognition Parallelism

RustO runs text recognition in parallel worker threads. The defaults (`rec_workers = auto`, `max_rec_workers = 8`) are tuned for desktop/server use. For **embedded or memory-constrained devices** pin the count explicitly.

### Via `InitializeConfig` (engine-wide default)

```rust
let config = InitializeConfig::ppv6("det.mnn", "rec.mnn", "dict.txt")
    .with_rec_workers(2)        // use exactly 2 worker threads
    .with_max_rec_workers(4);   // cap auto-scaling at 4

let mut ocr = RustO::initialize(config)?;
```

### Via `DetectTextBuilder` (per-image override)

```rust
let result = ocr
    .detect_text_builder(ImageSource::Path("page.jpg".into()))
    .with_rec_workers(1)        // single-threaded recognition for this image
    .run()?;
```

### Via JSON config

```json
{
  "model": "ppocrv6-tiny",
  "rec_workers": 2,
  "max_rec_workers": 4
}
```

or inside a `recognition` block:

```json
{
  "model": "ppocrv6-tiny",
  "recognition": { "rec_workers": 2, "max_rec_workers": 4 }
}
```

### Via CLI

```sh
rusto image.jpg --rec-workers 2 --max-rec-workers 4
```

> **Embedded / IoT tip:** Set `rec_workers = 1` and `max_rec_workers = 1` to eliminate thread-pool overhead entirely and keep latency deterministic. For HTTP server pools, hold multiple `RustO` instances (e.g. 4–8) instead of wrapping one in `Arc<Mutex<RustO>>` — a mutex serialises requests and causes tail-latency spikes.

---

## OmniDocBench Benchmarks

Measured on [OmniDocBench](benchmarks/omnidocbench/OMNIDOCBENCH_EVALUATION_REPORT.md) (`11th Gen Intel Core i7-1185G7 @ 3.00GHz`, 8 logical cores, 37.9 GB RAM, Linux x86_64):

### Base Text Detection & Recognition (1D Spatial Sorting)

| Configuration | Runtime | Binary | Weights | Mean Latency | Peak RSS | Accuracy (1 - NED) |
|---|---|---:|---:|---:|---:|---:|
| **RustO NN + PP-OCRv6 Tiny (FP32, Default)** | `rusto-nn` | **2.9 MB** | 5.9 MB | 1,127.7 ms | 307.9 MB | **46.53%** |
| **RustO NN + PP-OCRv6 Tiny (FP16)** | `rusto-nn` | **2.9 MB** | **3.1 MB** | 1,142.8 ms | 307.5 MB | **46.57%** |
| **RustO NN + PP-OCRv6 Tiny (INT8)** | `rusto-nn` | **2.9 MB** | 3.3 MB | 1,821.8 ms | 377.5 MB | **46.74%** |
| **RustO RTen + PP-OCRv6 Tiny (FP32)** | `rusto-rten` | 6.7 MB | 6.1 MB | 996.2 ms | **193.4 MB** | 45.43% |
| **RustO RTen + PP-OCRv6 Tiny (INT8)** | `rusto-rten` | 6.7 MB | 3.4 MB | 1,113.5 ms | **177.2 MB** | 45.52% |
| **RustO NN + Indra OCR** *(opt-in)* | `rusto-nn` | **2.9 MB** | 5.4 MB | **582.9 ms** | 188.0 MB | 45.70% |
| **RustO RTen + Indra OCR** *(opt-in)* | `rusto-rten` | 6.7 MB | 5.4 MB | 728.7 ms | **155.5 MB** | 45.69% |
| Python PaddleOCR + PP-OCRv6 Tiny | PaddlePaddle | 120.1 MB | 6.1 MB | 628.7 ms | 696.2 MB | 45.33% |
| Python RapidOCR + PP-OCRv6 Tiny | ONNX Runtime | 22.7 MB | 6.1 MB | 1,030.1 ms | 395.8 MB | 44.84% |
| Tesseract v5 (PSM 6) | C++ / Leptonica | 4.2 MB | 3.9 MB | 1,755.5 ms | 174.4 MB | 41.14% |

### Full Pipeline (Layout & Reading Order Recovery)

| Configuration | Mean Latency | Peak RSS | Accuracy (1 - NED) |
|---|---:|---:|---:|
| **RustO NN + Pure-Rust Heuristic Order (`ppocrv6-tiny`)** | 1,634.8 ms | 454.3 MB | **62.72%** |
| **RustO RTen + Pure-Rust Heuristic Order (`ppocrv6-tiny`)** | 1,529.5 ms | 266.8 MB | **60.98%** |
| **RustO NN + CDLA Full Pipeline (`ppocrv6-tiny`)** | 2,332.4 ms | 503.8 MB | **57.91%** |
| **RustO NN + Indra Order** *(opt-in)* | **599.9 ms** | **184.6 MB** | **63.90%** |
| Python Indra OCR (Official CLI) | 3,111.6 ms | 472.2 MB | 63.76% |
| Python PaddleOCR + PP-Structure (PicoDet) | 20,676.5 ms | 929.6 MB | 57.18% |
| Python RapidOCR + RapidLayout Tiny | 1,561.7 ms | 453.5 MB | 53.85% |
| Tesseract v5 (PSM 3) | 1,847.7 ms | 184.6 MB | 60.29% |

See the full methodology and P95 metrics in [benchmarks/omnidocbench/OMNIDOCBENCH_EVALUATION_REPORT.md](benchmarks/omnidocbench/OMNIDOCBENCH_EVALUATION_REPORT.md).

---

## Documentation

Full guides and API docs live at **[byrizki.com/rusto-rs](https://byrizki.com/rusto-rs/)** (`docs/`):

- [Models & Tiers](https://byrizki.com/rusto-rs/en/advance-guide/models-and-tiers) — Model matrix, `.mnn`/`.onnx`/`.rten` formats, FP16/INT8 quantization
- [Multilingual OCR](https://byrizki.com/rusto-rs/en/advance-guide/multilingual-ocr) — Language codes and standalone recognizer packs
- [Multi-Column, Unwarp & Orientation](https://byrizki.com/rusto-rs/en/advance-guide/multi-column-unwarp-orientation) — Reading order, UVDoc, layout regions
- [Spatial Layout Reconstruction](https://byrizki.com/rusto-rs/en/advance-guide/spatial-layout-reconstruction) — 2D layout spacing for receipts and tables
- [InitializeConfig API Reference](https://byrizki.com/rusto-rs/en/api-reference/initialize-config) & [OcrRunOptions](https://byrizki.com/rusto-rs/en/api-reference/ocr-run-options)
- [v0.3 → v0.4 Migration Guide](https://byrizki.com/rusto-rs/en/advance-guide/migration-guide)

---

## Repository Layout

```text
rusto-rs/
├── crates/
│   ├── rusto-nn/           # Pure-Rust MNN runtime + optional WebGPU / CUDA backends
│   ├── rusto-core/         # DBNet, CTC, UVDoc unwarp, PicoDet layout, reading order
│   ├── rusto-rten/         # RTen (.rten) session backend
│   ├── rusto-mnn/          # Optional C++ MNN session backend
│   ├── rusto-node/         # Node.js N-API crate
│   └── rusto-wasm/         # WebAssembly crate
├── src/                    # Workspace root crate, UniFFI/C FFI bindings, and `rusto` CLI
├── include/                # Canonical C header (`rusto.h`)
├── packages/               # Language SDK packages (node, python, go, java, dotnet, dart, flutter, web, react, react-native, android, ios)
├── docs/                   # Bilingual Docus docs site and live WASM demo
└── scripts/                # UniFFI binding generation and release tooling
```

---

## Development & Testing

```bash
# Run workspace library tests
cargo test --workspace --lib

# Build `rusto` CLI (opt-level=3 by default)
cargo build --release --bin rusto

# Build C FFI + RTen shared/static library
cargo build --release --locked --features ffi,rten --lib

# Size-optimized C FFI profile (opt-level="z")
cargo build --profile compressed --locked --features ffi,rten --lib

# Lint and check formatting
cargo clippy --workspace
cargo fmt --check
```

---

## Security & Snyk Partnership

RustO! is a community-driven open source project protected and monitored with the support of the **[Snyk Open Source Program](https://snyk.io/open-source/)**. Snyk provides automated developer security tooling to detect, track, and remediate vulnerabilities across our Rust engine, dependencies, container environments, and multi-platform packages.

- **Vulnerability Status**: [![Known Vulnerabilities](https://snyk.io/test/github/byrizki/rusto-rs/badge.svg)](https://snyk.io/test/github/byrizki/rusto-rs)
- **Reporting Vulnerabilities**: See our [Security Policy](SECURITY.md) for responsible disclosure steps.
- **Co-branding Grant**: We hereby grant [Snyk](https://snyk.io/) permission and a non-exclusive license to use, reproduce, and display the RustO! project logo and related project content on Snyk's website and marketing materials in accordance with the Snyk Open Source Program.

---

## License

RustO! source code is licensed under the [MIT License](LICENSE). Model weights carry their respective upstream licenses (Apache 2.0 for PaddleOCR models; modified OpenRAIL-M for DataLab Indra OCR weights).

---

## Acknowledgments

- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** — PP-OCRv6, PP-OCRv5, and PP-OCRv4 models
- **[DataLab](https://github.com/datalab-to)** — [Indra OCR](https://github.com/datalab-to/indra-ocr) models and the [OmniDocBench](https://github.com/datalab-to/omnidocbench) evaluation dataset
- **[Alibaba MNN](https://github.com/alibaba/MNN)** — Model serialization format targeted in pure Rust by `rusto-nn`
- **[RTen](https://github.com/robertknight/rten)** — Pure-Rust neural inference runtime (`rusto-rten`)
- **[RapidOCR](https://github.com/RapidAI/RapidOCR)** — Reference pipeline design
- **[Snyk](https://snyk.io)** — Open-source vulnerability scanning

---

## Citation

```bibtex
@software{rusto2024,
  title = {RustO! - Pure Rust OCR Library},
  author = {Rizki & Contributors},
  year = {2024},
  url = {https://github.com/byrizki/rusto-rs},
  note = {Pure-Rust OCR engine running PaddleOCR and Indra OCR models via rusto-nn and RTen}
}
```
