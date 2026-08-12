# RustO! 🦀

**Pure Rust OCR Library** - Fast, Safe, and Cross-Platform

[![Crates.io](https://img.shields.io/crates/v/rusto-rs.svg)](https://crates.io/crates/rusto-rs)
[![NuGet](https://img.shields.io/nuget/v/RustODotnet.svg)](https://www.nuget.org/packages/RustODotnet)
[![npm](https://img.shields.io/npm/v/react-native-rusto.svg)](https://www.npmjs.com/package/react-native-rusto)
[![CocoaPods](https://img.shields.io/cocoapods/v/RustO.svg)](https://cocoapods.org/pods/RustO)
[![Maven Central](https://img.shields.io/maven-central/v/com.byrizki.rusto/rusto-android.svg)](https://central.sonatype.com/artifact/com.byrizki.rusto/rusto-android)
[![Documentation](https://docs.rs/rusto-rs/badge.svg)](https://docs.rs/rusto-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/byrizki/rusto-rs/workflows/CI/badge.svg)](https://github.com/byrizki/rusto-rs/actions)

RustO! is a high-performance OCR (Optical Character Recognition) library written in pure Rust, based on [RapidOCR](https://github.com/RapidAI/RapidOCR) and powered by [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) models with MNN inference engine.

## 🎯 Why RustO!?

- **🚀 Pure Rust** - Zero OpenCV dependency, optional OpenCV backend available
- **🎯 High Accuracy** - 99.3% parity with OpenCV-based implementations
- **⚡ Fast Performance** - Optimized with LTO, single codegen unit compilation
- **🔒 Memory Safe** - Leverages Rust's safety guarantees
- **🌐 Cross-Platform** - Linux, macOS, Windows, iOS, Android support
- **🔧 FFI Ready** - C FFI bindings for integration with other languages
- **📦 Easy to Use** - Simple API, modern CLI with JSON/Text/TSV output

## 🏗️ Architecture

RustO! is built on top of proven OCR technology:

- **Based on**: [RapidOCR](https://github.com/RapidAI/RapidOCR) architecture
- **Models**: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) PP-OCRv6 (Default), PP-OCRv5, PP-OCRv4, and PP-OCRv3 models
- **Inference**: [MNN](https://github.com/alibaba/MNN) inference engine for high-performance cross-platform execution on mobile, desktop, and server
- **Image Processing**: Pure Rust implementation (image + imageproc crates)
- **Contour Detection**: Custom Rust implementation matching OpenCV behavior

## 📁 Project Structure

```
rusto-rs/
├── src/
│   ├── lib.rs          # Public API & exports
│   ├── config.rs       # RustOConfig, presets (PPV6, PPV5, PPV4, PPV3), & builders
│   ├── main.rs         # CLI application
│   ├── ffi.rs          # C FFI bindings
│   ├── det.rs          # Text detection (DBNet)
│   ├── rec.rs          # Text recognition (CTC)
│   ├── orient.rs       # Document orientation classification
│   ├── layout.rs       # Layout detection
│   ├── table.rs        # Table recognition & HTML structure
│   ├── doc_pipeline.rs # Document pipeline (layout + OCR)
│   ├── preprocess.rs   # Image preprocessing
│   ├── postprocess.rs  # Result postprocessing
│   ├── contours.rs     # Pure Rust contour detection
│   ├── geometry.rs     # Geometric transformations + NMS
│   ├── image_impl.rs   # Image abstraction layer
│   └── types.rs        # Type definitions, Frame, & Config structures
├── models/
│   ├── PPOCR_v6/       # PP-OCRv6 MNN models (Tiny prebundled, Small, Medium)
│   └── PPOCR_v5/       # PP-OCRv5 MNN models
├── packages/
│   ├── react-native/   # React Native TypeScript + iOS/Android bindings
│   ├── android/        # Android library (Kotlin/JNI)
│   ├── ios/            # iOS Swift package / CocoaPod
│   └── dotnet/         # .NET C# NuGet package
└── ...
```

---

## Quick Start

### 1. Build the Library

```bash
# Pure Rust build (default)
cargo build --release

# With FFI bindings
cargo build --release --features ffi

# With OpenCV backend (optional)
cargo build --release --features use-opencv
```

### 2. Run CLI Application

```bash
# JSON output (default)
cargo run --release -- \
  --det-model models/PPOCR_v6/det.mnn \
  --rec-model models/PPOCR_v6/rec.mnn \
  --dict models/PPOCR_v6/dict.txt \
  image.jpg

# Plain text output
cargo run --release -- \
  --det-model models/PPOCR_v6/det.mnn \
  --rec-model models/PPOCR_v6/rec.mnn \
  --dict models/PPOCR_v6/dict.txt \
  --format text \
  image.jpg

# TSV output
cargo run --release -- \
  --det-model models/PPOCR_v6/det.mnn \
  --rec-model models/PPOCR_v6/rec.mnn \
  --dict models/PPOCR_v6/dict.txt \
  --format tsv \
  image.jpg
```

### 3. Use as a Rust Library

Add to your `Cargo.toml`:

```toml
[dependencies]
rusto = "0.1"
```

Then in your code:

```rust
use rusto::{RustO, RustOConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure OCR with default PP-OCRv6 preset
    let config = RustOConfig::new(
        "models/PPOCR_v6/det.mnn",
        "models/PPOCR_v6/rec.mnn",
        "models/PPOCR_v6/dict.txt",
    )
    .with_text_score(0.5)
    .with_xy_threshold(0.5, 1.0); // Configure spatial text spacing
    
    // Create OCR instance
    let mut ocr = RustO::new(config)?;
    
    // Run OCR on an image
    let output = ocr.run("image.jpg")?;
    
    // 1. Get structured text results with axis-aligned bounding frames
    let results = output.to_text_results();
    for res in results {
        println!("Text: '{}' (Score: {:.2})", res.text, res.score);
        println!("  Frame: left={:.1}, top={:.1}, w={:.1}, h={:.1}", 
            res.frame.left, res.frame.top, res.frame.width, res.frame.height);
        println!("  Polygon: {:?}", res.box_points);
    }
    
    // 2. Reconstruct spatial document layout text
    let spatial_text = output.to_spatial_text(None, None);
    println!("Spatial Layout:\n{}", spatial_text);
    
    Ok(())
}
```

### 4. Template Presets & Architecture Support

RustO! provides pre-configured template presets for different PaddleOCR model generations:

```rust
use rusto::{RustOConfig, PPV6_MODEL_CONFIG, PPV5_MODEL_CONFIG, PPV4_MODEL_CONFIG, PPV3_MODEL_CONFIG};

// PP-OCRv6 (Default): limit_side_len=736, min, det_thresh=0.3, det_box_thresh=0.6, unclip=2.0
let v6_config = RustOConfig::ppv6("det.mnn", "rec.mnn", "dict.txt");

// PP-OCRv5: limit_side_len=736, min, det_thresh=0.3, det_box_thresh=0.5, unclip=2.0
let v5_config = RustOConfig::ppv5("det.mnn", "rec.mnn", "dict.txt");

// PP-OCRv4: limit_side_len=960, max, det_thresh=0.3, det_box_thresh=0.6, unclip=1.5
let v4_config = RustOConfig::ppv4("det.mnn", "rec.mnn", "dict.txt");

// PP-OCRv3: limit_side_len=960, max, det_thresh=0.3, det_box_thresh=0.6, unclip=1.5
let v3_config = RustOConfig::ppv3("det.mnn", "rec.mnn", "dict.txt");
```

### 5. Cross-Platform SDKs

#### React Native
```typescript
import { initialize, detectText, detectTextToSpatialText } from 'react-native-rusto';

// Initialize with bundled default PP-OCRv6 tiny models (no parameters needed!)
await initialize();

// Detect text with bounding frames
const results = await detectText('/path/to/image.jpg');
results.forEach((r) => {
  console.log(`${r.text} (${r.score}) - Frame:`, r.frame); // { width, height, top, left }
});

// Or format directly to spatial layout text
const spatialText = await detectTextToSpatialText('/path/to/image.jpg', 0.5, 1.0);
console.log(spatialText);
```

#### iOS (Swift)
```swift
import RustO

// Default PP-OCRv6 configuration
let config = RustOConfig.ppv6(
    det: "det.mnn",
    rec: "rec.mnn",
    dict: "dict.txt"
)
let ocr = try RustO(config: config)

let results = try ocr.recognizeFile("image.jpg")
for result in results {
    print("\(result.text) (\(result.score)): frame=\(result.frame.left),\(result.frame.top),\(result.frame.width)x\(result.frame.height)")
}
```

#### Android (Kotlin)
```kotlin
import com.byrizki.rusto.RustO
import com.byrizki.rusto.RustOConfig

val config = RustOConfig(
    template = "ppv6",
    detModelPath = "det.mnn",
    recModelPath = "rec.mnn",
    dictPath = "dict.txt"
)
val ocr = RustO(context, config)
val results = ocr.recognizeFile("/path/to/image.jpg")
```

#### .NET (C#)
```csharp
using RustODotnet;

var config = RustOConfig.Ppv6("det.mnn", "rec.mnn", "dict.txt");
using var ocr = new RustO(config);
var results = ocr.RecognizeFile("image.jpg");
```

---

## API Reference

### RustOConfig & Builders

Comprehensive configuration structure supporting granular parameter overrides:

```rust
let config = RustOConfig::ppv6("det.mnn", "rec.mnn", "dict.txt")
    // Detection tuning
    .with_det_thresh(0.3)
    .with_det_box_thresh(0.6)
    .with_limit_side_len(736)
    .with_limit_type("min")
    .with_unclip_ratio(2.0)
    .with_use_dilation(true)
    .with_max_candidates(1000)
    .with_score_mode("fast")
    // Recognition tuning
    .with_rec_img_shape([3, 48, 320])
    .with_rec_batch_num(6)
    // Global & Spatial tuning
    .with_text_score(0.5)
    .with_xy_threshold(0.5, 1.0)
    .with_min_height(30.0)
    .with_max_side_len(2000.0)
    // Optional modules
    .with_cls("models/cls.mnn", 0.9)
    .with_orientation("models/orient.mnn", 0.9)
    .with_unwarp("models/unwarp.mnn");
```

### Frame & TextResult

```rust
pub struct Frame {
    pub width: f32,
    pub height: f32,
    pub top: f32,
    pub left: f32,
}

pub struct TextResult {
    pub text: String,                    // Recognized text string
    pub score: f32,                      // Confidence score (0.0 - 1.0)
    pub box_points: [(f32, f32); 4],    // 4 rotated polygon corner points
    pub frame: Frame,                    // Axis-aligned bounding frame
}
```

---

## 📦 Models

RustO! uses lightweight, high-performance PaddleOCR models in MNN format:

### Model Series Supported
- **PP-OCRv6** (Default & Recommended) — MetaFormer-based PPLCNetV4 architecture with 50-language unified dictionary. Available in **Tiny** (prebundled, 6.0 MB total), **Small**, and **Medium** tiers.
- **PP-OCRv5** — High-accuracy detection with SVTR-LCNet recognition.
- **PP-OCRv4** — Lightweight mobile OCR models.
- **PP-OCRv3** — Legacy mobile OCR models.

### Downloading Pre-Converted MNN Models
Official models are hosted on [ModelScope RapidAI/RapidOCR](https://www.modelscope.cn/models/RapidAI/RapidOCR/tree/master/mnn/PP-OCRv6):

```bash
# PP-OCRv6 Tiny (Prebundled default)
curl -L -o models/PPOCR_v6/det.mnn "https://www.modelscope.cn/api/v1/models/RapidAI/RapidOCR/repo?Revision=master&FilePath=mnn%2FPP-OCRv6%2Fdet%2FPP-OCRv6_det_tiny.mnn"
curl -L -o models/PPOCR_v6/rec.mnn "https://www.modelscope.cn/api/v1/models/RapidAI/RapidOCR/repo?Revision=master&FilePath=mnn%2FPP-OCRv6%2Frec%2FPP-OCRv6_rec_tiny.mnn"
curl -L -o models/PPOCR_v6/dict.txt "https://www.modelscope.cn/api/v1/models/RapidAI/RapidOCR/repo?Revision=master&FilePath=paddle%2FPP-OCRv6%2Frec%2FPP-OCRv6_rec_tiny%2Fppocrv6_tiny_dict.txt"
```

---

## 🔌 C FFI & Shared Libraries

RustO! provides a high-performance C FFI interface for building desktop, mobile, and native bindings. Enable with the `ffi` feature:

```bash
cargo build --release --features ffi
```

This compiles shared libraries:
- **Linux**: `target/release/librusto.so`
- **macOS / iOS**: `target/release/librusto.dylib`
- **Windows**: `target/release/rusto.dll`

FFI APIs include `rocr_new_with_config(config_json)`, `rocr_run(inst, image_path)`, `rocr_run_to_spatial_text(inst, image_path, y_multiplier, x_multiplier)`, and direct memory pointer interfaces.

---

## ⚡ Performance

### Benchmarks

Tested on typical document images:

| Metric | Value |
|--------|-------|
| Detection | ~80ms |
| Recognition (per box) | ~120ms |
| Total (28 boxes) | ~3.5s |
| Memory Peak | ~200MB |

### Comparison with OpenCV-based implementations

| Aspect | RustO! | OpenCV-based |
|--------|--------|--------------|
| Speed | ✅ Similar (±10%) | Baseline |
| Accuracy | ✅ 99.3% parity | 100% |
| Binary Size | ✅ **Smaller** | Larger (OpenCV deps) |
| Memory Usage | ✅ **Lower** | Higher (OpenCV overhead) |
| Dependencies | ✅ **Minimal** | OpenCV required |
| Safety | ✅ **Memory safe** | Manual memory management |

---

## Configuration

### Cargo Features
```toml
[features]
default = []           # Pure Rust mode
use-opencv = ["opencv"] # Use OpenCV backend
ffi = []               # Enable C FFI bindings
```

### Build Profiles
```toml
[profile.release]
opt-level = 3          # Maximum optimization
lto = "fat"            # Link-time optimization
codegen-units = 1      # Single codegen unit for better optimization
strip = true           # Strip symbols
panic = "abort"        # Smaller binary
```

---

## Development

### Run Tests
```bash
cd rapidocr
cargo test
cargo test --features use-opencv  # Test OpenCV backend
```

### Run Benchmarks
```bash
cargo bench
```

### Check Code
```bash
cargo clippy
cargo fmt --check
```

---

## Known Issues

### Rust Library (contours.rs)
- ⚠️ Unused functions (400+ lines) - cleanup pending
- ⚠️ Minor lint warnings - non-blocking

### Remaining Parity Gap (0.7%)
- 2 minor text differences out of 28 boxes
- Caused by: Spacing (`"Gol. Darah:"` vs `"Gol. Darah :"`)
- Impact: Negligible for production use

---

## License

MIT (or your license)

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `cargo test`
5. Submit a pull request

---

## Support

- 📧 Email: support@rapidocr.com
- 💬 Discussions: GitHub Discussions
- 🐛 Issues: GitHub Issues

---

## 🙏 Acknowledgments

RustO! builds upon the excellent work of:

- **[RapidOCR](https://github.com/RapidAI/RapidOCR)** - Architecture and design inspiration
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** - State-of-the-art OCR models (PPOCRv4/v5)
- **[ONNX Runtime](https://github.com/microsoft/onnxruntime)** - Cross-platform inference engine
- **Rust Community** - Excellent tooling and libraries (image, imageproc, nalgebra)

## 📝 Citation

If you use RustO! in your research or project, please cite:

```bibtex
@software{rusto2024,
  title = {RustO! - Pure Rust OCR Library},
  author = {byrizki},
  year = {2024},
  url = {https://github.com/byrizki/rusto-rs},
  note = {Based on RapidOCR and powered by PaddleOCR models}
}
```

Also consider citing the underlying technologies:

- **PaddleOCR**: [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- **RapidOCR**: [https://github.com/RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR)

---

<div align="center">

**Status**: Production Ready 🚀  
**Version**: 0.1.6            
**License**: MIT

Made with ❤️ and 🦀 Rust

[Report Bug](https://github.com/byrizki/rusto-rs/issues) · [Request Feature](https://github.com/byrizki/rusto-rs/issues) · [Contribute](https://github.com/byrizki/rusto-rs/pulls)

</div>
