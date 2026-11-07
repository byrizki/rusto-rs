# RustO! Examples

This directory (alongside SDK `example/` apps under `packages/`) showcases **RustO! v0.4.0** across Rust, .NET, React Native, Flutter, and Dart using standalone models and zero-config **Indra OCR** auto-downloading from Hugging Face Hub.

## Overview of Example Projects

| Example | Location | Description |
| :--- | :--- | :--- |
| **Rust Core Examples** | `examples/*.rs` | Direct `rusto` Rust crate examples (calibration/optimization, spatial layout, document layout + table pipeline, performance benchmarks) |
| **.NET Console & Smoke Suite** | [`examples/dotnet`](./dotnet/README.md) | C# / .NET consumer app demonstrating `RustO.Initialize()`, fluent `.AsLines().Run()` / `.AsSpatial().Run()`, and offline models |
| **React Native (New Architecture)** | [`examples/react-native-new-arch`](./react-native-new-arch/README.md) | React Native TurboModule / Fabric verification app using `@rustors/react-native` |
| **React Native (Legacy Bridge)** | [`examples/react-native-legacy`](./react-native-legacy/README.md) | React Native Classic Architecture verification app using `@rustors/react-native` |
| **React Native Bare / Expo Dev Build** | `packages/react-native/example` | Interactive bare React Native showcase (also compatible with Expo Development Builds via `npx expo prebuild`) |
| **Flutter Mobile & Desktop App** | [`packages/flutter/example`](../packages/flutter/example) | Full Flutter verification app demonstrating `rusto_flutter`, bundled assets, structured detection, and 2D spatial layout |
| **Dart CLI Example** | [`packages/dart/example`](../packages/dart/example) | Standalone Dart FFI command-line example using `RustO.initialize()` and fluent `.asLines().run()` / `.asSpatial().run()` |

---

## Model Assets (Zero-Config Auto-Download via Hugging Face)

In **RustO! v0.4.0**, all high-level SDKs default to the **Indra OCR** (`indraocr`) pipeline and automatically download missing models into your local cache directory (`~/.cache/rusto/models` or `$RUSTO_CACHE_DIR`) on first run from Hugging Face Hub.

To pre-download standalone models locally for offline examples or Rust CLI demos:

```bash
bash docs/scripts/download-models.sh
```

Or download specific model files directly from [Hugging Face (`byrizki/rusto-models`)](https://huggingface.co/byrizki/rusto-models).

---

## Running the Examples

### 1. Rust Core Examples (`examples/*.rs`)

Run from the repository root:

```bash
# PP-OCRv6 Tiny with built-in calibration & optimization (.rten models)
cargo run --release --example optimized_ocr_demo

# 2D spatial layout reconstruction
cargo run --release --example spatial_export

# Full document layout & Markdown export demo
cargo run --release --example doc_pipeline_demo -- \
  --image path/to/document.jpg \
  --layout-model models/layout/layout_cdla.rten \
  --det-model models/indra/indra_det.rten \
  --rec-model models/indra/indra_rec.rten \
  --keys-path models/indra/indra_meta.json
```

Other Rust examples in `examples/`: `ocr_with_fixes.rs`, `ocr_without_fixes.rs`, `test_layout_simple.rs`, `perf_test.rs`, and `raw_ocr.rs`.

### 2. .NET Consumer Example (`examples/dotnet`)

```bash
# Build native FFI library first when testing from source
cargo build --release -p rusto-ffi

cd examples/dotnet
dotnet run --project RustOConsumer
```

Inside C#, the consumer uses the v0.4.0 fluent API:

```csharp
using var ocr = RustO.Initialize();
var lines = ocr.DetectText("invoice.png").WithAutoUnwarp().WithLayoutReading().AsLines().Run();
var spatial = ocr.DetectText("invoice.png").AsSpatial().Run();
```

### 3. React Native Bare & Architecture Examples

- **Bare Showcase (`packages/react-native/example`)**:
  ```bash
  cd packages/react-native/example
  yarn install
  yarn android   # or: cd ios && pod install && cd .. && yarn ios
  ```
- **New Architecture (`examples/react-native-new-arch`) & Legacy (`examples/react-native-legacy`)**:
  See [`examples/react-native-new-arch/README.md`](./react-native-new-arch/README.md) and [`examples/react-native-legacy/README.md`](./react-native-legacy/README.md) for automated Android/iOS E2E verification scripts.
- **Expo Projects (Development Builds)**:
  Because `@rustors/react-native` links compiled Rust native libraries (`rusto-android` AAR & `RustO.xcframework`), Expo Go is not supported. In an Expo app, install `@rustors/react-native`, generate native directories with `npx expo prebuild`, and launch with `npx expo run:android` or `npx expo run:ios` — zero-config `await initialize()` works out of the box.

### 4. Flutter Example (`packages/flutter/example`)

```bash
cd packages/flutter/example
flutter pub get
flutter run
```

Demonstrates `RustO.initialize(...)` from `package:rusto_flutter/rusto_flutter.dart`, bundled `assets/models/*.rten` extraction, `.asLines().run()`, `.asSpatial().run()`, and `engine.close()`.

### 5. Dart CLI Example (`packages/dart/example`)

```bash
cargo build --release -p rusto-ffi
cd packages/dart/example
dart pub get
dart run main.dart
```
