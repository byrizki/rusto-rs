# rusto (Dart SDK)

Official Dart FFI bindings for **RustO!** — a high-performance, pure-Rust OCR engine powered by Indra OCR, PaddleOCR, and the RTen inference runtime.

> Building a Flutter mobile or desktop application? Use [`rusto_flutter`](../flutter/README.md), which bundles native mobile library packaging and re-exports `package:rusto/rusto.dart`.

## Features

- **Zero-Config Indra OCR**: `RustO.initialize()` automatically downloads and caches the default `indraocr` `.rten` models on first run.
- **Pure-Rust FFI**: Low-overhead synchronous UniFFI bindings with automatic `librusto` discovery and `RUSTO_LIB_PATH` override support.
- **Document Intelligence**: UVDoc document unwarping (`withAutoUnwarp()`), 4-way page orientation correction (`withOrientationDetection()`), and neural layout reading-order recovery (`withLayoutReading()`).
- **Fluent Builder API**: Chain `.asLines()`, `.asWords()`, `.asSpatial()`, or `.withOutput(...)` into a single `.run()` terminal call.
- **Flexible Image Sources**: File paths, `Uint8List` encoded bytes, `ImageSource`, or raw 24-bit interleaved RGB buffers via `.rgb(...)`.

## Installation

Add `rusto` to your `pubspec.yaml`:

```yaml
dependencies:
  rusto: ^0.4.0
```

Build or install the native `librusto` library (`librusto.so`, `librusto.dylib`, or `rusto.dll`). If `librusto` is located outside standard system or `target/release` directories, set `RUSTO_LIB_PATH` or call `RustO.configureBindings(libraryPath: ...)`.

## Quick Start (Zero-Config)

```dart
import 'package:rusto/rusto.dart';

void main() {
  // Initializes Indra OCR and auto-downloads .rten models if not cached
  final engine = RustO.initialize();

  try {
    // Structured line-level recognition with unwarping & layout reading order
    final res = engine
        .detectText('invoice.png')
        .withAutoUnwarp()
        .withLayoutReading()
        .asLines()
        .run();

    if (res is DetectTextResultStructured) {
      for (final item in res.results) {
        print(
          '[${item.score.toStringAsFixed(2)}] '
          '(${item.layoutType ?? 'line'} #${item.regionId ?? '-'}) ${item.text}',
        );
      }
    }

    // Reconstructed 2D spatial layout
    final spatial = engine.detectText('invoice.png').asSpatial().run();
    if (spatial is DetectTextResultSpatial) {
      print(spatial.text);
    }
  } finally {
    engine.close();
  }
}
```

## Custom Presets & Offline `.rten` Models

Pass a structured `InitializeConfig` helper or JSON configuration string to `RustO.initialize(...)`:

```dart
import 'dart:convert';
import 'package:rusto/rusto.dart';

// 1. Canonical model + language (auto-downloads .rten files if missing)
final tinyCfg = createInitializeConfigForModel('ppocrv6-tiny', 'latin', null);
final tinyEngine = RustO.initialize(tinyCfg);
tinyEngine.close();

// 2. Explicit local .rten assets
final offlineJson = jsonEncode({
  'template': 'indra',
  'enableReadingOrder': true,
  'detection': {
    'enabled': true,
    'modelPath': './models/indra_det.rten',
  },
  'recognition': {
    'enabled': true,
    'modelPath': './models/indra_rec.rten',
    'dictPath': './models/indra_meta.json',
  },
  'unwarp': {
    'enabled': true,
    'modelPath': './models/uvdoc.rten',
  },
  'layout': {
    'enabled': true,
    'modelPath': './models/layout_cdla.rten',
  },
});

final offlineEngine = RustO.initialize(offlineJson);
offlineEngine.close();
```

Pre-download `.rten` models ahead of time:

```dart
final cachedPaths = RustO.downloadModel(
  model: 'indraocr',
  lang: 'latin',
  cacheDir: './models-cache',
);
```

## Encoded Bytes & Raw RGB Buffers

Use `detectTextBuilder()` (or `detectText(...)`) for strongly-typed chaining with `Uint8List` image payloads or raw 24-bit RGB pixels:

```dart
import 'dart:io';
import 'dart:typed_data';
import 'package:rusto/rusto.dart';

void runInputs(RustO engine) {
  // 1. Encoded PNG/JPEG bytes
  final imageBytes = File('receipt.png').readAsBytesSync();
  final wordRes = engine
      .detectTextBuilder()
      .data(imageBytes)
      .asWords()
      .run();

  // 2. Raw interleaved 24-bit RGB buffer
  const width = 1280;
  const height = 720;
  const stride = width * 3;
  final rgbPixels = Uint8List(stride * height);

  final rgbRes = engine
      .detectTextBuilder()
      .rgb(rgbPixels, width, height, stride)
      .asLines()
      .run();
}
```

## 6-Phase Pipeline Options

Configure individual pipeline stages on `DetectTextBuilder`:

```dart
final res = engine
    .detectTextBuilder()
    .image('scan.jpg')
    .withOrientationDetection()
    .withAutoUnwarp()
    .withLayoutReading()
    .withPreDetection(descreenStrength: 0.3, targetMaxSide: 1280)
    .withDetection(limitSideLen: 960, limitType: 'max')
    .withPostDetection(threshold: 0.3, boxThreshold: 0.6, unclipRatio: 1.6)
    .withPreRecognition(lineYThreshold: 0.5, wordXThreshold: 0.4)
    .withRecognition(batchSize: 8)
    .withPostRecognition(scoreThreshold: 0.5)
    .withOutput(OutputGranularity.lines)
    .run();
```

## Result Types

- `DetectTextResultStructured`: Returned by `.asLines()` and `.asWords()`. Inspect `res.results` (`List<TextResult>`):
  - `text` (`String`), `score` (`double`), `boxPoints` (`List<Point>`), `frame` (`Frame`)
  - `regionId` (`int?`) and `layoutType` (`LayoutType?`) when `withLayoutReading()` is enabled.
- `DetectTextResultSpatial`: Returned by `.asSpatial()`. Inspect `res.text` (`String`).

## License

Apache-2.0 OR MIT
