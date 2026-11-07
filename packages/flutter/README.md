# rusto_flutter

Official Flutter FFI plugin for **RustO!** — a high-performance, pure-Rust OCR engine powered by Indra OCR, PaddleOCR, and pure-Rust MNN (`rusto-nn`) inference runtime.

`package:rusto_flutter/rusto_flutter.dart` bundles native library packaging for Flutter targets and directly re-exports the `RustO` API from `package:rusto/rusto.dart`.

## Features

- **Zero-Config or Offline Models**: Auto-download the default **Indra OCR** pipeline from Hugging Face Hub into your app cache directory, or extract bundled `.mnn` / `.rten` assets from Flutter `rootBundle` for 100% offline mobile execution.
- **Fluent Builder API**: Chain `.asLines()`, `.asWords()`, `.asSpatial()`, `.asTsv()`, or `.withOutput(...)` into a single `.run()` call on `RustO`.
- **Document Intelligence**: UVDoc document unwarping (`withAutoUnwarp()`), 4-way orientation detection (`withOrientationDetection()`), and neural layout reading-order recovery (`withLayoutReading()`).
- **UI-Smooth Isolate Execution**: Run synchronous FFI inference inside `Isolate.run(...)` to keep Flutter animations at 60/120 FPS.

## Installation

Add `rusto_flutter` to your Flutter app's `pubspec.yaml`:

```yaml
dependencies:
  rusto_flutter: ^0.4.0
```

## Quick Start (Zero-Config with `Isolate.run`)

On mobile platforms, pass your writable cache directory (`getTemporaryDirectory()`) so `RustO.initialize(...)` can cache downloaded models from Hugging Face Hub, and run inference inside `Isolate.run` to avoid blocking the UI thread:

```dart
import 'dart:convert';
import 'dart:isolate';
import 'package:path_provider/path_provider.dart';
import 'package:rusto_flutter/rusto_flutter.dart';

Future<List<TextResult>> recognizeInvoice(String imagePath) async {
  final tempDir = await getTemporaryDirectory();
  final cacheDir = '${tempDir.path}/rusto_models';

  return Isolate.run(() {
    final engine = RustO.initialize(jsonEncode({'cacheDir': cacheDir}));
    try {
      final res = engine
          .detectText(imagePath)
          .withAutoUnwarp()
          .withLayoutReading()
          .asLines()
          .run();

      return res is DetectTextResultStructured ? res.results : const <TextResult>[];
    } finally {
      engine.close();
    }
  });
}
```

> On desktop targets (macOS, Linux, Windows), `RustO.initialize()` with no arguments automatically resolves the default user cache directory.

## Offline Bundled `.rten` Assets

To ship a completely offline Flutter app, bundle `.rten` models in `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/models/det.rten
    - assets/models/rec.rten
    - assets/models/dict.txt
```

Extract the assets to your app documents/cache directory once and initialize `RustO`:

```dart
import 'dart:convert';
import 'dart:io';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:rusto_flutter/rusto_flutter.dart';

Future<String> extractAsset(String assetKey, String fileName) async {
  final dir = await getApplicationSupportDirectory();
  final file = File('${dir.path}/$fileName');
  if (!file.existsSync() || file.lengthSync() == 0) {
    final bytes = await rootBundle.load(assetKey);
    await file.writeAsBytes(bytes.buffer.asUint8List(), flush: true);
  }
  return file.path;
}

Future<void> runOfflineOcr(String imagePath) async {
  final detPath = await extractAsset('assets/models/det.rten', 'det.rten');
  final recPath = await extractAsset('assets/models/rec.rten', 'rec.rten');
  final dictPath = await extractAsset('assets/models/dict.txt', 'dict.txt');

  final configJson = jsonEncode({
    'template': 'ppv6', // or 'indra'
    'detection': {'modelPath': detPath, 'enabled': true},
    'recognition': {'modelPath': recPath, 'dictPath': dictPath, 'enabled': true},
  });

  final engine = RustO.initialize(configJson);
  try {
    // 1. Structured line results
    final structured = engine
        .detectTextBuilder()
        .image(imagePath)
        .withPostRecognition(scoreThreshold: 0.3)
        .asLines()
        .run();

    if (structured is DetectTextResultStructured) {
      for (final line in structured.results) {
        print('[${line.score.toStringAsFixed(2)}] ${line.text}');
      }
    }

    // 2. 2D spatial text reconstruction
    final spatial = engine
        .detectTextBuilder()
        .image(imagePath)
        .asSpatial()
        .run();

    if (spatial is DetectTextResultSpatial) {
      print(spatial.text);
    }
  } finally {
    engine.close();
  }
}
```

## Raw RGB Camera / Image Buffers

Feed decoded 24-bit RGB frames directly via `detectTextBuilder().rgb(...)`:

```dart
final res = engine
    .detectTextBuilder()
    .rgb(rgbBytes, width, height, width * 3)
    .asWords()
    .run();
```

## Result Types

- `DetectTextResultStructured`: Access `res.results` (`List<TextResult>`), where each item provides `text`, `score`, `boxPoints`, `frame`, `regionId`, and `layoutType`.
- `DetectTextResultSpatial`: Access `res.text` (`String`) containing the reconstructed 2D text layout.

See [`example/lib/main.dart`](example/lib/main.dart) for a complete Flutter verification app.

## License

Apache-2.0 OR MIT
