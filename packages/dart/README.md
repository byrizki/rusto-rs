# rusto

[![pub.dev](https://img.shields.io/pub/v/rusto.svg?logo=dart&logoColor=white&color=0175C2)](https://pub.dev/packages/rusto)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pure Rust OCR library bindings for Dart and Flutter based on RapidOCR and PaddleOCR.

## Features

- High performance text detection and recognition powered by Rust and ONNX/RTEN
- Structured output with bounding box coordinates, confidence scores, and line frames
- Spatial layout generation preserving physical document geometry
- Seamless FFI integration with zero runtime overhead

## Getting Started

```dart
import 'package:rusto/rusto.dart';

final engine = RustO.initialize('{"detection":{"model_path":"det.rten"},"recognition":{"model_path":"rec.rten","dict_path":"dict.txt"}}');
final results = engine.detectTextFile('invoice.png');
for (final line in results) {
  print('${line.text} (score: ${line.score})');
}
engine.close();
```
