# rusto

[![pub.dev](https://img.shields.io/pub/v/rusto.svg?logo=dart&logoColor=white&color=0175C2)](https://pub.dev/packages/rusto)
[![Documentation](https://img.shields.io/badge/docs-byrizki.com%2Frusto--rs-blue.svg)](https://byrizki.com/rusto-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pure Rust OCR library bindings for Dart and Flutter based on RapidOCR and PaddleOCR.

> 📖 **Full Documentation & Guides**: [byrizki.com/rusto-rs](https://byrizki.com/rusto-rs/)

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

## 📖 Documentation

For detailed guides, model configuration, and API reference:
- [Dart & Flutter Guide](https://byrizki.com/rusto-rs/how-to/use-flutter)
- [Full Documentation](https://byrizki.com/rusto-rs/)

