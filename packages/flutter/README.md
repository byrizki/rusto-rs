# rusto_flutter

[![pub.dev](https://img.shields.io/pub/v/rusto_flutter.svg?logo=flutter&logoColor=white&color=02569B)](https://pub.dev/packages/rusto_flutter)
[![Documentation](https://img.shields.io/badge/docs-byrizki.com%2Frusto--rs-blue.svg)](https://byrizki.com/rusto-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Flutter plugin for RustO, a high-performance pure Rust OCR engine based on RapidOCR and PaddleOCR.

> 📖 **Full Documentation & Guides**: [byrizki.com/rusto-rs](https://byrizki.com/rusto-rs/)

## Features

- High-performance text detection and recognition on mobile and desktop
- Pre-compiled native libraries for Android (x86_64, arm64-v8a, armeabi-v7a)
- Zero-overhead C FFI bridge directly into Rust OCR core
- Built-in spatial text layout generation preserving physical document formatting

## Usage

```dart
import 'package:rusto_flutter/rusto_flutter.dart';

// Use RustO directly via FFI
final engine = RustO.initialize('{"detection":{"model_path":"det.rten"},"recognition":{"model_path":"rec.rten","dict_path":"dict.txt"}}');
final results = engine.detectTextData(imageBytes);
for (final line in results) {
  print('${line.text} (score: ${line.score})');
}
engine.dispose();
```

## 📖 Documentation

For detailed guides, model setup, and API reference:
- [Flutter Guide](https://byrizki.com/rusto-rs/how-to/use-flutter)
- [Full Documentation](https://byrizki.com/rusto-rs/)

