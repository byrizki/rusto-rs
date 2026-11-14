# Changelog

## 0.4.1

- Update RustO native library and bindings to 0.4.1.

## 0.4.0

### Breaking Changes

- Replaced v0.3 per-source detection helpers and legacy teardown methods with the unified fluent `DetectTextBuilder` API (`.asLines()`, `.asWords()`, `.asSpatial()`, `.withOutput(...)`) ending in a single `.run()` terminal method, plus `engine.close()` for deterministic native resource cleanup.
- Replaced flat v0.3 runtime option payloads with structured 6-phase `OcrRunOptions` (`preDetection`, `detection`, `postDetection`, `preRecognition`, `recognition`, `postRecognition`).

### Features

- Added **Indra OCR** (`indraocr`) as the zero-config default `.rten` pipeline with automatic model download and caching via `RustO.initialize()` and `RustO.downloadModel(...)`.
- Added `RustO.version()` and `RustO.isReady()` sourced directly from Rust core.
- Added neural layout analysis, transformer-based reading-order recovery (`withLayoutReading()`), and UVDoc document unwarping (`withAutoUnwarp()`).
- Enriched `TextResult` with `regionId` and `layoutType` semantic document metadata.

## 0.3.2

- Update RustO native library and bindings to 0.3.2.

## 0.3.1

- Initial release of RustO Dart FFI bindings.
- Support structured text detection and spatial text layout generation.
- Support dynamic library auto-discovery and explicit `RUSTO_LIB_PATH` resolution.
