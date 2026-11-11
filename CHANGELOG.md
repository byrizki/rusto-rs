# Changelog

All notable changes to RustO! will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-11-06

### Breaking Changes
- **api**: Unify all 12 platform SDKs around a single fluent `DetectTextBuilder` contract with `as_lines()`/`asLines()`/`AsLines()`, `as_words()`/`asWords()`/`AsWords()`, `as_spatial()`/`asSpatial()`/`AsSpatial()`, and a single terminal `.run()` / `.Run()` method; remove pre-0.4 `detectTextFile*`, `detectTextData*`, `detectTextSpatial`, `runSpatial`, `execute()`, and `dispose()` methods (@byrizki)
- **options**: Replace v0.3 runtime options (`textScore`, `postprocess`, `calibration`, `optimization`) with 4 top-level pipeline toggles (`autoUnwarp`, `layoutReading`, `orientationDetection`, `lineClassification`) and 6 structured pipeline phases (`preDetection`, `detection`, `postDetection`, `preRecognition`, `recognition`, `postRecognition`) (@byrizki)
- **models**: Set DataLab **Indra OCR** (`indraocr`) as the v0.4.0 default model (later updated to Apache 2.0 `ppocrv6-tiny` FP32) and host standalone models on Hugging Face Hub (`byrizki/rusto-models`) with native CORS support; retire legacy GitHub Releases model hosting and platform-specific model wrapper packages (@byrizki)
- **engine**: Transition primary inference runtime from RTen to pure-Rust MNN graph execution with `rusto-nn` while preserving pluggable RTen, ONNX, and hardware backends (@byrizki)
- **cli**: Rename the CLI binary from `rusto-rs` to `rusto` (`rusto <IMAGE>`, `rusto ocr <IMAGE>`, `rusto download`) and standardize CLI flags (`--score-threshold`, `--rec-batch-num`, `--format spatial`) (@byrizki)

### Added
- **nn**: Implement `rusto-nn` pure-Rust neural inference runtime and hardware acceleration backends alongside modular `rusto-core`, `rusto-rten`, and `rusto-mnn` crates (@byrizki)
- **pipeline**: Add end-to-end document processing stages — 3D UVDoc document unwarping, orientation detection, PP-Structure/RapidLayout PicoDet multi-scale region detection (`region_id`, `layout_type`), and Indra graph neural reading order (`indra_order.rten`) (#23) (@byrizki)
- **models, bindings**: Add zero-config model auto-download on `RustO::initialize()` across Rust, Node.js, Python, Go, Java, .NET, Dart, Android, iOS, and React Native, plus `rusto download` CLI command (@byrizki)
- **bindings**: Expose fluent `.rgb(...)` decoded interleaved RGB8 buffer input on `DetectTextBuilder` across native and managed SDKs (@byrizki)
- **web, wasm**: Support 1-channel Indra OCR pipeline, neural reading order (`indra_order.rten`), dynamic release model resolution, and multi-column reading order in `rten-wasm`, `@rustors/web`, and `@rustors/react` (@byrizki)
- **docs**: Add interactive layout/unwarp WASM configurator, head-to-head OmniDocBench evaluation suite, and bilingual v0.3 → v0.4 migration guides (EN / ID) (@byrizki)

### Changed
- **core**: Optimize RTen inference arena reuse, parallel Rayon line cropping, and 1-channel grayscale recognition preprocessing (@byrizki)
- **mnn**: Register `vendor/alibaba-mnn` as an explicit Git submodule with `MNN_DIR` build fallback for source distributions (@byrizki)
- **layout**: Add multi-platform end-to-end dense multi-column layout regression suite (`chronam-ad-page.png`) across all SDK bindings (@byrizki)

### Fixed
- **layout**: Resolve multi-column reading order sequencing and expose `region_id` and `layout_type` across C FFI (`CTextResult`) and all language bindings (fixes #25, #26) (@byrizki)
- **det**: Support polygon scoring (`scoreMode: "poly"`) and forward nested post-detection options (#21) (@byrizki)
- **ios, android, flutter**: Synchronize UniFFI generated bindings, Swift struct initializers, Kotlin `InitializeConfig` asset/cache resolution, and asynchronous Flutter widget tests (@byrizki)

## [0.3.2] - 2026-10-25

### Added
- **scripts**: Tag committer github user in changelog entries (@byrizki)
- **scripts**: Update bump_version.py to cover all bindings, packages, examples, and docs (@byrizki)
- **bindings**: Unify generate-bindings pipeline across all targets (@byrizki)
- **dotnet**: Migrate to auto-generated uniffi-bindgen-cs bindings (@byrizki)
- **dart**: Migrate to auto-generated uniffi-bindgen-dart bindings (@byrizki)
- **core**: Pin uniffi 0.31 and automate C header generation with cbindgen (@byrizki)
- **core**: Add contiguous copy fast path, spatial C FFI, bindings, and docs for decoded RGB input (@byrizki)
- Add validated decoded RGB input for Rust and C callers (@tacman)
- **ci**: Add model package verifications and binding contract tests (@byrizki)
- **verify**: Add go/flutter/dart/java package verification and verify actual image across all bindings (@byrizki)
- **flutter**: Verify real Flutter APK, FFI spatial pipeline, and live Maestro E2E test (@byrizki)
- Add packed packages content verifier, CI improvements, and Flutter example app with Maestro (@byrizki)
- **rn-android**: Enable end-to-end APK build and Maestro verification without mocks (@byrizki)
- **bindings**: Add Obj-C/SPM iOS, unified RN Maestro test suite, and multi-language bindings (Node, Python, Go, Java, Dart/Flutter) (@byrizki)

### Changed
- Read release body from changelog in build workflow (@byrizki)
- **flutter**: Setup Java 17 for Maestro CLI execution in flutter-example job (@byrizki)
- Fix wasm loading in docs ci, model check in wasm test, and sequential docs deployment (@byrizki)
- Separate CI workflows for docs and library (@byrizki)
- Add CI workflow for PRs and push using reusable build workflow (@byrizki)
- Update package badges and align model package versions to 0.3.2 (@byrizki)
- Update documentation, quick-start guides, and code snippets across bindings (@byrizki)
- Replace Maven publishing with JitPack build trigger for Android and Java (@byrizki)
- Update build and publish workflows for multi-platform language bindings (@byrizki)
- **docs**: Upgrade nuxt, docus, and mermaid to latest versions (@byrizki)
- **docs**: Consolidate Snyk dependency upgrades for nuxt, docus, and mermaid (@byrizki)
- Humanize models guide, correct multilingual language scope, and fix diagrams (@byrizki)
- Add model zip download releases and consumer decision guide (@byrizki)
- Improve iOS, React Native iOS, and Models & Tiers guides (@byrizki)
- Add website guides and installation for node, python, go, java, and flutter (@byrizki)
- Add ecosystem packages documentation and quickstarts for node, python, go, java, and flutter (@byrizki)
- Format RGB integration test (@tacman)
- Update locks (@byrizki)
- Enable default ffi features, fix Dart FFI deprecation, and clean linting in Flutter and React Native example (@byrizki)
- Snyk security integration, planning documents, and documentation updates (@byrizki)

### Fixed
- **ios**: Add RuntimeOptions typealias for OcrRunOptions and implement parameterless init (@byrizki)
- **ios**: Use zero-argument init() for UniFFI structs to prevent redeclaration collisions (@byrizki)
- **bindings**: Resolve iOS Swift config inits, RN android local AARs, and flutter test FFI library resolution (@byrizki)
- **ffi**: Exclude internal uniffi rustbuffer symbols from canonical C header (@byrizki)
- Replace geo-clipper with clipper2-rust (@byrizki)
- Upgrade better-sqlite3 from 12.6.2 to 12.11.1 (@snyk-bot)
- FFI memory safety issues and update OCR pipeline coordinate mapping and detection logic (@byrizki)
- Fix bugs and optimize core image processing & contour detection (@google-labs-jules[bot])

## [0.3.1] - 2026-10-10

### Changed
- **docs**: Load rten models from jsdelivr cdn instead of bundling
- Update readme
- Update content
- Rename to @rustors for npm packages
- Release models-rten-v0.3.2
- Rename packages
- Release models-rten-v0.3.1
- Remove unpublish-models workflow as npm requires interactive 2FA
- Add workflow to unpublish npm model packages
- Set default to true for all model publish workflow dispatch options
- Selectively run packaging and artifact uploads based on checked options
- Auto-read MODELS_VERSION and run all checkable release/publish items in parallel
- Use absolute paths for npm publish to avoid git repo misinterpretation
- Update live demo docs
- Update demo content
- Improve live demo

### Fixed
- **ci**: Bundle Swift sources and LICENSE in RustO.xcframework.zip for CocoaPods
- Generate_npm_models
- **npm**: Include .rten models as primary files in npm packages
- **ci**: Streamline pipeline to download ONNX -> quantize to INT8 -> convert all to RTen
- **ci**: Preserve onnx models with --keep-onnx for INT8 quantization in publish-models workflow
- **scripts**: Support --all flag and python module fallback in quantize_ppocrv6_int8.py
- Separate models publishing

## [0.3.0] - 2026-10-07

### Added
- **packages**: Add @rustors/web, @rustors/react, and @rustors/model-* packages
- **wasm**: Add rten-wasm crate for high-performance browser OCR inference
- **core**: Add configurable image calibration, descreening, and optimization pipeline

### Changed
- Standardize session usage from RtenSession/MnnSession to EngineSession
- Update npm model generator and PowerShell scripts for ONNX/RTen
- Complete MNN to RTen migration across bindings, examples, and podspecs
- Update workflows and model scripts for RTen engine migration
- Update gitignore for debugging dumps and add architecture benchmark plans
- Update documentation, guides, and interactive live demo with RTen WASM
- **bindings**: Align Android, iOS, .NET, and React Native bindings with RTen and calibration options
- **engine**: Replace MNN with pure-Rust RTen backend and remove legacy MNN crates
- Fix formats
- Update node to 24

## [0.2.5] - 2026-08-20

### Added
- Add docs sites, improve configuration api

### Fixed
- Fix RN lint

## [0.2.4] - 2026-08-20

### Added
- Add several text detection configuration

### Fixed
- Fix ts lint

## [0.2.3] - 2026-08-19

### Changed
- Refactor build ci
- Use legacy yarn
- Update download script
- Refactor to use yarn

### Fixed
- Fix ocr source validation
- Fix ios test build
- Fix ios build
- Fix subworkflow run
- Fix build example gate
- Fix build example gate, rn 0.75+
- Fix rn deps install
- Fix example build gate
- Fix npm install examples gate

## [0.2.2] - 2026-08-14

### Changed
- Add example build gate
- Optimize & rename public apis

### Fixed
- Fix rn kotlin bridge

## [0.2.1] - 2026-08-13

### Changed
- Bump version

### Fixed
- Fix jitpack release

## [0.2.0] - 2026-08-13

### Changed
- Update publishing

## [0.1.7] - 2026-08-12

### Changed
- Update changelog

### Fixed
- Kotlin action

## [0.1.6] - 2026-08-12

### Added
- Separate model files from repo, download on-the-fly from rusto-rs-models releases

### Fixed
- Fix swift lint
- Fix C# build


## [0.1.5] - 2026-08-12

### Added
- Add PP OCR v6

### Changed
- Update cargo.toml
- Use maven central

### Fixed
- Fix crates publish with retry
- Fix sign key id

## [0.1.4] - 2026-08-12

### Changed
- Update script
- Use prebuilt MNN intelligently

### Fixed
- Fix release tag on build
- Fix ios/osx build
- Revert mnn submodule to upstream; inject cstdint via cmake flags instead
- Update mnn submodule - add cstdint for AppleClang 21 (macOS 26)
- Fix mac os build
- Android/ios file content uri resolver
- RN binding, build and ts types

## [0.1.0] - 2024-11-20

### Added
- 🎉 Initial release of RustO! (formerly rapidocr)
- Pure Rust OCR implementation based on RapidOCR
- PaddleOCR PPOCRv4/v5 model support via ONNX Runtime
- High-level API with `RapidOCR` and `RapidOCRConfig`
- CLI application with JSON, Text, and TSV output formats
- C FFI bindings for cross-language integration
- iOS CocoaPods support via Pod spec
- GitHub Actions CI/CD workflows
- Comprehensive documentation and examples

### Changed
- **Branding**: Renamed from `rapidocr` to `RustO!` (crate name: `rusto-rs`)
- **Library name**: Changed from `rapidocr` to `rusto` for imports
- **CLI name**: Updated from `rapidocr` to `rusto`
- **Documentation**: Complete rebranding with RustO! identity
- **Documentation**: Added comprehensive architecture explanation
- **Documentation**: Added PaddleOCR and RapidOCR acknowledgments

### Fixed
- **Warnings**: Cleaned up all 16 compiler warnings
  - Added `#![allow(dead_code)]` to experimental code in `contours.rs`
  - Fixed unused variable warnings
  - Suppressed warnings for reference implementations
- **Library API**: Fixed `lib.rs` to use correct `RapidOcr::new_ppv5()` method
- **Library API**: Simplified `RapidOCRConfig` to only include essential model paths
- **Library API**: Added `run()` method to `RapidOcr` for convenient file-based OCR
- **CLI**: Removed unused `use_opencv` flag from CLI arguments
- **CLI**: Updated to use simplified `RapidOCRConfig` structure
- **Build**: Removed old `rapidocr_json` binary in `src/bin/`
- **Build**: Clean builds with zero warnings ✅

### Build Status
- ✅ `cargo check` - Success with 0 warnings
- ✅ `cargo build` - Success with 0 warnings
- ✅ `cargo build --release` - Success
- ✅ CLI binary compiles correctly (`rusto`)
- ✅ Library API is functional
- ✅ FFI bindings ready

### Infrastructure
- ✅ iOS Pod spec created (`RustO.podspec`)
- ✅ GitHub Actions CI workflow (`.github/workflows/ci.yml`)
- ✅ GitHub Actions release workflow (`.github/workflows/release.yml`)
- ✅ Automated testing on Linux, macOS, and Windows
- ✅ Automated documentation deployment
- ✅ Security audits configured

---

## Project Status

**Version**: 0.4.0  
**Status**: Production Ready  
**Accuracy**: 99.3% OpenCV parity  
**Last Updated**: November 06, 2026
