---
trigger: model_decision
description: When need to do edit on library binding across platform and package distribution
---

# Multi-Platform Bindings & Distribution Rules

## 1. Zero-Mock Binding Verification
When developing or testing language bindings (Node.js, Python, Go, Java, Dart, Flutter, React Native, .NET):
- Never rely solely on unit tests with mock objects, dummy byte arrays, or empty inputs.
- Always provide automated end-to-end integration tests that feed canonical, real-world fixtures (real images, real model weights, real documents) through the binding and assert on extracted semantic content.

## 2. Pre-Release Package Integrity Gate
Before publishing or creating releases across package registries (NPM, NuGet, Maven Central, CocoaPods, PyPI, Pub.dev):
- Release and CI workflows must execute automated archive verification (e.g. `scripts/verify-packed-packages.py`).
- Validate mandatory manifest existence (`package.json`, `.nuspec`, `AndroidManifest.xml`, `METADATA`, `pubspec.yaml`, `go.mod`, `MANIFEST.MF`).
- Verify binary entrypoint availability and non-zero payload file sizes.
- Reject archives containing forbidden files (`.DS_Store`, `Thumbs.db`, `.git/`, `node_modules/`, `__pycache__/`, `.tmp`).

## 3. Cgo Dynamic Library Resolution
When authoring Go packages that use Cgo to link against local workspace shared libraries:
- Always supply runtime rpath flags in `#cgo LDFLAGS` alongside `-L` flags:
  ```go
  /*
  #cgo CFLAGS: -I${SRCDIR}
  #cgo LDFLAGS: -L${SRCDIR}/../../target/release -Wl,-rpath,${SRCDIR}/../../target/release -l<libname>
  */
  import "C"
  ```
- Downstream consumers and `go test` must run without requiring manual `LD_LIBRARY_PATH` configuration (Rule 5: Library-Side Encapsulation).

## 4. Flutter FFI Testing In FakeAsync
In Flutter widget and integration tests (`testWidgets`) that interface with native FFI or C ABI calls:
- Read local fixture files synchronously (`File(path).readAsBytesSync()`) rather than awaiting asynchronous microtasks (`await readAsBytes()`).
- This prevents microtask scheduling stalls within Flutter's `FakeAsync` zone during native inference execution.

## 5. Decoupled Model Distribution
- Version and release machine learning models independently from engine binaries (e.g. `models-rten-v*.*.*`).
- Package models natively per ecosystem:
  - iOS: Resource bundles (`.zip`) for CocoaPods / SPM.
  - Android: Asset-only `.aar` packages.
  - .NET: Content `.nupkg` packages with MSBuild targets.
  - NPM: JavaScript/TypeScript modules with bundled `.rten`/`.onnx` files.
  - Direct runtimes (Python, Go, Java, Dart): Support runtime filesystem paths and in-memory byte buffers.

## 6. Swift Package Manager & UniFFI Separation
In Swift Package Manager (`Package.swift`):
- Separate low-level C FFI headers and module maps into a dedicated target (e.g. `RustOFFI` with `publicHeadersPath: "."`).
- Have the higher-level Swift/Objective-C target depend on the FFI target (`dependencies: ["RustOFFI"]`).
- Avoid mixing raw C headers and Swift module files in the same target to prevent header search path ambiguity across CocoaPods and SPM.

## 7. Dart & Flutter Workspace Dependency Overrides
In multi-package Dart and Flutter repositories containing published packages and consumer example apps:
- Published packages (`packages/flutter/pubspec.yaml`) must depend on hosted packages (`rusto: ^x.y.z`), never local path dependencies.
- Local consumer example apps (`packages/flutter/example/pubspec.yaml`) that require active workspace development code must place local path references under `dependency_overrides:`, not `dependencies:`.
  ```yaml
  dependencies:
    flutter:
      sdk: flutter
    rusto_flutter:
      path: ../

  dependency_overrides:
    rusto:
      path: ../../dart
  ```
- This prevents `pub.dev` version solving conflicts while allowing local integration testing.

## 8. Multi-Registry Release Artifact Isolation
In CI/CD release and publish workflows interfacing with multiple package registries (PyPI, NPM, Pub.dev, NuGet, Maven):
- Never use broad generic wildcards (such as `rusto-*.tar.gz` or `*.*`) when downloading release artifacts into packaging directories.
- Always scope artifact downloads to strict version and extension globs (e.g., `-p "rusto-${VERSION}*.whl" -p "rusto-${VERSION}.tar.gz"`).
- Explicitly prune non-target or foreign platform archives prior to package integrity verification or upload steps (e.g. removing Android `.tar.gz` or `.aar` from Python distribution folders).

## 9. Public API Documentation Parity Across Bindings
Every exposed public class, struct, record, enum, property, and method across all platform bindings (Dart, Flutter, Node.js, Web, React, React Native, Java, Android Kotlin, iOS Swift, C# .NET, Go, Python) must maintain comprehensive documentation:
- Use native idiomatic doc comment styles:
  - TypeScript/JavaScript: JSDoc (`/** ... */`)
  - Kotlin: KDoc (`/** ... */`)
  - Java: Javadoc (`/** ... */`)
  - C# (.NET): XML doc comments (`/// <summary>`)
  - Swift: Swift Doc (`/// ...`)
  - Go: Go doc (`// Package ...`, `// Method ...`)
  - Python: Docstrings (`"""..."""`)
  - Rust / UniFFI: Doc comments (`/// ...`)
  - Dart: Dartdoc (`/// ...`)
- Document all method parameters, return types, units (e.g. pixels, normalized coordinates), and exception conditions.
- Generated binding files (e.g. UniFFI, cbindgen) must hide internal codecs and annotate public surface types to satisfy package quality gates (e.g. pub.dev 20%+ documentation threshold).

## 10. Minimal Public API Contract & Pipeline Co-Location
Across all platform bindings (Node.js, Python, Go, Java, Dart, Flutter, React Native, iOS, Android, .NET, Web, React):
- Expose only the essential 3-verb core lifecycle and detection contract:
  1. `initialize(config)`
  2. `detectText(source, options)` (polymorphic over path, bytes, or pixel buffers)
  3. `close()` / `dispose()`
- Do NOT expose standalone methods or secondary endpoints for sub-tasks (such as layout detection or document unwarping) if they can be co-located as pipeline options inside `detectText` (e.g. `options.layout = true`, `options.unwarp = true`).
- Maintain symmetrical lifecycle naming: ensure all bindings provide or alias `close()` alongside ecosystem conventions (`AutoCloseable` / `IDisposable`).

## 11. Synchronized Multi-Target UniFFI Regeneration
Whenever the UniFFI interface definition in Rust (`src/ffi_bindgen.rs`) changes:
- Immediately re-generate bindings across all supported languages before running verification:
  - Python: `cargo run --features uniffi-cli --bin uniffi-bindgen generate --language python`
  - Kotlin: `cargo run --features uniffi-cli --bin uniffi-bindgen generate --language kotlin`
  - Swift: `cargo run --features uniffi-cli --bin uniffi-bindgen generate --language swift`
  - Dart: `uniffi-bindgen-dart generate target/debug/librusto.so`
  - C# (.NET): `uniffi-bindgen-cs target/debug/librusto.so --library`
- Prune deleted symbols and obsolete types from all high-level language wrappers (`__init__.py`, `index.d.ts`, `RustODotnet.cs`, `RustO.kt`, etc.) to prevent runtime symbol missing errors (`AttributeError`, `StateError`).

## 12. Centralized Model Asset SSOT
- Models must have a single canonical source of truth: root `models/` directory managed via centralized download scripts (`scripts/download_models.sh`) or cloud releases (Hugging Face / GitHub Releases).
- Avoid per-package duplicated model distributions (such as `packages/models/`). When packaging schemes evolve, clean up untracked or gitignored legacy model directories to prevent repository disk bloat.


