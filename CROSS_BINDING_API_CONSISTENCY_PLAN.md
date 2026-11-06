# Cross-Binding API Consistency Plan

## Status

Implemented in working tree. Breaking major-version API. No public legacy aliases or fallback methods.

## Canonical Names

| Binding | Engine creation | OCR | Lifecycle | Readiness & Version | Pre-download |
| --- | --- | --- | --- | --- | --- |
| Rust | `RustO::initialize` | `detect_text` | `Drop` | `version()`, `is_ready()` | `download_model_pipeline` |
| C FFI | `rocr_initialize` | `rocr_detect_text_*` | `rocr_free` | `rocr_version()`, `rocr_is_ready()` | `rocr_download_model` |
| Node.js | `RustO.initialize` | `detectText` | `close()` | `version()`, `isReady()` | `RustO.downloadModel` |
| Web | `initialize` / `WebOcrEngine` | `detectText` | `close()` | `version()`, `isReady()` | `downloadModel` |
| Python | `RustO.initialize` | `detect_text` | `close()` | `version()`, `is_ready()` | `download_model` |
| Go | `rusto.Initialize` | `DetectText` | `Close()` | `Version()`, `IsReady()` | `DownloadModel` |
| .NET | `RustO.Initialize` | `DetectText` | `Dispose()` | `Version()`, `IsReady()` | `RustO.DownloadModel` |
| Java | `RustO.initialize` | `detectText` | `close()` | `version()`, `isReady()` | `RustO.downloadModel` |
| Android | `RustO.initialize` | `detectText` | `close()` | `version()`, `isReady()` | `RustO.downloadModel` |
| iOS | `RustO.initialize` | `detectText` | `deinit` / `close()` | `version()`, `isReady()` | `RustO.downloadModel` |
| Dart / Flutter | `RustO.initialize` | `detectText` | `close()` | `version()`, `isReady()` | `RustO.downloadModel` |
| React Native | `initialize` | `detectText` | `close()` | `version()`, `isReady()` | module asset download |

Casing follows host language. Operation semantics stay same.

## Shared Contract

### Engine Lifecycle, Versioning & Readiness

Every binding provides canonical readiness and version inquiries sourced directly from Rust:
- `version()` / `rocr_version()`: Returns the cargo package version string (`0.4.0`).
- `isReady()` / `rocr_is_ready()`: Returns `true` if the engine handle/pipeline is initialized and ready for inference, `false` otherwise.
- `close()` / `rocr_free()` / `Dispose()`: Closes the engine and cleans up worker/model resources. `terminate()` is strictly removed.
- `downloadModel(model?, lang?, cacheDir?)`: Unified model pre-downloading method across all platforms. `downloadAllModels` is removed.

### Encapsulation & Purity (TypeScript Packages)

- `@rustors/web` exports only the high-level OCR API (`initialize`, `detectText`, `detectTextRgb`, `close`, `isReady`, `version`, `downloadModel`, `WebOcrEngine`, `DetectTextBuilder`) and public type definitions.
- All internal validator utilities (`normalizeOptions`, `requireOnlyKeys`, `isObject`, etc.) and geometry flow math (`sortCandidatesReadingOrder`, `formatSpatialText`, `groupCandidatesIntoLines`, `calculateFrameFromQuad`) are **entirely removed** from the TypeScript codebase.
- All layout reading order, spatial 2D text reconstruction, line clustering, and word tokenization are executed directly in WebAssembly using `rusto-core` as the single source of truth.


### Initialization

`InitializeConfig` contains static model/session configuration only. It is passed once to `initialize` / `Initialize`. Image preprocessing is exclusively per-request under `OcrRunOptions` / `DetectTextOptions`, so every call can hot swap it without recreating model sessions.

### Input

Native bindings use `ImageSource`:

```text
URI/path | image bytes
```

React Native keeps its bridge-safe union:

```ts
type ImageSource =
  | { uri: string }
  | { base64: string }
  | { bytes: Uint8Array | ArrayBuffer };
```

### Request options

```text
OcrRunOptions / DetectTextOptions
├── output: lines | words | spatial
├── lineYThreshold?: finite number >= 0
├── wordXThreshold?: finite number >= 0
├── textScore?: finite number in [0, 1]
├── classification?: boolean
├── orientation?: boolean
├── minHeight?, maxSideLen?, minSideLen?, widthHeightRatio?: request-local resize controls
├── detection?: limitSideLen, limitType, mean, std
└── postprocess?: threshold, boxThreshold, maxCandidates, unclipRatio, useDilation
```

Wire JSON uses camelCase. Defaults: `lines`, `0.5`, `0.4`, engine score, `false`, `false`.

### Result

Native bindings return `DetectTextResult`:

```text
Structured(List<TextResult>)  // lines or words
Spatial(String)               // spatial
```

React Native keeps overloads:

```ts
detectText(source, { output: 'spatial' }): Promise<string>
detectText(source, options?): Promise<TextResult[]>
```

`TextResult` always contains text, score, TL/TR/BR/BL quad, and axis-aligned frame.

## Output Semantics

- `lines`: candidates group by vertical-center distance `medianHeight × lineYThreshold`.
- `words`: same row logic; recognized whitespace is hard boundary; X threshold merges only fragments without whitespace boundary.
- `spatial`: same Y grouping; X threshold controls spacing; rows join with `\n`.
- Per-request options never mutate initialized engine configuration.

## C FFI

Only canonical OCR calls remain:

```c
ROCRHandle *rocr_initialize(const char *initialize_config_json);
int rocr_detect_text_file(ROCRHandle *, const char *path, const char *options_json,
                          CTextResult **results, size_t *count);
int rocr_detect_text_data(ROCRHandle *, const uint8_t *data, size_t length,
                          const char *options_json, CTextResult **results, size_t *count);
char *rocr_detect_text_file_spatial(ROCRHandle *, const char *path,
                                    const char *options_json);
char *rocr_detect_text_data_spatial(ROCRHandle *, const uint8_t *data, size_t length,
                                    const char *options_json);
```

Rules:

- structured calls reject `output: "spatial"`;
- spatial calls require `output: "spatial"`;
- free structured results with `rocr_free_results`;
- free spatial strings with `rocr_free_string`;
- free engine with `rocr_free`.

## Binding Shapes

### Rust

```rust
let mut ocr = RustO::initialize(config)?;
let result = ocr.detect_text(&ImageSource::Path("invoice.jpg".into()), &options)?;
```

### Android

```kotlin
val ocr = RustO.initialize(context, config)
val result = ocr.detectText(ImageSource.Uri("/data/invoice.jpg"), options)
```

### iOS

```swift
let ocr = try RustO.initialize(config: config)
let result = try ocr.detectText(.uri("invoice.jpg"), options: options)
```

### .NET

```csharp
using var ocr = RustO.Initialize(config);
var result = ocr.DetectText(new UriImageSource("invoice.jpg"), options);
```

### React Native

```ts
await initialize(config);
const result = await detectText({ uri: imageUri }, { output: 'words' });
```

## Removed Public API

Remove source-specific OCR and formatter methods:

```text
new / create constructors for engine lifecycle
run / recognize / Recognize* / recognizeFile*
raw, CSV, positioned-text, and dedicated spatial methods
raw JSON Android request overloads
old C rocr_new, rocr_ocr_*, output-handle APIs
terminate() (replaced universally with close() / Dispose())
downloadAllModels() (replaced with targeted downloadModel(model?, lang?, cacheDir?))
Geometry & Layout flow helpers from TypeScript public index (sortCandidatesReadingOrder, formatSpatialText, etc.)
Internal validator functions from TypeScript public index (normalizeOptions, requireOnlyKeys, etc.)
```

No deprecated wrappers. No aliases. No fallback behavior.

## Validation

Completed:

```bash
cargo test --features ffi --lib --tests
dotnet build packages/dotnet/RustODotnet.csproj --no-restore -v:minimal
```

Both pass. Android/iOS/RN native compile needs consumer Gradle/Xcode and React Native dependencies unavailable locally.

`cargo fmt --check` still fails on pre-existing formatting drift in `src/orient.rs` and `src/postprocess.rs`; this change does not format unrelated files.
