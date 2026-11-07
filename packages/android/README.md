# RustO! Android SDK (`rusto-android`)

Official Kotlin / Android AAR bindings for **RustO!** — a high-performance, pure-Rust OCR engine powered by Indra OCR, PaddleOCR, and pure-Rust MNN (`rusto-nn`) inference runtime.

## Features

- **Zero-Config Indra OCR**: `RustO.initialize(context)` automatically downloads and caches the default `indraocr` pipeline from Hugging Face Hub in `context.cacheDir/rusto/models` — or extracts bundled `det.mnn` (or `det.rten`) / `rec.mnn` (or `rec.rten`) / `dict.txt` from APK `assets/` when present.
- **Native Prebuilt JNI Slices**: Ships precompiled `librusto.so` binaries inside the AAR (`arm64-v8a`, `armeabi-v7a`, `x86_64`) for Android API 21+.
- **Document Intelligence**: UVDoc 3D unwarping (`withAutoUnwarp()`), 4-way page orientation correction (`withOrientationDetection()`), semantic layout analysis, and neural reading-order recovery (`withLayoutReading()`).
- **Fluent Builder API**: Chain `.asLines()`, `.asWords()`, `.asSpatial()`, `.asTsv()`, or `.withOutput(...)` into a single `.run()` call returning a sealed `DetectTextResult`.
- **Android-Native Inputs**: Accepts file paths, `file://` URIs, `content://` `Uri` streams via `ContentResolver`, encoded `ByteArray` / `Bitmap` payloads, and raw 24-bit RGB buffers via `.rgb(...)`.
- **`AutoCloseable` Lifecycle**: Idiomatic Kotlin `RustO.initialize(context).use { ocr -> ... }` or explicit `ocr.close()`.

## Installation (JitPack)

Add JitPack to `settings.gradle.kts` (or root `build.gradle`):

```kotlin
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
    }
}
```

Add the `rusto-android` dependency in your module's `build.gradle.kts`:

```kotlin
dependencies {
    implementation("com.github.byrizki.rusto-rs:rusto-android:v0.4.0")
}
```

If using zero-config on-demand model downloads, ensure your `AndroidManifest.xml` declares internet access:

```xml
<uses-permission android:name="android.permission.INTERNET" />
```

## Quick Start (Zero-Config)

Run initialization and OCR on `Dispatchers.IO` or `Dispatchers.Default`:

```kotlin
import android.content.Context
import com.byrizki.rusto.DetectTextResult
import com.byrizki.rusto.RustO
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

suspend fun scanInvoice(context: Context, imagePathOrContentUri: String) =
    withContext(Dispatchers.IO) {
        RustO.initialize(context).use { ocr ->
            // 1. Structured line-level results with unwarping & layout reading order
            val structured = ocr.detectText(imagePathOrContentUri)
                .withAutoUnwarp()
                .withLayoutReading()
                .asLines()
                .run()

            if (structured is DetectTextResult.Structured) {
                for (item in structured.items) {
                    println(
                        "[${"%.2f".format(item.score)}] " +
                            "(${item.layoutType ?: "line"} #${item.regionId ?: "-"}) ${item.text}"
                    )
                }
            }

            // 2. Reconstructed 2D spatial plain-text layout
            val spatial = ocr.detectText(imagePathOrContentUri)
                .asSpatial()
                .run()

            if (spatial is DetectTextResult.Spatial) {
                println(spatial.text)
            }
        }
    }
```

## Presets & Offline APK Asset Bundling

### Built-In Presets

```kotlin
import com.byrizki.rusto.InitializeConfig
import com.byrizki.rusto.RustO

// Default high-accuracy Indra OCR preset
val indraOcr = RustO.initialize(context, InitializeConfig.indra())

// Lightweight PP-OCRv6 Tiny preset (auto-downloaded if not cached)
val tinyOcr = RustO.initialize(
    context,
    InitializeConfig.forModel(model = "ppocrv6-tiny", lang = "latin")
)

// Explicit PP-OCRv6 preset with custom asset paths
val customPpv6 = RustO.initialize(
    context,
    InitializeConfig.ppv6(
        detModelPath = "models/ppocrv6_tiny_det.mnn",
        recModelPath = "models/ppocrv6_tiny_rec.mnn",
        dictPath = "models/ppocrv6_dict.txt",
    )
)
```

### Offline Bundled `.mnn` / `.rten` Models in `assets/`

1. Place `det.mnn` (or `det.rten`), `rec.mnn` (or `rec.rten`), and `dict.txt` (or `meta.json`) directly in `app/src/main/assets/` for automatic offline discovery via `RustO.initialize(context)`, **or** place named models in `app/src/main/assets/models/` and pass their relative asset paths in `InitializeConfig` (`DetectionConfig`, `RecognitionConfig`, `UnwarpConfig`, `LayoutConfig`).
2. `RustO.initialize(...)` transparently extracts bundled APK assets to `context.cacheDir` on first use.

You can also pre-download models ahead of time:

```kotlin
val downloadedPaths = RustO.downloadModel(
    model = "indraocr",
    lang = "latin",
    cacheDir = context.cacheDir.resolve("rusto/models").absolutePath,
)
```

## `Bitmap`, `Uri`, & Raw RGB8 Input

### Android `Bitmap` via Encoded Bytes or Raw RGB8 Pixels

```kotlin
import android.graphics.Bitmap
import com.byrizki.rusto.DetectTextResult
import com.byrizki.rusto.RustO
import java.io.ByteArrayOutputStream

// Option A: Encoded PNG/JPEG ByteArray from Bitmap
fun recognizeBitmapEncoded(ocr: RustO, bitmap: Bitmap): DetectTextResult {
    val stream = ByteArrayOutputStream()
    bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
    return ocr.detectText(stream.toByteArray())
        .asLines()
        .run()
}

// Option B: Zero-compression interleaved RGB8 pixels from ARGB_8888 Bitmap
fun recognizeBitmapRgb(ocr: RustO, bitmap: Bitmap): DetectTextResult {
    val argb = if (bitmap.config == Bitmap.Config.ARGB_8888) {
        bitmap
    } else {
        bitmap.copy(Bitmap.Config.ARGB_8888, false)
    }
    val width = argb.width
    val height = argb.height
    val intPixels = IntArray(width * height)
    argb.getPixels(intPixels, 0, width, 0, 0, width, height)

    val rgbBytes = ByteArray(width * height * 3)
    var out = 0
    for (px in intPixels) {
        rgbBytes[out++] = ((px shr 16) and 0xFF).toByte() // R
        rgbBytes[out++] = ((px shr 8) and 0xFF).toByte()  // G
        rgbBytes[out++] = (px and 0xFF).toByte()          // B
    }

    return ocr.detectText()
        .rgb(rgbBytes, width, height, stride = width.toLong() * 3L)
        .asLines()
        .run()
}
```

### Android `content://` `Uri` (Photo Picker / CameraX)

```kotlin
val res = ocr.detectText()
    .image(photoPickerUri) // android.net.Uri or String
    .asWords()
    .run()
```

## 6-Phase Pipeline Tuning

`DetectTextBuilder` supports Kotlin trailing-lambda configuration blocks for all 6 pipeline phases:

```kotlin
val result = ocr.detectText(path)
    .withOrientationDetection()
    .withAutoUnwarp()
    .withLayoutReading()
    .withPreDetection { copy(descreenStrength = 0.3f, targetMaxSide = 1280u) }
    .withDetection { copy(limitSideLen = 960, limitType = "max") }
    .withPostDetection { copy(threshold = 0.3f, boxThreshold = 0.6f, unclipRatio = 1.6f) }
    .withPreRecognition { copy(lineYThreshold = 0.5f, wordXThreshold = 0.4f) }
    .withRecognition { copy(batchSize = 8) }
    .withPostRecognition { copy(scoreThreshold = 0.5f) }
    .withOutput(com.byrizki.rusto.OutputGranularity.LINES)
    .run()
```

## Result Reference

`DetectTextResult` is a Kotlin `sealed interface`:

- `DetectTextResult.Structured(val items: List<TextResult>)`:
  - `text: String`
  - `score: Float` (`0.0f` – `1.0f`)
  - `boxPoints: List<Point2D>` (`x`, `y`)
  - `frame: Frame` (`width`, `height`, `top`, `left`)
  - `regionId: Int?` (layout region index when `withLayoutReading()` is enabled)
  - `layoutType: String?` (`"text"`, `"title"`, `"table"`, `"header"`, `"footer"`, `"figure"`, `"equation"`, etc.)
- `DetectTextResult.Spatial(val text: String)`: Formatted 2D plain-text document layout.

## License

Apache-2.0 OR MIT

