# RustO! Java SDK (`rusto-java`)

Pure Java / JVM bindings (via JNA) for **RustO!** — a high-performance, pure-Rust OCR engine powered by Indra OCR, PaddleOCR, and the pure-Rust MNN (`rusto-nn`) inference runtime.

> Looking for the Android AAR with `Context` asset resolution? See [`packages/android`](../android/README.md) (`rusto-android`).

## Features

- **Zero-Config Indra OCR**: `RustO.initialize()` automatically downloads and caches the default `indraocr` models on first run from Hugging Face Hub.
- **`AutoCloseable` Lifecycle**: Safe, deterministic native handle cleanup with Java `try-with-resources`.
- **Document Intelligence**: Enable UVDoc unwarping (`withAutoUnwarp()`), 4-way page orientation correction (`withOrientationDetection()`), and neural layout reading order (`withLayoutReading()`).
- **Fluent Builder API**: Chain `.asLines()`, `.asWords()`, `.asSpatial()`, `.asTsv()`, or `.withOutput(...)` into a single `.run()` terminal method returning `RustO.DetectTextResult`.
- **Flexible Inputs**: File paths, `byte[]` encoded images, or raw 24-bit RGB pixel buffers via `.rgb(...)`.

## Installation (JitPack)

### Gradle (Kotlin / Groovy DSL)

Add the JitPack repository and the `rusto-java` dependency:

```groovy
repositories {
    mavenCentral()
    maven { url 'https://jitpack.io' }
}

dependencies {
    implementation 'com.github.byrizki.rusto-rs:rusto-java:0.4.0'
}
```

### Maven

```xml
<repositories>
    <repository>
        <id>jitpack.io</id>
        <url>https://jitpack.io</url>
    </repository>
</repositories>

<dependencies>
    <dependency>
        <groupId>com.github.byrizki.rusto-rs</groupId>
        <artifactId>rusto-java</artifactId>
        <version>0.4.0</version>
    </dependency>
</dependencies>
```

> Ensure the native `librusto` library (`librusto.so`, `librusto.dylib`, or `rusto.dll`) is on `-Djna.library.path` or your system library path when running locally.

## Quick Start (Zero-Config)

```java
import com.byrizki.rusto.RustO;
import com.byrizki.rusto.TextResult;

public class Example {
    public static void main(String[] args) {
        try (RustO engine = RustO.initialize()) {
            // Structured line-level recognition with unwarping & layout reading order
            RustO.DetectTextResult res = engine.detectText("invoice.png")
                .withAutoUnwarp()
                .withLayoutReading()
                .asLines()
                .run();

            for (TextResult item : res.results()) {
                System.out.printf(
                    "[%.2f] (%s #%s) %s%n",
                    item.score(),
                    item.layoutType() != null ? item.layoutType() : "line",
                    item.regionId() != null ? item.regionId() : "-",
                    item.text()
                );
            }

            // 2D spatial layout reconstruction
            RustO.DetectTextResult spatial = engine.detectText("invoice.png")
                .asSpatial()
                .run();

            if (spatial.isSpatial()) {
                System.out.println(spatial.text());
            }
        }
    }
}
```

## Custom Initialization & Offline Models (.mnn / .rten)

Pass a JSON configuration string to `RustO.initialize(configJson)` to select another preset (`"ppocrv6-tiny"`, `"ppocrv5-mobile"`, etc.) or local `.mnn` / `.rten` files:

```java
String offlineConfig = """
    {
      "template": "indra",
      "enableReadingOrder": true,
      "detection": {
        "enabled": true,
        "modelPath": "./models/indra_det.mnn"
      },
      "recognition": {
        "enabled": true,
        "modelPath": "./models/indra_rec.mnn",
        "dictPath": "./models/indra_meta.json"
      },
      "unwarp": {
        "enabled": true,
        "modelPath": "./models/uvdoc.mnn"
      },
      "layout": {
        "enabled": true,
        "modelPath": "./models/layout_cdla.mnn"
      }
    }
    """;

try (RustO engine = RustO.initialize(offlineConfig)) {
    RustO.DetectTextResult words = engine.detectText("document.png")
        .asWords()
        .run();
}
```

## Encoded Bytes & Raw RGB Buffers

```java
import java.nio.file.Files;
import java.nio.file.Path;

try (RustO engine = RustO.initialize()) {
    // 1. Encoded PNG/JPEG bytes
    byte[] pngBytes = Files.readAllBytes(Path.of("receipt.png"));
    RustO.DetectTextResult fromBytes = engine.detectText(pngBytes)
        .asLines()
        .run();

    // 2. Raw interleaved 24-bit RGB buffer
    int width = 1280;
    int height = 720;
    long stride = width * 3L;
    byte[] rgbPixels = new byte[(int) (stride * height)];

    RustO.DetectTextResult fromRgb = engine.detectText()
        .rgb(rgbPixels, width, height, stride)
        .asLines()
        .run();
}
```

## Result Reference

### `RustO.DetectTextResult`

- `res.isStructured()` / `res.isSpatial()`: Indicates whether the result holds structured `TextResult` records or a 2D spatial string.
- `res.results()`: Returns `List<TextResult>` for `.asLines()` and `.asWords()`.
- `res.text()`: Returns `String` for `.asSpatial()`.

### `TextResult` Record

- `String text()`: Recognized UTF-8 text.
- `float score()`: Recognition confidence (`0.0f` – `1.0f`).
- `float[][] boxPoints()`: 4-corner polygon vertices `[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]`.
- `Frame frame()`: Axis-aligned bounding box (`width()`, `height()`, `top()`, `left()`).
- `Integer regionId()`: Detected layout region index when `withLayoutReading()` is enabled, or `null`.
- `String layoutType()`: Semantic layout category (`"text"`, `"title"`, `"table"`, `"header"`, `"footer"`, `"figure"`, etc.) or `null`.

## License

Apache-2.0 OR MIT
