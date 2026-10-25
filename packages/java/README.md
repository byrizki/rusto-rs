# rusto-java

[![JitPack](https://jitpack.io/v/byrizki/rusto-rs.svg)](https://jitpack.io/#byrizki/rusto-rs)
[![Documentation](https://img.shields.io/badge/docs-byrizki.com%2Frusto--rs-blue.svg)](https://byrizki.com/rusto-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pure-Rust Optical Character Recognition (OCR) engine for JVM environments (Java, Kotlin, Scala) via JNA.

> 📖 **Full Documentation & Guides**: [byrizki.com/rusto-rs](https://byrizki.com/rusto-rs/)

## Features

- **Zero JNI Boilerplate**: Clean Java Native Access (JNA) bindings directly invoking RustO C ABI.
- **AutoCloseable**: Implements `AutoCloseable` for clean `try-with-resources` memory reclamation.
- **Rich Document Models**: Immutable records/classes for `TextResult`, `Frame`, and bounding boxes.
- **Spatial Formatting**: Extract structured bounding boxes or reconstructed 2D visual layouts.

## Installation

### Gradle (Kotlin DSL)

```kotlin
repositories {
    mavenCentral()
    maven { url = uri("https://jitpack.io") }
}

dependencies {
    implementation("com.github.byrizki.rusto-rs:rusto-java:0.3.2")
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

<dependency>
    <groupId>com.github.byrizki.rusto-rs</groupId>
    <artifactId>rusto-java</artifactId>
    <version>0.3.2</version>
</dependency>
```

Ensure `librusto` is in your `java.library.path` or `jna.library.path`.

## Quick Start

```java
import com.byrizki.rusto.RustO;
import com.byrizki.rusto.TextResult;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        String config = """
            {
                "detection": { "model_path": "models/PPOCR_v6/det.rten" },
                "recognition": { "model_path": "models/PPOCR_v6/rec.rten", "dict_path": "models/PPOCR_v6/dict.txt" }
            }
            """;

        try (RustO engine = RustO.initialize(config)) {
            // Structured OCR
            List<TextResult> results = engine.detectTextFile("invoice.png", "{}");
            for (TextResult r : results) {
                System.out.printf("[%0.2f] %s%n", r.score(), r.text());
            }

            // Spatial OCR
            String spatial = engine.detectTextFileSpatial("invoice.png", "{}");
            System.out.println("--- Spatial Text ---");
            System.out.println(spatial);
        }
    }
}
```

## 📖 Documentation

For full documentation, guides, and API reference:
- [Java Guide](https://byrizki.com/rusto-rs/how-to/use-java)
- [Full Documentation](https://byrizki.com/rusto-rs/)

## License

MIT License
