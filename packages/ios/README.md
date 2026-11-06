# RustO! iOS SDK (`RustO`)

Official Swift / iOS XCFramework SDK for **RustO!** — a high-performance, pure-Rust OCR engine powered by Indra OCR, PaddleOCR, and the RTen inference runtime.

## Features

- **Zero-Config Indra OCR**: `try RustO.initialize()` uses bundled `det.rten` / `rec.rten` / `dict.txt` resources when present in your app bundle, or automatically downloads and caches the default `indraocr` `.rten` pipeline in `Library/Caches/rusto/models`.
- **Universal XCFramework**: Native `arm64` iOS device and `arm64` / `x86_64` iOS Simulator slices targeting iOS 13.0+.
- **Document Intelligence**: UVDoc document unwarping (`withAutoUnwarp()`), 4-way orientation detection (`withOrientationDetection()`), semantic layout analysis, and neural reading-order recovery (`withLayoutReading()`).
- **Fluent Builder API**: Chain `.asLines()`, `.asWords()`, `.asSpatial()`, or `.withOutput(...)` into a single `try ...run()` call returning Swift's `DetectTextResult` enum.
- **Flexible Image Inputs**: File paths / `file://` URLs, `UIImage` / `Data` (`pngData()` / `jpegData()`), `ImageSource`, and raw 24-bit RGB pixel buffers from `CGImage` via `.rgb(...)`.

## Installation

### Swift Package Manager (Recommended)

Add `rusto-rs` to your `Package.swift` or via Xcode (**File -> Add Package Dependencies...**):

```swift
dependencies: [
    .package(url: "https://github.com/byrizki/rusto-rs.git", from: "0.4.0")
]
```

Then add the `RustO` product to your target dependencies.

### CocoaPods

Add `RustO` to your `Podfile`:

```ruby
platform :ios, '13.0'

target 'YourApp' do
  use_frameworks!
  pod 'RustO', '~> 0.4.0'
end
```

Run `pod install` in your `ios/` directory.

## Quick Start (Zero-Config)

```swift
import RustO

func recognizeInvoice(at imagePath: String) throws {
    // Uses bundled models if present, otherwise auto-downloads Indra OCR into Caches/rusto/models
    let ocr = try RustO.initialize()

    // 1. Structured line-level recognition with unwarping & layout reading order
    let result = try ocr
        .detectText(image: imagePath)
        .withAutoUnwarp()
        .withLayoutReading()
        .asLines()
        .run()

    if case let .structured(results) = result {
        for item in results {
            let layout = item.layoutType.map { "\($0)" } ?? "line"
            let region = item.regionId.map(String.init) ?? "-"
            print(String(format: "[%.2f] (%@ #%@) %@", item.score, layout, region, item.text))
        }
    }

    // 2. Reconstructed 2D spatial plain-text layout
    let spatialResult = try ocr
        .detectText(image: imagePath)
        .asSpatial()
        .run()

    if case let .spatial(text) = spatialResult {
        print(text)
    }
}
```

## Bundled Offline `.rten` Models & Presets

Add your `.rten` and dictionary files (into the main app target's **Copy Bundle Resources** or inside `RustoModels.bundle`) and initialize with `RustO.initializeWithBundledModels(config:)`:

```swift
import RustO

// 1. Default Indra OCR preset (resolves bundled files or auto-downloads)
let indraOcr = try RustO.initializeWithBundledModels(config: .indra())

// 2. Explicit bundled .rten filenames in Bundle.main or RustoModels.bundle
let bundledPpv6 = try RustO.initializeWithBundledModels(
    config: .ppv6(
        detModelPath: "ppocrv6_tiny_det.rten",
        recModelPath: "ppocrv6_tiny_rec.rten",
        dictPath: "ppocrv6_dict.txt"
    )
)

// 3. Canonical model + language (auto-downloaded to Caches/rusto/models)
let multiLangOcr = try RustO.initializeWithBundledModels(
    config: .forModel(model: "ppocrv6-tiny", lang: "latin")
)
```

Pre-warm the model cache ahead of time if desired:

```swift
let cachedFiles = try RustO.downloadModel(model: "indraocr", lang: "latin")
```

## `UIImage`, `CGImage`, `Data`, & Raw RGB8 Input

```swift
import CoreGraphics
import RustO
import UIKit

// 1. UIImage via PNG/JPEG Data
func recognize(uiImage: UIImage, with ocr: RustO) throws -> DetectTextResult {
    guard let data = uiImage.jpegData(compressionQuality: 0.95) ?? uiImage.pngData() else {
        throw RustoBindingError.InvalidInput(msg: "Unable to encode UIImage")
    }
    return try ocr.detectText(data: data)
        .withAutoUnwarp()
        .asLines()
        .run()
}

// 2. CGImage rendered directly to interleaved 24-bit RGB8 pixels (zero image re-encoding)
func recognize(cgImage: CGImage, with ocr: RustO) throws -> DetectTextResult {
    let width = cgImage.width
    let height = cgImage.height
    let rgbaBytesPerRow = width * 4
    var rgba = [UInt8](repeating: 0, count: rgbaBytesPerRow * height)

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: &rgba,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: rgbaBytesPerRow,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    ) else {
        throw RustoBindingError.InvalidInput(msg: "Failed to allocate CGContext")
    }
    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

    var rgb = Data(capacity: width * height * 3)
    for i in stride(from: 0, to: rgba.count, by: 4) {
        rgb.append(rgba[i])     // R
        rgb.append(rgba[i + 1]) // G
        rgb.append(rgba[i + 2]) // B
    }

    return try ocr.detectText()
        .rgb(rgb, width: UInt32(width), height: UInt32(height), stride: UInt64(width * 3))
        .asWords()
        .run()
}
```

## 6-Phase Pipeline Tuning

Configure any stage per call using Swift trailing closures on `DetectTextBuilder`:

```swift
let result = try ocr.detectText(image: imagePath)
    .withOrientationDetection()
    .withAutoUnwarp()
    .withLayoutReading()
    .withPreDetection { $0.descreenStrength = 0.3; $0.targetMaxSide = 1280 }
    .withDetection { $0.limitSideLen = 960; $0.limitType = "max" }
    .withPostDetection { $0.threshold = 0.3; $0.boxThreshold = 0.6; $0.unclipRatio = 1.6 }
    .withPreRecognition { $0.lineYThreshold = 0.5; $0.wordXThreshold = 0.4 }
    .withRecognition { $0.batchSize = 8 }
    .withPostRecognition { $0.scoreThreshold = 0.5 }
    .withOutput(.lines)
    .run()
```

## Result Reference & Memory Management

- `DetectTextResult`:
  - `.structured(results: [TextResult])` — returned by `.asLines()` and `.asWords()`. Each `TextResult` contains:
    - `text: String`
    - `score: Float` (`0.0` – `1.0`)
    - `boxPoints: [Point2D]` (`x`, `y`)
    - `frame: Frame` (`width`, `height`, `top`, `left`)
    - `regionId: Int32?` (layout region index when `withLayoutReading()` is enabled)
    - `layoutType: LayoutType?` (`.text`, `.title`, `.table`, `.header`, `.footer`, `.figure`, `.equation`, etc.)
  - `.spatial(text: String)` — returned by `.asSpatial()`, preserving 2D document column and line layout.
- **ARC Lifecycle**: `RustO` automatically frees its underlying Rust engine handle inside `deinit` when the instance goes out of scope.

## License

Apache-2.0 OR MIT

