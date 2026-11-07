# RustODotnet (.NET SDK)

Official C# / .NET bindings for **RustO!** — a high-performance, pure-Rust OCR engine powered by Indra OCR, PaddleOCR, and pure-Rust MNN (`rusto-nn`) inference runtime.

## Features

- **Zero-Config Indra OCR**: `using var ocr = RustO.Initialize();` automatically downloads and caches the default `indraocr` models on first run from Hugging Face Hub — no separate NuGet model packages required.
- **Standalone Model Support**: Load local `.mnn` or `.rten` models directly for offline or air-gapped deployments, or pre-warm caches via `RustO.DownloadModel(...)`.
- **Document Intelligence**: Built-in UVDoc document unwarping (`WithAutoUnwarp()`), 4-way page orientation detection (`WithOrientationDetection()`), and neural reading-order recovery (`WithLayoutReading()`).
- **Fluent Builder API**: Chain `.AsLines()`, `.AsWords()`, `.AsSpatial()`, `.AsTsv()`, or `.WithOutput(...)` into a single `.Run()` terminal method.
- **Universal Image Inputs**: File paths, `byte[]` encoded images, `ImageSource`, or raw 24-bit interleaved RGB buffers via `.Rgb(...)`.

## Installation

```bash
dotnet add package RustODotnet
```

## Quick Start (Zero-Config)

`RustO.Initialize()` uses the **Indra OCR** pipeline by default and automatically downloads assets from Hugging Face Hub into your user cache (`~/.cache/rusto/models` or `RUSTO_CACHE_DIR`) on first use:

```csharp
using System;
using RustODotnet;

using var ocr = RustO.Initialize();

// Structured line-level recognition with unwarping & layout reading order
var result = ocr.DetectText("invoice.png")
    .WithAutoUnwarp()
    .WithLayoutReading()
    .AsLines()
    .Run();

if (result is StructuredDetectTextResult structured)
{
    foreach (var item in structured.Items)
    {
        Console.WriteLine(
            $"[{item.Score:F2}] ({item.LayoutType ?? "line"} #{item.RegionId?.ToString() ?? "-"}) {item.Text}"
        );
    }
}

// Reconstructed 2D spatial layout
var spatialResult = ocr.DetectText("invoice.png")
    .AsSpatial()
    .Run();

if (spatialResult is SpatialDetectTextResult spatial)
{
    Console.WriteLine(spatial.Text);
}
```

## Presets & Standalone Models (.mnn / .rten)

Configure model presets (`InitializeConfig.Indra()`, `InitializeConfig.Ppv6()`, `InitializeConfig.ForModel(...)`) or point to local `.mnn` / `.rten` model files for offline environments:

```csharp
using RustODotnet;

// 1. Canonical model & language preset (auto-downloads files if missing)
using var indraOcr = RustO.Initialize(InitializeConfig.Indra());
using var tinyOcr = RustO.Initialize(InitializeConfig.ForModel("ppocrv6-tiny", lang: "latin"));

// 2. Offline standalone models (e.g. placed in ./models/ next to your executable)
var offlineConfig = new InitializeConfig
{
    Template = "indra",
    EnableReadingOrder = true,
    Detection = new DetectionConfig { ModelPath = "models/indra_det.mnn" },
    Recognition = new RecognitionConfig
    {
        ModelPath = "models/indra_rec.mnn",
        DictPath = "models/indra_meta.json"
    },
    Unwarp = new UnwarpConfig { ModelPath = "models/uvdoc.mnn" },
    Layout = new LayoutConfig { ModelPath = "models/layout_cdla.mnn" }
};

using var offlineOcr = RustO.Initialize(offlineConfig);
```

> If `det.mnn` (or `det.rten`), `rec.mnn` (or `rec.rten`), and `dict.txt` (or `meta.json`) exist in your application base directory (or `models/` subdirectory), `RustO.Initialize()` automatically resolves and uses those bundled files offline.

### Pre-Downloading Models

For container builds or CI environments, download model assets ahead of time:

```csharp
string[] cachedFiles = RustO.DownloadModel(model: "indraocr", lang: "latin", cacheDir: "./models-cache");
```

## Byte Buffers & Raw RGB Pixels

```csharp
using System.IO;
using RustODotnet;

using var ocr = RustO.Initialize();

// 1. Encoded PNG/JPEG byte array
byte[] imageBytes = File.ReadAllBytes("receipt.png");
var wordsResult = ocr.DetectText(imageBytes).AsWords().Run();

// 2. Raw interleaved 24-bit RGB pixel buffer
uint width = 1280;
uint height = 720;
ulong stride = width * 3;
byte[] rgbPixels = new byte[stride * height];

var rgbResult = ocr.DetectText()
    .Rgb(rgbPixels, width, height, stride)
    .AsLines()
    .Run();
```

## 6-Phase Pipeline Tuning

Customize any stage of the pipeline per call on `DetectTextBuilder`:

```csharp
var tuned = ocr.DetectText("scan.jpg")
    .WithOrientationDetection()
    .WithAutoUnwarp()
    .WithLayoutReading()
    .WithPreDetection(p => p.TargetMaxSide = 1280)
    .WithDetection(d =>
    {
        d.LimitSideLen = 960;
        d.LimitType = "max";
    })
    .WithPostDetection(pd =>
    {
        pd.Threshold = 0.3f;
        pd.BoxThreshold = 0.6f;
        pd.UnclipRatio = 1.6f;
    })
    .WithRecognition(r => r.BatchSize = 8)
    .WithPostRecognition(pr => pr.ScoreThreshold = 0.5f)
    .WithOutput(OutputGranularity.Lines)
    .Run();
```

## Result Types & Lifecycle

- `StructuredDetectTextResult`: Returned for `.AsLines()` and `.AsWords()`. Inspect `structured.Items` (`IReadOnlyList<TextResult>`):
  - `Text` (`string`), `Score` (`float`), `BoxPoints` (`Point2D[]`), `Frame` (`Frame`)
  - `RegionId` (`int?`) and `LayoutType` (`string`: `"text"`, `"title"`, `"table"`, `"header"`, `"footer"`, `"figure"`, etc.) when `WithLayoutReading()` is enabled.
- `SpatialDetectTextResult`: Returned for `.AsSpatial()`. Inspect `spatial.Text` (`string`).
- **Resource Cleanup**: `RustO` implements `IDisposable` — prefer `using var ocr = RustO.Initialize();` or call `ocr.Close()` when finished.

## License

Apache-2.0 OR MIT
