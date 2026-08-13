# RustODotnet

High-performance cross-platform OCR library for .NET based on **RustO!** and **PaddleOCR**, powered by the lightweight **MNN** inference engine.

[![NuGet](https://img.shields.io/nuget/v/RustODotnet.svg)](https://www.nuget.org/packages/RustODotnet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📦 Installation

Install the core runtime package via NuGet:

```bash
dotnet add package RustODotnet
```

### Choose an OCR Model Package

Models are distributed as modular NuGet packages so you can pick the tier that suits your app:

```bash
# Recommended default (~6 MB)
dotnet add package RustODotnet.Models.PPOCRv6.Tiny

# Higher accuracy PP-OCRv6 tiers
dotnet add package RustODotnet.Models.PPOCRv6.Small    # ~30 MB
dotnet add package RustODotnet.Models.PPOCRv6.Medium   # ~134 MB

# PP-OCRv5 tiers
dotnet add package RustODotnet.Models.PPOCRv5.Mobile   # ~28 MB
dotnet add package RustODotnet.Models.PPOCRv5.Server   # ~270 MB

# PP-OCRv4 tiers (with text orientation classifier)
dotnet add package RustODotnet.Models.PPOCRv4.Mobile   # ~23 MB
dotnet add package RustODotnet.Models.PPOCRv4.Server   # ~300 MB
```

> **Note:** When referencing any model package, model files are automatically copied to your output `models/` directory at build time.

---

## 🚀 Quick Start

### 1. Basic OCR (Automatic Model Discovery)

```csharp
using System;
using RustODotnet;

// Initialize engine (automatically discovers models in models/ folder)
using var ocr = new RustO();

// Run OCR on an image file
var results = ocr.RecognizeFile("document.jpg");

foreach (var res in results)
{
    Console.WriteLine($"Text: '{res.Text}' (Confidence: {res.Score:P1})");
    Console.WriteLine($"  Frame: X={res.Frame.Left}, Y={res.Frame.Top}, W={res.Frame.Width}, H={res.Frame.Height}");
}
```

---

### 2. Formatted Spatial Layout Output

Preserve the spatial structure (columns, paragraphs, tables) of documents:

```csharp
using var ocr = new RustO();

// Get text formatted as it appears visually
string spatialText = ocr.RecognizeFileToSpatialText("invoice.png");
Console.WriteLine(spatialText);
```

---

### 3. Advanced Configuration

```csharp
using RustODotnet;

var config = new RustOConfig
{
    Template = "ppv6",
    DetModelPath = "models/det.mnn",
    RecModelPath = "models/rec.mnn",
    DictPath = "models/dict.txt",
    TextScore = 0.5f,
    DetThresh = 0.3f,
    DetBoxThresh = 0.6f,
    LimitSideLen = 960,
    UseDilation = true
};

using var ocr = new RustO(config);
var results = ocr.RecognizeFile("photo.jpg");
```

---

### 4. Recognize from In-Memory Bytes

```csharp
byte[] imageBytes = File.ReadAllBytes("receipt.jpg");

using var ocr = new RustO();
var results = ocr.RecognizeBytes(imageBytes);
```

---

## 🌐 Supported Platforms

Native shared binaries are included out of the box for:
- **Windows**: x64 (`rusto.dll`)
- **Linux**: x64 (`librusto.so`)
- **macOS**: x64 & ARM64 Apple Silicon (`librusto.dylib`)

Target frameworks: `net8.0`, `net6.0`, `netstandard2.1`.

## 📄 License

MIT License.
