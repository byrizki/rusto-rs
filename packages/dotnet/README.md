# RustODotnet

`RustODotnet` exposes canonical RustO OCR API for .NET. Initialize model session once, run many request-local detections, then dispose engine.

```bash
dotnet add package RustODotnet
# Add matching language-model package when needed:
# dotnet add package RustODotnet.Models.PPOCRv6.Tiny
```

## Complete example

```csharp
using System;
using System.IO;
using RustODotnet;

var config = InitializeConfig.Ppv6();
using var ocr = RustO.Initialize(config);

var result = ocr.DetectText(
    new UriImageSource("invoice.jpg"),
    new OcrRunOptions
    {
        Output = OutputGranularity.Words,
        TextScore = 0.55f,
        MaxSideLen = 1600f,
        Detection = new DetectionRunOptions
        {
            LimitSideLen = 960,
            LimitType = "max",
        },
        Postprocess = new PostprocessRunOptions
        {
            UseDilation = true,
        },
    });

switch (result)
{
    case StructuredDetectTextResult structured:
        foreach (var item in structured.Items)
            Console.WriteLine($"{item.Text} ({item.Score:F2}) at {item.Frame.Left}, {item.Frame.Top}");
        break;
    case SpatialDetectTextResult spatial:
        Console.WriteLine(spatial.Text);
        break;
}
```

Example structured output:

```text
Total (0.98) at 40, 120
$12.50 (0.96) at 140, 120
```

## Public API

| API | Purpose |
|---|---|
| `RustO.Initialize(InitializeConfig? config)` | Creates engine and loads model resources. |
| `ocr.DetectText(ImageSource, OcrRunOptions? options)` | Runs lines, words, or spatial OCR. |
| `UriImageSource(string)` | Path or `file:` URI input. |
| `BytesImageSource(byte[])` | Encoded image bytes input. |
| `StructuredDetectTextResult` | `Items: IReadOnlyList<TextResult>` for `Lines` and `Words`. |
| `SpatialDetectTextResult` | `Text: string` for `Spatial`. |
| `RustO.Dispose()` | Releases native engine. Prefer `using var`. |

## Initialization config

Initialization selects model preset/resources. It does not control per-image preprocessing.

```csharp
using var ocr = RustO.Initialize(InitializeConfig.Ppv6(
    detection: new DetectionConfig { ModelPath = "models/det.mnn" },
    recognition: new RecognitionConfig
    {
        ModelPath = "models/rec.mnn",
        DictPath = "models/dict.txt",
    }));
```

| Field | Type | Meaning |
|---|---|---|
| `Template` | `"ppv6"`, `"ppv5"`, `"ppv4"`, `"ppv3"` | Preset defaults. Use `InitializeConfig.Ppv6()` etc. when possible. |
| `Detection.ModelPath` | `string` | Detection model; default logical name `det.mnn`. |
| `Recognition.ModelPath` | `string` | Recognition model; default logical name `rec.mnn`. |
| `Recognition.DictPath` | `string` | Dictionary; default logical name `dict.txt`. |
| `Classification` / `Orientation` | optional config | Optional auxiliary model paths. Enable respective request option to use. |

Model resolution checks model-package content and common application/model paths. For custom resources, pass explicit paths. Core and model NuGet package versions must match.

## Request-local options

`OcrRunOptions` applies only to one `DetectText` call. Unset properties merge with initialized defaults and do not mutate next call.

```csharp
var options = new OcrRunOptions
{
    Output = OutputGranularity.Words,
    LineYThreshold = 0.5f,
    WordXThreshold = 0.4f,
    TextScore = 0.55f,
    MaxSideLen = 1600f,
    Detection = new DetectionRunOptions
    {
        LimitSideLen = 960,
        LimitType = "max",
        Mean = new[] { 0.485f, 0.456f, 0.406f },
        Std = new[] { 0.229f, 0.224f, 0.225f },
    },
    Postprocess = new PostprocessRunOptions
    {
        Threshold = 0.3f,
        BoxThreshold = 0.6f,
        MaxCandidates = 1000,
        UnclipRatio = 2.0f,
        UseDilation = true,
    },
};
```

| Property | Valid values | Meaning |
|---|---|---|
| `Output` | `Lines`, `Words`, `Spatial` | Default `Lines`; structured collection for first two, string for spatial. |
| `LineYThreshold`, `WordXThreshold` | finite `>= 0` | Line and word/spatial grouping tolerance. |
| `TextScore` | finite `[0, 1]` | Recognition-confidence cutoff. |
| `Classification`, `Orientation` | `bool` | Per-request optional stages; models must be configured first. |
| `MinHeight`, `MaxSideLen`, `MinSideLen` | finite `> 0` | Request-local resize bounds. `MinSideLen` cannot exceed `MaxSideLen` when both set. |
| `WidthHeightRatio` | finite `> 0` or `-1` | Recognition padding ratio. |
| `Detection.LimitSideLen` | integer `1..32767` | Detector resize target/bound. |
| `Detection.LimitType` | `"min"` / `"max"` | Short-side minimum / long-side cap. |
| `Detection.Mean`, `Std` | three finite values | RGB normalization; `Std` values non-zero. |
| `Postprocess.Threshold`, `BoxThreshold` | finite `[0, 1]` | Binarization and polygon confidence. |
| `Postprocess.MaxCandidates` | integer `>= 1` | Candidate limit. |
| `Postprocess.UnclipRatio` | finite `> 0` | Polygon expansion. |
| `Postprocess.UseDilation` | `bool` | Helps faint or broken strokes. |

`Detection` and `Postprocess` are root-level `OcrRunOptions` properties. No nested `Preprocessing` API exists.

## Sources and output

```csharp
var fromPath = ocr.DetectText(new UriImageSource("/tmp/receipt.png"));
var fromBytes = ocr.DetectText(new BytesImageSource(File.ReadAllBytes("receipt.png")));

var spatial = ocr.DetectText(
    new UriImageSource("table.png"),
    new OcrRunOptions { Output = OutputGranularity.Spatial });

if (spatial is SpatialDetectTextResult layout)
    Console.WriteLine(layout.Text);
// Item                 Amount
// Coffee                $3.50
// Total                $12.50
```

`TextResult` has `Text`, `Score`, `BoxPoints`, and `Frame`. `BoxPoints` contains top-left, top-right, bottom-right, bottom-left image-pixel coordinates. `Frame` is axis-aligned with `Left`, `Top`, `Width`, `Height`.

## Errors and lifecycle

- `RustOException` indicates native initialization, model, image decode, option, or OCR failure.
- Constructors reject null source payloads. Validate input file/resource availability before detection.
- Always dispose `RustO`; `using var` handles success and failure paths.
- Invalid request options do not require reinitialization. Correct options and call `DetectText` again.
