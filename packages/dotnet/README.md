# RustODotnet

```csharp
using RustODotnet;

using var ocr = RustO.Initialize(new InitializeConfig { Template = "ppv6" });
var result = ocr.DetectText(
    new UriImageSource("invoice.jpg"),
    new OcrRunOptions {
        Output = OutputGranularity.Words,
        LineYThreshold = 0.5f,
        WordXThreshold = 0.4f,
    }
);

if (result is StructuredDetectTextResult structured)
    foreach (var text in structured.Items) Console.WriteLine(text.Text);

var spatial = ocr.DetectText(
    new BytesImageSource(File.ReadAllBytes("invoice.jpg")),
    new OcrRunOptions { Output = OutputGranularity.Spatial }
);
if (spatial is SpatialDetectTextResult formatted) Console.WriteLine(formatted.Text);
```

Public API: `RustO.Initialize`, `DetectText`, `ImageSource`, `OcrRunOptions`,
`DetectTextResult`, and `IDisposable.Dispose`. `lines` and `words` yield
`StructuredDetectTextResult`; `spatial` yields `SpatialDetectTextResult`.
