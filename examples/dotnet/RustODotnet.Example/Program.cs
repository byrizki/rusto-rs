using RustODotnet;

var source = new UriImageSource("/absolute/path/invoice.png");
var options = new OcrRunOptions
{
    Output = OutputGranularity.Words,
    LineYThreshold = 0.5f,
    WordXThreshold = 0.4f,
};

Console.WriteLine("RustODotnet consumer compiled.");
Console.WriteLine($"Source: {source.Value}; output: {options.OutputWireValue}");
Console.WriteLine("To run OCR, package native runtime assets and matching models, then call RustO.Initialize(...).DetectText(...).");

if (args.Length > 0)
{
    using var ocr = RustO.Initialize(new InitializeConfig { Template = "ppv6" });
    var result = ocr.DetectText(new UriImageSource(args[0]), options);
    switch (result)
    {
        case StructuredDetectTextResult structured:
            foreach (var item in structured.Items) Console.WriteLine(item.Text);
            break;
        case SpatialDetectTextResult spatial:
            Console.WriteLine(spatial.Text);
            break;
    }
}
