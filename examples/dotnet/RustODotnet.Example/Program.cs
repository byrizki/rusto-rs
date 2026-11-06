using System;
using RustODotnet;

var source = new UriImageSource("/absolute/path/invoice.png");
var options = new OcrRunOptions
{
    Output = OutputGranularity.Words,
    PreRecognition = new PreRecognitionOptions
    {
        LineYThreshold = 0.5f,
        WordXThreshold = 0.4f,
    },
};

Console.WriteLine("RustODotnet consumer compiled.");
Console.WriteLine($"Source: {source.Value}; output: {options.OutputWireValue}");
Console.WriteLine("To run OCR, call RustO.Initialize().DetectText(...).Run().");

if (args.Length > 0)
{
    using var ocr = RustO.Initialize();
    var result = ocr.DetectText(args[0])
        .WithPreRecognition(pre =>
        {
            pre.LineYThreshold = 0.5f;
            pre.WordXThreshold = 0.4f;
        })
        .AsLines()
        .Run();

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
