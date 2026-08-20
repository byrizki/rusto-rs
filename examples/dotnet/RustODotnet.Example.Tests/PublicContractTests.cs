using System.Text.Json;
using RustODotnet;
using Xunit;

namespace RustODotnet.Example.Tests;

public sealed class PublicContractTests
{
    [Fact]
    public void PresetFactoryUsesCanonicalTemplate()
    {
        Assert.Equal("ppv6", InitializeConfig.Ppv6().Template);
    }

    [Fact]
    public void RuntimePreprocessingUsesSharedCamelCaseWireContract()
    {
        var json = JsonSerializer.Serialize(new OcrRunOptions
        {
            Preprocessing = new PreprocessingRunOptions
            {
                MinHeight = 24f,
                MaxSideLen = 1600f,
                MinSideLen = 48f,
                Detection = new DetectionRunOptions
                {
                    Postprocess = new PostprocessRunOptions { UseDilation = true },
                },
            },
        });

        Assert.Contains("\"preprocessing\":{", json);
        Assert.Contains("\"minHeight\":24", json);
        Assert.Contains("\"maxSideLen\":1600", json);
        Assert.Contains("\"minSideLen\":48", json);
        Assert.Contains("\"useDilation\":true", json);
    }

    [Fact]
    public void SourcesRetainCallerValues()
    {
        var uri = new UriImageSource("/tmp/invoice.png");
        var bytes = new BytesImageSource(new byte[] { 1, 2, 3 });

        Assert.Equal("/tmp/invoice.png", uri.Value);
        Assert.Equal(new byte[] { 1, 2, 3 }, bytes.Value);
    }

    [Fact]
    public void SourcesRejectNullValues()
    {
        Assert.Throws<ArgumentNullException>(() => new UriImageSource(null!));
        Assert.Throws<ArgumentNullException>(() => new BytesImageSource(null!));
    }

    [Theory]
    [InlineData(OutputGranularity.Lines, "lines")]
    [InlineData(OutputGranularity.Words, "words")]
    [InlineData(OutputGranularity.Spatial, "spatial")]
    public void RuntimeOptionsUseSharedCamelCaseWireContract(OutputGranularity output, string expected)
    {
        var options = new OcrRunOptions
        {
            Output = output,
            LineYThreshold = 0.5f,
            WordXThreshold = 0.4f,
            TextScore = 0.8f,
            Classification = true,
            Orientation = false,
        };
        var json = JsonSerializer.Serialize(options);

        Assert.Equal(expected, options.OutputWireValue);
        Assert.Contains($"\"output\":\"{expected}\"", json);
        Assert.Contains("\"lineYThreshold\":0.5", json);
        Assert.Contains("\"wordXThreshold\":0.4", json);
        Assert.Contains("\"textScore\":0.8", json);
    }

    [Fact]
    public void ResultBranchesExposeCanonicalValues()
    {
        DetectTextResult structured = new StructuredDetectTextResult(Array.Empty<TextResult>());
        DetectTextResult spatial = new SpatialDetectTextResult("invoice total");

        Assert.Empty(Assert.IsType<StructuredDetectTextResult>(structured).Items);
        Assert.Equal("invoice total", Assert.IsType<SpatialDetectTextResult>(spatial).Text);
    }
}
