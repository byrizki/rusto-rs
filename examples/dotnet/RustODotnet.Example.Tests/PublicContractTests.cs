using System;
using System.Text.Json;
using RustODotnet;
using Xunit;

namespace RustODotnet.Example.Tests;

public sealed class PublicContractTests
{
    [Fact]
    public void PresetFactoriesUseCanonicalTemplates()
    {
        Assert.Equal("indra", InitializeConfig.Indra().Template);
        Assert.Equal("ppv6", InitializeConfig.Ppv6().Template);
        Assert.Equal("ppv5", InitializeConfig.Ppv5().Template);
        Assert.Equal("ppv4", InitializeConfig.Ppv4().Template);
        Assert.Equal("ppv3", InitializeConfig.Ppv3().Template);
    }

    [Fact]
    public void ForModelFactoryUsesCanonicalFields()
    {
        var config = InitializeConfig.ForModel("indraocr", "latin", "/tmp/models");
        Assert.Equal("indraocr", config.Model);
        Assert.Equal("latin", config.Lang);
        Assert.Equal("/tmp/models", config.ModelPath);

        var json = JsonSerializer.Serialize(config);
        Assert.Contains("\"model\":\"indraocr\"", json);
        Assert.Contains("\"lang\":\"latin\"", json);
        Assert.Contains("\"model_path\":\"/tmp/models\"", json);
    }

    [Fact]
    public void RuntimeOptionsUseSixPhaseCamelCaseWireContract()
    {
        var json = JsonSerializer.Serialize(new OcrRunOptions
        {
            AutoUnwarp = true,
            LayoutReading = true,
            OrientationDetection = false,
            LineClassification = true,
            PreDetection = new PreDetectionOptions { DescreenStrength = 0.25f, TargetMaxSide = 1600 },
            Detection = new DetectionRunOptions { LimitSideLen = 960 },
            PostDetection = new PostDetectionOptions
            {
                MinHeight = 24f,
                MaxSideLen = 1600f,
                MinSideLen = 48f,
                UseDilation = true,
            },
            PreRecognition = new PreRecognitionOptions { LineYThreshold = 0.5f, WordXThreshold = 0.4f },
            Recognition = new RecognitionRunOptions { BatchSize = 6 },
            PostRecognition = new PostRecognitionOptions { ScoreThreshold = 0.8f },
        });

        Assert.Contains("\"autoUnwarp\":true", json);
        Assert.Contains("\"layoutReading\":true", json);
        Assert.Contains("\"orientationDetection\":false", json);
        Assert.Contains("\"lineClassification\":true", json);
        Assert.Contains("\"preDetection\":{", json);
        Assert.Contains("\"detection\":{", json);
        Assert.Contains("\"postDetection\":{", json);
        Assert.Contains("\"minHeight\":24", json);
        Assert.Contains("\"maxSideLen\":1600", json);
        Assert.Contains("\"minSideLen\":48", json);
        Assert.Contains("\"useDilation\":true", json);
        Assert.Contains("\"preRecognition\":{", json);
        Assert.Contains("\"lineYThreshold\":0.5", json);
        Assert.Contains("\"wordXThreshold\":0.4", json);
        Assert.Contains("\"recognition\":{", json);
        Assert.Contains("\"batchSize\":6", json);
        Assert.Contains("\"postRecognition\":{\"scoreThreshold\":0.8}", json);
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
    public void OutputGranularitySerializesCanonicalWireValue(OutputGranularity output, string expected)
    {
        var options = new OcrRunOptions { Output = output };
        var json = JsonSerializer.Serialize(options);

        Assert.Equal(expected, options.OutputWireValue);
        Assert.Contains($"\"output\":\"{expected}\"", json);
    }

    [Fact]
    public void ResultBranchesExposeCanonicalValuesAndLayoutMetadata()
    {
        var item = new TextResult
        {
            Text = "Invoice #42",
            Score = 0.99f,
            RegionId = 2,
            LayoutType = "title",
        };
        DetectTextResult structured = new StructuredDetectTextResult(new[] { item });
        DetectTextResult spatial = new SpatialDetectTextResult("invoice total");

        var structuredRes = Assert.IsType<StructuredDetectTextResult>(structured);
        Assert.Single(structuredRes.Items);
        Assert.Equal(2, structuredRes.Items[0].RegionId);
        Assert.Equal("title", structuredRes.Items[0].LayoutType);
        Assert.Equal("invoice total", Assert.IsType<SpatialDetectTextResult>(spatial).Text);
    }
}
