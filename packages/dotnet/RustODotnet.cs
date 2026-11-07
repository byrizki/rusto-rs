using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace RustODotnet
{
    /// <summary>
    /// Represents a 2D coordinate point with single-precision floating-point values in image pixels.
    /// </summary>
    public struct Point2D
    {
        /// <summary>Horizontal coordinate (X).</summary>
        public float X { get; set; }
        /// <summary>Vertical coordinate (Y).</summary>
        public float Y { get; set; }
        
        /// <summary>Initializes a new instance of Point2D.</summary>
        public Point2D(float x, float y) { X = x; Y = y; }
    }

    /// <summary>
    /// Represents an axis-aligned bounding box frame with position and dimensions in image pixels.
    /// </summary>
    public struct Frame
    {
        /// <summary>Width of the bounding box.</summary>
        public float Width { get; set; }
        /// <summary>Height of the bounding box.</summary>
        public float Height { get; set; }
        /// <summary>Top coordinate (Y) of the bounding box.</summary>
        public float Top { get; set; }
        /// <summary>Left coordinate (X) of the bounding box.</summary>
        public float Left { get; set; }

        /// <summary>Initializes a new instance of Frame.</summary>
        public Frame(float width, float height, float top, float left)
        {
            Width = width;
            Height = height;
            Top = top;
            Left = left;
        }
    }

    /// <summary>
    /// Represents a recognized text result with text content, confidence score, bounding polygon, and axis-aligned frame.
    /// </summary>
    public class TextResult
    {
        /// <summary>Recognized text string.</summary>
        public string Text { get; set; }
        /// <summary>Confidence score between 0.0 and 1.0.</summary>
        public float Score { get; set; }
        /// <summary>Four vertex quad points [top-left, top-right, bottom-right, bottom-left].</summary>
        public Point2D[] BoxPoints { get; set; }
        /// <summary>Axis-aligned bounding box frame.</summary>
        public Frame Frame { get; set; }
        /// <summary>Detected document layout region index when layout reading is enabled.</summary>
        public int? RegionId { get; set; }
        /// <summary>Detected semantic layout category when layout reading is enabled.</summary>
        public string LayoutType { get; set; }
    }

    /// <summary>
    /// Output format granularity for text detection.
    /// </summary>
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public enum OutputGranularity
    {
        /// <summary>Group tokens into lines (default).</summary>
        Lines,
        /// <summary>Return individual word tokens.</summary>
        Words,
        /// <summary>Return a reconstructed 2D spatial text layout.</summary>
        Spatial,
        /// <summary>Return formatted TSV (tab-separated values).</summary>
        Tsv
    }

    public class DetectionConfig
    {
        [JsonPropertyName("model_path")]
        public string ModelPath { get; set; }

        [JsonPropertyName("score_mode")]
        public string ScoreMode { get; set; }
    }

    public class RecognitionConfig
    {
        [JsonPropertyName("model_path")]
        public string ModelPath { get; set; }

        [JsonPropertyName("dict_path")]
        public string DictPath { get; set; }

        [JsonPropertyName("rec_batch_num")]
        public int? RecBatchNum { get; set; }
    }

    public class ClassificationConfig
    {
        [JsonPropertyName("model_path")]
        public string ModelPath { get; set; }
    }

    public class OrientationConfig
    {
        [JsonPropertyName("model_path")]
        public string ModelPath { get; set; }
    }

    public class UnwarpConfig
    {
        [JsonPropertyName("model_path")]
        public string ModelPath { get; set; }
    }

    public class LayoutConfig
    {
        [JsonPropertyName("model_path")]
        public string ModelPath { get; set; }

        [JsonPropertyName("conf_thresh")]
        public float? ConfThresh { get; set; }

        [JsonPropertyName("target_size")]
        public int? TargetSize { get; set; }
    }

    /// <summary>
    /// Configuration for engine initialization and model paths.
    /// </summary>
    public class InitializeConfig
    {
        /// <summary>Canonical model identifier (e.g. "ppocrv6-tiny", "indraocr", "ppocrv6-small", "ppocrv5", "ppocrv4").</summary>
        [JsonPropertyName("model")]
        public string Model { get; set; }

        /// <summary>Directory path containing det, rec, and dict model files.</summary>
        [JsonPropertyName("model_path")]
        public string ModelPath { get; set; }

        /// <summary>Language identifier (e.g. "latin", "cyrillic", "ar", "ja", "ko", "devanagari").</summary>
        [JsonPropertyName("lang")]
        public string Lang { get; set; }

        /// <summary>Custom cache directory override path.</summary>
        [JsonPropertyName("cache_dir")]
        public string CacheDir { get; set; }

        /// <summary>Enable neural reading order recovery transformer.</summary>
        [JsonPropertyName("enable_reading_order")]
        public bool? EnableReadingOrder { get; set; }

        /// <summary>Document layout variant: "cdla" (default 10-class) or "publaynet" (5-class).</summary>
        [JsonPropertyName("layout_variant")]
        public string LayoutVariant { get; set; }

        /// <summary>Template preset name (e.g. "ppv6", "ppv5", "ppv4", "ppv3").</summary>
        [JsonPropertyName("template")]
        public string Template { get; set; }

        /// <summary>Detection model configuration.</summary>
        [JsonPropertyName("detection")]
        public DetectionConfig Detection { get; set; }

        /// <summary>Recognition model and dictionary configuration.</summary>
        [JsonPropertyName("recognition")]
        public RecognitionConfig Recognition { get; set; }

        /// <summary>Optional classification model configuration.</summary>
        [JsonPropertyName("classification")]
        public ClassificationConfig Classification { get; set; }

        /// <summary>Optional document orientation model configuration.</summary>
        [JsonPropertyName("orientation")]
        public OrientationConfig Orientation { get; set; }

        /// <summary>Optional document dewarping model configuration.</summary>
        [JsonPropertyName("unwarp")]
        public UnwarpConfig Unwarp { get; set; }

        /// <summary>Optional document layout model configuration.</summary>
        [JsonPropertyName("layout")]
        public LayoutConfig Layout { get; set; }

        /// <summary>Creates a configuration for a canonical model identifier with optional language and model directory.</summary>
        public static InitializeConfig ForModel(string model = "indraocr", string lang = null, string modelPath = null) =>
            new() { Model = model, Lang = lang, ModelPath = modelPath };

        /// <summary>Creates a preset configuration for IndraOCR (high-accuracy default).</summary>
        public static InitializeConfig Indra(string detModelPath = null, string recModelPath = null, string dictPath = null) =>
            CreatePreset("indra", detModelPath, recModelPath, dictPath, recBatchNum: 1);

        /// <summary>Creates a preset configuration for PP-OCRv3.</summary>
        public static InitializeConfig Ppv3(string detModelPath = null, string recModelPath = null, string dictPath = null) =>
            CreatePreset("ppv3", detModelPath, recModelPath, dictPath);

        /// <summary>Creates a preset configuration for PP-OCRv4.</summary>
        public static InitializeConfig Ppv4(string detModelPath = null, string recModelPath = null, string dictPath = null) =>
            CreatePreset("ppv4", detModelPath, recModelPath, dictPath);

        /// <summary>Creates a preset configuration for PP-OCRv5.</summary>
        public static InitializeConfig Ppv5(string detModelPath = null, string recModelPath = null, string dictPath = null) =>
            CreatePreset("ppv5", detModelPath, recModelPath, dictPath);

        /// <summary>Creates a preset configuration for PP-OCRv6 (50+ languages unified).</summary>
        public static InitializeConfig Ppv6(string detModelPath = null, string recModelPath = null, string dictPath = null) =>
            CreatePreset("ppv6", detModelPath, recModelPath, dictPath);

        private static InitializeConfig CreatePreset(string template, string detModelPath, string recModelPath, string dictPath, int? recBatchNum = null)
        {
            var cfg = new InitializeConfig { Template = template };
            if (!string.IsNullOrEmpty(detModelPath))
            {
                cfg.Detection = new DetectionConfig { ModelPath = detModelPath };
            }
            if (!string.IsNullOrEmpty(recModelPath) || !string.IsNullOrEmpty(dictPath))
            {
                cfg.Recognition = new RecognitionConfig
                {
                    ModelPath = recModelPath,
                    DictPath = dictPath,
                    RecBatchNum = recBatchNum
                };
            }
            return cfg;
        }
    }

    /// <summary>
    /// Phase 1: Pre-detection options (descreen, downscale, padding).
    /// </summary>
    public class PreDetectionOptions
    {
        [JsonPropertyName("descreenStrength")]
        public float? DescreenStrength { get; set; }

        [JsonPropertyName("targetMaxSide")]
        public uint? TargetMaxSide { get; set; }

        [JsonPropertyName("cropPaddingX")]
        public float? CropPaddingX { get; set; }

        [JsonPropertyName("cropPaddingY")]
        public float? CropPaddingY { get; set; }
    }

    /// <summary>
    /// Phase 2: Detection inference options (limits, normalization).
    /// </summary>
    public class DetectionRunOptions
    {
        [JsonPropertyName("limitSideLen")]
        public int? LimitSideLen { get; set; }

        [JsonPropertyName("limitType")]
        public string LimitType { get; set; }

        [JsonPropertyName("mean")]
        public float[] Mean { get; set; }

        [JsonPropertyName("std")]
        public float[] Std { get; set; }
    }

    /// <summary>
    /// Phase 3: Post-detection options (thresholds, unclip, dilation, box geometry).
    /// </summary>
    public class PostDetectionOptions
    {
        [JsonPropertyName("threshold")]
        public float? Threshold { get; set; }

        [JsonPropertyName("boxThreshold")]
        public float? BoxThreshold { get; set; }

        [JsonPropertyName("unclipRatio")]
        public float? UnclipRatio { get; set; }

        [JsonPropertyName("useDilation")]
        public bool? UseDilation { get; set; }

        [JsonPropertyName("scoreMode")]
        public string ScoreMode { get; set; }

        [JsonPropertyName("maxCandidates")]
        public int? MaxCandidates { get; set; }

        [JsonPropertyName("minHeight")]
        public float? MinHeight { get; set; }

        [JsonPropertyName("minSideLen")]
        public float? MinSideLen { get; set; }

        [JsonPropertyName("maxSideLen")]
        public float? MaxSideLen { get; set; }

        [JsonPropertyName("widthHeightRatio")]
        public float? WidthHeightRatio { get; set; }
    }

    /// <summary>
    /// Phase 4: Pre-recognition grouping options (line_y, word_x tolerances).
    /// </summary>
    public class PreRecognitionOptions
    {
        [JsonPropertyName("lineYThreshold")]
        public float? LineYThreshold { get; set; }

        [JsonPropertyName("wordXThreshold")]
        public float? WordXThreshold { get; set; }
    }

    /// <summary>
    /// Phase 5: Recognition inference options (batch size).
    /// </summary>
    public class RecognitionRunOptions
    {
        [JsonPropertyName("batchSize")]
        public int? BatchSize { get; set; }
    }

    /// <summary>
    /// Phase 6: Post-recognition filtering options (score threshold).
    /// </summary>
    public class PostRecognitionOptions
    {
        [JsonPropertyName("scoreThreshold")]
        public float? ScoreThreshold { get; set; }
    }

    /// <summary>
    /// Per-request OCR runtime execution options.
    /// </summary>
    public class OcrRunOptions
    {
        [JsonIgnore]
        public OutputGranularity Output { get; set; } = OutputGranularity.Lines;

        [JsonPropertyName("output")]
        public string OutputWireValue => Output.ToString().ToLowerInvariant();

        [JsonPropertyName("autoUnwarp")]
        public bool? AutoUnwarp { get; set; }

        [JsonPropertyName("layoutReading")]
        public bool? LayoutReading { get; set; }

        [JsonPropertyName("orientationDetection")]
        public bool? OrientationDetection { get; set; }

        [JsonPropertyName("lineClassification")]
        public bool? LineClassification { get; set; }

        [JsonPropertyName("preDetection")]
        public PreDetectionOptions PreDetection { get; set; }

        [JsonPropertyName("detection")]
        public DetectionRunOptions Detection { get; set; }

        [JsonPropertyName("postDetection")]
        public PostDetectionOptions PostDetection { get; set; }

        [JsonPropertyName("preRecognition")]
        public PreRecognitionOptions PreRecognition { get; set; }

        [JsonPropertyName("recognition")]
        public RecognitionRunOptions Recognition { get; set; }

        [JsonPropertyName("postRecognition")]
        public PostRecognitionOptions PostRecognition { get; set; }
    }

    /// <summary>
    /// Base class for image input sources.
    /// </summary>
    public abstract class ImageSource { }

    /// <summary>
    /// File path or file: URI image source.
    /// </summary>
    public sealed class UriImageSource : ImageSource
    {
        /// <summary>Filesystem path or URI.</summary>
        public string Value { get; }
        /// <summary>Initializes a new instance of UriImageSource.</summary>
        public UriImageSource(string value) => Value = value ?? throw new ArgumentNullException(nameof(value));
    }

    /// <summary>
    /// In-memory encoded image byte buffer source.
    /// </summary>
    public sealed class BytesImageSource : ImageSource
    {
        /// <summary>Encoded image bytes.</summary>
        public byte[] Value { get; }
        /// <summary>Initializes a new instance of BytesImageSource.</summary>
        public BytesImageSource(byte[] value) => Value = value ?? throw new ArgumentNullException(nameof(value));
    }

    /// <summary>
    /// Base class for OCR text detection results.
    /// </summary>
    public abstract class DetectTextResult { }

    /// <summary>
    /// Structured OCR detection result containing a list of recognized items.
    /// </summary>
    public sealed class StructuredDetectTextResult : DetectTextResult
    {
        /// <summary>List of recognized TextResult items.</summary>
        public IReadOnlyList<TextResult> Items { get; }
        /// <summary>Initializes a new instance of StructuredDetectTextResult.</summary>
        public StructuredDetectTextResult(IReadOnlyList<TextResult> items) => Items = items;
    }

    /// <summary>
    /// Reconstructed 2D spatial text detection result preserving document layout.
    /// </summary>
    public sealed class SpatialDetectTextResult : DetectTextResult
    {
        /// <summary>Reconstructed 2D spatial text string.</summary>
        public string Text { get; }
        /// <summary>Initializes a new instance of SpatialDetectTextResult.</summary>
        public SpatialDetectTextResult(string text) => Text = text;
    }

    /// <summary>
    /// Exception thrown when native OCR operations fail.
    /// </summary>
    public class RustOException : Exception
    {
        /// <summary>Initializes a new instance of RustOException.</summary>
        public RustOException(string message) : base(message) { }
    }

    /// <summary>
    /// Fluent builder for chained text detection configuration and execution.
    /// </summary>
    public class DetectTextBuilder
    {
        private readonly RustO _engine;
        private ImageSource _source;
        private byte[] _rgbPixels;
        private uint _rgbWidth;
        private uint _rgbHeight;
        private ulong _rgbStride;
        private OcrRunOptions _options = new();

        public DetectTextBuilder(RustO engine, ImageSource source = null)
        {
            _engine = engine ?? throw new ArgumentNullException(nameof(engine));
            _source = source;
        }

        public DetectTextBuilder Image(string pathOrUri)
        {
            _source = new UriImageSource(pathOrUri);
            _rgbPixels = null;
            return this;
        }

        public DetectTextBuilder Data(byte[] bytes)
        {
            _source = new BytesImageSource(bytes);
            _rgbPixels = null;
            return this;
        }

        public DetectTextBuilder Rgb(byte[] pixels, uint width, uint height, ulong? stride = null)
        {
            _rgbPixels = pixels ?? throw new ArgumentNullException(nameof(pixels));
            _rgbWidth = width;
            _rgbHeight = height;
            _rgbStride = stride ?? ((ulong)width * 3UL);
            _source = null;
            return this;
        }

        public DetectTextBuilder Source(ImageSource source)
        {
            _source = source ?? throw new ArgumentNullException(nameof(source));
            _rgbPixels = null;
            return this;
        }

        public DetectTextBuilder WithOptions(OcrRunOptions options)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            return this;
        }

        public DetectTextBuilder WithAutoUnwarp(bool enabled = true)
        {
            _options.AutoUnwarp = enabled;
            return this;
        }

        public DetectTextBuilder WithLayoutReading(bool enabled = true)
        {
            _options.LayoutReading = enabled;
            return this;
        }

        public DetectTextBuilder WithOrientationDetection(bool enabled = true)
        {
            _options.OrientationDetection = enabled;
            return this;
        }

        public DetectTextBuilder WithLineClassification(bool enabled = true)
        {
            _options.LineClassification = enabled;
            return this;
        }

        public DetectTextBuilder WithPreDetection(PreDetectionOptions preDetection)
        {
            _options.PreDetection = preDetection;
            return this;
        }

        public DetectTextBuilder WithPreDetection(Action<PreDetectionOptions> configure)
        {
            _options.PreDetection ??= new PreDetectionOptions();
            configure?.Invoke(_options.PreDetection);
            return this;
        }

        public DetectTextBuilder WithDetection(DetectionRunOptions detection)
        {
            _options.Detection = detection;
            return this;
        }

        public DetectTextBuilder WithDetection(Action<DetectionRunOptions> configure)
        {
            _options.Detection ??= new DetectionRunOptions();
            configure?.Invoke(_options.Detection);
            return this;
        }

        public DetectTextBuilder WithPostDetection(PostDetectionOptions postDetection)
        {
            _options.PostDetection = postDetection;
            return this;
        }

        public DetectTextBuilder WithPostDetection(Action<PostDetectionOptions> configure)
        {
            _options.PostDetection ??= new PostDetectionOptions();
            configure?.Invoke(_options.PostDetection);
            return this;
        }

        public DetectTextBuilder WithPreRecognition(PreRecognitionOptions preRecognition)
        {
            _options.PreRecognition = preRecognition;
            return this;
        }

        public DetectTextBuilder WithPreRecognition(Action<PreRecognitionOptions> configure)
        {
            _options.PreRecognition ??= new PreRecognitionOptions();
            configure?.Invoke(_options.PreRecognition);
            return this;
        }

        public DetectTextBuilder WithRecognition(RecognitionRunOptions recognition)
        {
            _options.Recognition = recognition;
            return this;
        }

        public DetectTextBuilder WithRecognition(Action<RecognitionRunOptions> configure)
        {
            _options.Recognition ??= new RecognitionRunOptions();
            configure?.Invoke(_options.Recognition);
            return this;
        }

        public DetectTextBuilder WithPostRecognition(PostRecognitionOptions postRecognition)
        {
            _options.PostRecognition = postRecognition;
            return this;
        }

        public DetectTextBuilder WithPostRecognition(Action<PostRecognitionOptions> configure)
        {
            _options.PostRecognition ??= new PostRecognitionOptions();
            configure?.Invoke(_options.PostRecognition);
            return this;
        }

        public DetectTextBuilder WithOutput(OutputGranularity granularity)
        {
            _options.Output = granularity;
            return this;
        }

        public DetectTextBuilder AsLines() => WithOutput(OutputGranularity.Lines);
        public DetectTextBuilder AsWords() => WithOutput(OutputGranularity.Words);
        public DetectTextBuilder AsSpatial() => WithOutput(OutputGranularity.Spatial);
        public DetectTextBuilder AsTsv() => WithOutput(OutputGranularity.Tsv);

        public DetectTextResult Run()
        {
            if (_rgbPixels != null)
            {
                return _engine.DetectTextRgb(_rgbPixels, _rgbWidth, _rgbHeight, _rgbStride, _options);
            }
            if (_source == null)
            {
                throw new InvalidOperationException("Image source must be specified before calling Run().");
            }
            return _engine.DetectText(_source, _options);
        }
    }

    /// <summary>
    /// High-performance, pure-Rust OCR engine wrapper for .NET.
    /// </summary>
    public class RustO : IDisposable
    {
        private readonly RustODotnet.Generated.RustO _inner;
        private bool _disposed;

        static RustO()
        {
#if NET5_0_OR_GREATER || NETCOREAPP3_0_OR_GREATER
            NativeLibrary.SetDllImportResolver(typeof(RustO).Assembly, DllImportResolver);
#endif
        }

#if NET5_0_OR_GREATER || NETCOREAPP3_0_OR_GREATER
        private static IntPtr DllImportResolver(string libraryName, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (libraryName == "rusto")
            {
                var candidates = new[]
                {
                    "librusto.so",
                    "target/release/librusto.so",
                    "target/debug/librusto.so",
                    "../target/release/librusto.so",
                    "../target/debug/librusto.so",
                    "../../target/release/librusto.so",
                    "../../target/debug/librusto.so",
                    "../../../target/release/librusto.so",
                    "../../../target/debug/librusto.so",
                    "runtimes/linux-x64/native/librusto.so",
                    "runtimes/osx-arm64/native/librusto.dylib",
                    "runtimes/win-x64/native/rusto.dll"
                };
                foreach (var cand in candidates)
                {
                    if (NativeLibrary.TryLoad(cand, out var handle))
                        return handle;
                    var full = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, cand));
                    if (File.Exists(full) && NativeLibrary.TryLoad(full, out handle))
                        return handle;
                }
            }
            return IntPtr.Zero;
        }
#endif

        /// <summary>
        /// Underlying native engine version string.
        /// </summary>
        public static string Version => RustODotnet.Generated.RustoMethods.RustoVersion();

        /// <summary>
        /// Returns whether the specified engine is ready for inference.
        /// </summary>
        public static bool IsReady(RustO engine) => engine != null && engine.IsReady();

        private RustO(InitializeConfig config)
        {
            config ??= new InitializeConfig();

            var hasModelOrPath = !string.IsNullOrEmpty(config.Model) || !string.IsNullOrEmpty(config.ModelPath);

            if (hasModelOrPath)
            {
                if (!string.IsNullOrEmpty(config.ModelPath))
                    config.ModelPath = ResolveModelPath(config.ModelPath);

                if (!string.IsNullOrEmpty(config.Detection?.ModelPath))
                    config.Detection.ModelPath = ResolveModelPath(config.Detection.ModelPath);

                if (!string.IsNullOrEmpty(config.Recognition?.ModelPath))
                    config.Recognition.ModelPath = ResolveModelPath(config.Recognition.ModelPath);

                if (!string.IsNullOrEmpty(config.Recognition?.DictPath))
                    config.Recognition.DictPath = ResolveModelPath(config.Recognition.DictPath);
            }
            else if (config.Detection != null || config.Recognition != null)
            {
                if (!string.IsNullOrEmpty(config.Detection?.ModelPath))
                    config.Detection.ModelPath = ResolveModelPath(config.Detection.ModelPath);

                if (!string.IsNullOrEmpty(config.Recognition?.ModelPath))
                    config.Recognition.ModelPath = ResolveModelPath(config.Recognition.ModelPath);

                if (!string.IsNullOrEmpty(config.Recognition?.DictPath))
                    config.Recognition.DictPath = ResolveModelPath(config.Recognition.DictPath);
            }
            else
            {
                var bundledDet = ResolveModelPath("det.mnn");
                if (!File.Exists(bundledDet)) bundledDet = ResolveModelPath("det.rten");
                var bundledRec = ResolveModelPath("rec.mnn");
                if (!File.Exists(bundledRec)) bundledRec = ResolveModelPath("rec.rten");
                if (File.Exists(bundledDet) && File.Exists(bundledRec))
                {
                    config.Detection = new DetectionConfig { ModelPath = bundledDet };
                    var dictPath = ResolveModelPath("dict.txt");
                    if (!File.Exists(dictPath)) dictPath = ResolveModelPath("meta.json");
                    config.Recognition = new RecognitionConfig
                    {
                        ModelPath = bundledRec,
                        DictPath = dictPath
                    };
                }
            }

            if (config.Classification?.ModelPath != null)
                config.Classification.ModelPath = ResolveModelPath(config.Classification.ModelPath);
            if (config.Orientation?.ModelPath != null)
                config.Orientation.ModelPath = ResolveModelPath(config.Orientation.ModelPath);
            if (config.Unwarp?.ModelPath != null)
                config.Unwarp.ModelPath = ResolveModelPath(config.Unwarp.ModelPath);
            if (config.Layout?.ModelPath != null)
                config.Layout.ModelPath = ResolveModelPath(config.Layout.ModelPath);

            var options = new JsonSerializerOptions
            {
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
            };
            string json = JsonSerializer.Serialize(config, options);
            try
            {
                _inner = RustODotnet.Generated.RustO.InitializeFromJson(json);
            }
            catch (Exception ex)
            {
                throw new RustOException($"Failed to initialize RustO with config: {ex.Message}");
            }
        }

        /// <summary>
        /// Downloads model assets for a specific OCR pipeline ahead of time.
        /// </summary>
        /// <param name="model">Canonical model name (e.g. "ppocrv6-tiny", "indraocr", "ppocrv5").</param>
        /// <param name="lang">Optional language identifier.</param>
        /// <param name="cacheDir">Optional custom cache directory.</param>
        /// <returns>Array of downloaded local model paths.</returns>
        public static string[] DownloadModel(string model = null, string lang = null, string cacheDir = null) =>
            RustODotnet.Generated.RustoMethods.DownloadModel(model, lang, cacheDir);

        /// <summary>
        /// Returns whether the engine is ready for inference.
        /// </summary>
        public bool IsReady() => !_disposed && _inner.IsReady();

        /// <summary>
        /// Returns the engine library version string.
        /// </summary>
        public string GetVersion() => _inner.Version();

        /// <summary>
        /// Initializes a new RustO OCR engine instance with model session configuration.
        /// </summary>
        /// <param name="config">Initialization configuration or preset. Defaults to PP-OCRv6.</param>
        /// <returns>Initialized RustO engine instance.</returns>
        public static RustO Initialize(InitializeConfig config = null) => new RustO(config);

        private static string ResolveModelPath(string path)
        {
            if (Path.IsPathRooted(path) && File.Exists(path))
            {
                return path;
            }

            var appDir = AppContext.BaseDirectory;
            var candidates = new List<string> { path };
            var ext = Path.GetExtension(path);
            if (!string.IsNullOrEmpty(ext))
            {
                var withoutExt = path.Substring(0, path.Length - ext.Length);
                candidates.Add(withoutExt + ".rten");
                candidates.Add(withoutExt + ".onnx");
                candidates.Add(withoutExt + ".mnn");
            }

            foreach (var cand in candidates)
            {
                var modelsPath = Path.Combine(appDir, "models", cand);
                if (File.Exists(modelsPath)) return modelsPath;

                var directPath = Path.Combine(appDir, cand);
                if (File.Exists(directPath)) return directPath;

                if (File.Exists(cand)) return Path.GetFullPath(cand);
            }

            return path;
        }

        /// <summary>
        /// Creates a fluent DetectTextBuilder for text detection and recognition.
        /// </summary>
        public DetectTextBuilder DetectText() => new DetectTextBuilder(this);

        /// <summary>
        /// Creates a fluent DetectTextBuilder with the given image file path or URI.
        /// </summary>
        public DetectTextBuilder DetectText(string pathOrUri) => new DetectTextBuilder(this).Image(pathOrUri);

        /// <summary>
        /// Creates a fluent DetectTextBuilder with the given byte array.
        /// </summary>
        public DetectTextBuilder DetectText(byte[] bytes) => new DetectTextBuilder(this).Data(bytes);

        /// <summary>
        /// Creates a fluent DetectTextBuilder with the given ImageSource.
        /// </summary>
        public DetectTextBuilder DetectText(ImageSource source) => new DetectTextBuilder(this, source);

        /// <summary>
        /// Creates a fluent DetectTextBuilder for text detection and recognition.
        /// </summary>
        public DetectTextBuilder DetectTextBuilder() => new DetectTextBuilder(this);

        /// <summary>
        /// Detects text in an image source (file path/URI or byte array) with the specified runtime options.
        /// </summary>
        /// <param name="source">ImageSource specifying image path or in-memory byte buffer.</param>
        /// <param name="options">Runtime execution options.</param>
        /// <returns>StructuredDetectTextResult or SpatialDetectTextResult depending on output option.</returns>
        public DetectTextResult DetectText(ImageSource source, OcrRunOptions options)
        {
            ThrowIfDisposed();
            options ??= new OcrRunOptions();
            ValidateOptions(options, allowSpatial: true);

            RustODotnet.Generated.ImageSource uniffiSource = source switch
            {
                UriImageSource uri => new RustODotnet.Generated.ImageSource.Path(uri.Value),
                BytesImageSource bytes when bytes.Value != null => new RustODotnet.Generated.ImageSource.Bytes(bytes.Value),
                _ => throw new ArgumentException("ImageSource must contain a URI or image bytes.", nameof(source))
            };

            var uniffiOpts = BuildUniffiOptions(options);

            try
            {
                return MapDetectTextResult(_inner.DetectText(uniffiSource, uniffiOpts));
            }
            catch (Exception ex)
            {
                throw new RustOException($"Text detection failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Detects text from raw decoded interleaved 24-bit RGB pixel buffer.
        /// </summary>
        /// <param name="pixels">Interleaved RGB pixel bytes.</param>
        /// <param name="width">Image width in pixels.</param>
        /// <param name="height">Image height in pixels.</param>
        /// <param name="stride">Row stride in bytes.</param>
        /// <param name="options">Optional runtime execution options.</param>
        /// <returns>StructuredDetectTextResult or SpatialDetectTextResult.</returns>
        public DetectTextResult DetectTextRgb(byte[] pixels, uint width, uint height, ulong stride, OcrRunOptions options = null)
        {
            ThrowIfDisposed();
            options ??= new OcrRunOptions();
            ValidateOptions(options, allowSpatial: true);

            var uniffiOpts = BuildUniffiOptions(options);

            try
            {
                return MapDetectTextResult(_inner.DetectTextRgb(pixels, width, height, stride, uniffiOpts));
            }
            catch (Exception ex)
            {
                throw new RustOException($"Text detection failed: {ex.Message}");
            }
        }

        private static DetectTextResult MapDetectTextResult(RustODotnet.Generated.DetectTextResult result) =>
            result switch
            {
                RustODotnet.Generated.DetectTextResult.Structured structured => new StructuredDetectTextResult(
                    structured.Results.Select(r => new TextResult
                    {
                        Text = r.Text,
                        Score = r.Score,
                        BoxPoints = r.BoxPoints.Select(p => new Point2D(p.X, p.Y)).ToArray(),
                        Frame = new Frame(r.Frame.Width, r.Frame.Height, r.Frame.Top, r.Frame.Left),
                        RegionId = r.RegionId,
                        LayoutType = FormatLayoutType(r.LayoutType)
                    }).ToArray()
                ),
                RustODotnet.Generated.DetectTextResult.Spatial spatial => new SpatialDetectTextResult(spatial.Text),
                _ => throw new RustOException("Unexpected detection result format")
            };

        private static string FormatLayoutType(RustODotnet.Generated.LayoutType? layoutType) =>
            layoutType switch
            {
                null => null,
                RustODotnet.Generated.LayoutType.Text => "text",
                RustODotnet.Generated.LayoutType.Title => "title",
                RustODotnet.Generated.LayoutType.Figure => "figure",
                RustODotnet.Generated.LayoutType.FigureCaption => "figureCaption",
                RustODotnet.Generated.LayoutType.Table => "table",
                RustODotnet.Generated.LayoutType.TableCaption => "tableCaption",
                RustODotnet.Generated.LayoutType.Header => "header",
                RustODotnet.Generated.LayoutType.Footer => "footer",
                RustODotnet.Generated.LayoutType.Reference => "reference",
                RustODotnet.Generated.LayoutType.Equation => "equation",
                _ => null
            };

        private static RustODotnet.Generated.OcrRunOptions BuildUniffiOptions(OcrRunOptions options)
        {
            var defaultOpts = RustODotnet.Generated.RustoMethods.DefaultOcrRunOptions();
            var uniffiOutput = options.Output switch
            {
                OutputGranularity.Words => RustODotnet.Generated.OutputGranularity.Words,
                OutputGranularity.Spatial => RustODotnet.Generated.OutputGranularity.Spatial,
                OutputGranularity.Tsv => RustODotnet.Generated.OutputGranularity.Tsv,
                _ => RustODotnet.Generated.OutputGranularity.Lines
            };

            return defaultOpts with
            {
                Output = uniffiOutput,
                AutoUnwarp = options.AutoUnwarp,
                LayoutReading = options.LayoutReading,
                OrientationDetection = options.OrientationDetection,
                LineClassification = options.LineClassification,
                PreDetection = options.PreDetection != null ? new RustODotnet.Generated.PreDetectionOptions(
                    options.PreDetection.DescreenStrength,
                    options.PreDetection.TargetMaxSide,
                    options.PreDetection.CropPaddingX,
                    options.PreDetection.CropPaddingY
                ) : defaultOpts.PreDetection,
                Detection = options.Detection != null ? new RustODotnet.Generated.DetectionRunOptions(
                    options.Detection.LimitSideLen,
                    options.Detection.LimitType,
                    options.Detection.Mean,
                    options.Detection.Std
                ) : defaultOpts.Detection,
                PostDetection = options.PostDetection != null ? new RustODotnet.Generated.PostDetectionOptions(
                    options.PostDetection.Threshold,
                    options.PostDetection.BoxThreshold,
                    options.PostDetection.UnclipRatio,
                    options.PostDetection.UseDilation,
                    options.PostDetection.ScoreMode,
                    options.PostDetection.MaxCandidates,
                    options.PostDetection.MinHeight,
                    options.PostDetection.MinSideLen,
                    options.PostDetection.MaxSideLen,
                    options.PostDetection.WidthHeightRatio
                ) : defaultOpts.PostDetection,
                PreRecognition = options.PreRecognition != null ? new RustODotnet.Generated.PreRecognitionOptions(
                    options.PreRecognition.LineYThreshold,
                    options.PreRecognition.WordXThreshold
                ) : defaultOpts.PreRecognition,
                Recognition = options.Recognition != null ? new RustODotnet.Generated.RecognitionRunOptions(
                    options.Recognition.BatchSize
                ) : defaultOpts.Recognition,
                PostRecognition = options.PostRecognition != null ? new RustODotnet.Generated.PostRecognitionOptions(
                    options.PostRecognition.ScoreThreshold
                ) : defaultOpts.PostRecognition
            };
        }

        private static void ValidateOptions(OcrRunOptions options, bool allowSpatial)
        {
            if (options == null) throw new ArgumentNullException(nameof(options));
            OutputGranularity output = options.Output;
            if (!allowSpatial && output == OutputGranularity.Spatial)
            {
                throw new RustOException("Structured recognition does not support output 'spatial'. Use DetectText with OutputGranularity.Spatial.");
            }
            if (!allowSpatial && output != OutputGranularity.Lines && output != OutputGranularity.Words)
            {
                throw new RustOException($"Invalid output '{output}'. Use OutputGranularity.Lines or OutputGranularity.Words.");
            }
            if (options.PreDetection != null)
            {
                ValidateProbability(options.PreDetection.DescreenStrength, nameof(options.PreDetection.DescreenStrength));
                ValidateNonNegative(options.PreDetection.CropPaddingX, nameof(options.PreDetection.CropPaddingX));
                ValidateNonNegative(options.PreDetection.CropPaddingY, nameof(options.PreDetection.CropPaddingY));
            }
            if (options.Detection != null)
            {
                var d = options.Detection;
                if (d.LimitSideLen.HasValue && (d.LimitSideLen.Value < 1 || d.LimitSideLen.Value > 32767))
                    throw new RustOException("LimitSideLen must be between 1 and 32767.");
                if (d.LimitType != null && d.LimitType != "min" && d.LimitType != "max")
                    throw new RustOException("LimitType must be min or max.");
                ValidateVector(d.Mean, false, "Mean");
                ValidateVector(d.Std, true, "Std");
            }
            if (options.PostDetection != null)
            {
                var pd = options.PostDetection;
                ValidateProbability(pd.Threshold, "Threshold");
                ValidateProbability(pd.BoxThreshold, "BoxThreshold");
                if (pd.MaxCandidates.HasValue && pd.MaxCandidates.Value < 1)
                    throw new RustOException("MaxCandidates must be at least one.");
                ValidatePositive(pd.UnclipRatio, "UnclipRatio");
                if (pd.ScoreMode != null && pd.ScoreMode != "fast" && pd.ScoreMode != "slow")
                    throw new RustOException("ScoreMode must be fast or slow.");
                ValidatePositive(pd.MinHeight, "MinHeight");
                ValidatePositive(pd.MaxSideLen, "MaxSideLen");
                ValidatePositive(pd.MinSideLen, "MinSideLen");
                if (pd.WidthHeightRatio.HasValue && (!float.IsFinite(pd.WidthHeightRatio.Value) || (pd.WidthHeightRatio.Value <= 0 && pd.WidthHeightRatio.Value != -1)))
                    throw new RustOException("WidthHeightRatio must be positive or -1.");
                if (pd.MinSideLen.HasValue && pd.MaxSideLen.HasValue && pd.MinSideLen.Value > pd.MaxSideLen.Value)
                    throw new RustOException("MinSideLen must not exceed MaxSideLen.");
            }
            if (options.PreRecognition != null)
            {
                ValidateNonNegative(options.PreRecognition.LineYThreshold, nameof(options.PreRecognition.LineYThreshold));
                ValidateNonNegative(options.PreRecognition.WordXThreshold, nameof(options.PreRecognition.WordXThreshold));
            }
            if (options.Recognition != null)
            {
                if (options.Recognition.BatchSize.HasValue && options.Recognition.BatchSize.Value < 1)
                    throw new RustOException("BatchSize must be at least one.");
            }
            if (options.PostRecognition != null)
            {
                ValidateProbability(options.PostRecognition.ScoreThreshold, nameof(options.PostRecognition.ScoreThreshold));
            }
        }

        private static void ValidateVector(float[] values, bool nonZero, string name)
        {
            if (values == null) return;
            if (values.Length != 3 || values.Any(value => !float.IsFinite(value) || (nonZero && value == 0))) throw new RustOException($"{name} must contain three valid values.");
        }

        private static void ValidateProbability(float? value, string name)
        {
            if (value.HasValue && (!float.IsFinite(value.Value) || value.Value < 0 || value.Value > 1)) throw new RustOException($"{name} must be finite and between 0 and 1.");
        }

        private static void ValidatePositive(float? value, string name)
        {
            if (value.HasValue && (!float.IsFinite(value.Value) || value.Value <= 0)) throw new RustOException($"{name} must be finite and greater than zero.");
        }

        private static void ValidateNonNegative(float? value, string name)
        {
            if (value.HasValue && (!float.IsFinite(value.Value) || value.Value < 0))
            {
                throw new RustOException($"{name} must be finite and non-negative.");
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(RustO));
            }
        }

        /// <summary>
        /// Releases native engine resources.
        /// </summary>
        public void Close() => Dispose();

        /// <summary>
        /// Releases native engine resources.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _inner?.Dispose();
                _disposed = true;
            }
        }

        ~RustO()
        {
            Dispose();
        }
    }
}
