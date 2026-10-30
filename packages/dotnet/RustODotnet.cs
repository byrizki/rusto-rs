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
        Spatial
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

    /// <summary>
    /// Configuration for engine initialization and model paths.
    /// </summary>
    public class InitializeConfig
    {
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

        /// <summary>Creates a preset configuration for PP-OCRv3.</summary>
        public static InitializeConfig Ppv3() => new() { Template = "ppv3" };
        /// <summary>Creates a preset configuration for PP-OCRv4.</summary>
        public static InitializeConfig Ppv4() => new() { Template = "ppv4" };
        /// <summary>Creates a preset configuration for PP-OCRv5.</summary>
        public static InitializeConfig Ppv5() => new() { Template = "ppv5" };
        /// <summary>Creates a preset configuration for PP-OCRv6 (50+ languages unified).</summary>
        public static InitializeConfig Ppv6() => new() { Template = "ppv6" };
    }

    public class PostprocessRunOptions
    {
        [JsonPropertyName("thresh")] public float? Threshold { get; set; }
        [JsonPropertyName("boxThresh")] public float? BoxThreshold { get; set; }
        [JsonPropertyName("maxCandidates")] public int? MaxCandidates { get; set; }
        [JsonPropertyName("unclipRatio")] public float? UnclipRatio { get; set; }
        [JsonPropertyName("useDilation")] public bool? UseDilation { get; set; }
        [JsonPropertyName("scoreMode")] public string ScoreMode { get; set; }
    }

    public class DetectionRunOptions
    {
        [JsonPropertyName("limitSideLen")] public int? LimitSideLen { get; set; }
        [JsonPropertyName("limitType")] public string LimitType { get; set; }
        [JsonPropertyName("mean")] public float[] Mean { get; set; }
        [JsonPropertyName("std")] public float[] Std { get; set; }
        [JsonPropertyName("postprocess")] public PostprocessRunOptions Postprocess { get; set; }
    }

    public class OcrRunOptions
    {
        [JsonIgnore]
        public OutputGranularity Output { get; set; } = OutputGranularity.Lines;

        [JsonPropertyName("output")]
        public string OutputWireValue => Output.ToString().ToLowerInvariant();

        [JsonPropertyName("lineYThreshold")]
        public float? LineYThreshold { get; set; }
        [JsonPropertyName("wordXThreshold")]
        public float? WordXThreshold { get; set; }
        [JsonPropertyName("textScore")]
        public float? TextScore { get; set; }
        [JsonPropertyName("classification")]
        public bool? Classification { get; set; }
        [JsonPropertyName("orientation")]
        public bool? Orientation { get; set; }
        [JsonPropertyName("minHeight")] public float? MinHeight { get; set; }
        [JsonPropertyName("maxSideLen")] public float? MaxSideLen { get; set; }
        [JsonPropertyName("minSideLen")] public float? MinSideLen { get; set; }
        [JsonPropertyName("widthHeightRatio")] public float? WidthHeightRatio { get; set; }
        [JsonPropertyName("detection")] public DetectionRunOptions Detection { get; set; }
        [JsonPropertyName("postprocess")] public PostprocessRunOptions Postprocess { get; set; }
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

        private RustO(InitializeConfig config)
        {
            config ??= new InitializeConfig();
            config.Detection ??= new DetectionConfig();
            config.Recognition ??= new RecognitionConfig();

            var detName = config.Detection.ModelPath ?? "det.rten";
            var recName = config.Recognition.ModelPath ?? "rec.rten";
            var dictName = config.Recognition.DictPath ?? "dict.txt";

            config.Detection.ModelPath = ResolveModelPath(detName);
            config.Recognition.ModelPath = ResolveModelPath(recName);
            config.Recognition.DictPath = ResolveModelPath(dictName);

            if (config.Classification?.ModelPath != null)
                config.Classification.ModelPath = ResolveModelPath(config.Classification.ModelPath);
            if (config.Orientation?.ModelPath != null)
                config.Orientation.ModelPath = ResolveModelPath(config.Orientation.ModelPath);
            if (config.Unwarp?.ModelPath != null)
                config.Unwarp.ModelPath = ResolveModelPath(config.Unwarp.ModelPath);

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
        /// Detects text in an image source (file path/URI or byte array) with the specified runtime options.
        /// </summary>
        /// <param name="source">ImageSource specifying image path or in-memory byte buffer.</param>
        /// <param name="options">Optional runtime execution options.</param>
        /// <returns>StructuredDetectTextResult or SpatialDetectTextResult depending on output option.</returns>
        public DetectTextResult DetectText(ImageSource source, OcrRunOptions options = null)
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
                var result = _inner.DetectText(uniffiSource, uniffiOpts);
                return result switch
                {
                    RustODotnet.Generated.DetectTextResult.Structured structured => new StructuredDetectTextResult(
                        structured.Results.Select(r => new TextResult
                        {
                            Text = r.Text,
                            Score = r.Score,
                            BoxPoints = r.BoxPoints.Select(p => new Point2D(p.X, p.Y)).ToArray(),
                            Frame = new Frame(r.Frame.Width, r.Frame.Height, r.Frame.Top, r.Frame.Left)
                        }).ToArray()
                    ),
                    RustODotnet.Generated.DetectTextResult.Spatial spatial => new SpatialDetectTextResult(spatial.Text),
                    _ => throw new RustOException("Unexpected detection result format")
                };
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
                var result = _inner.DetectTextRgb(pixels, width, height, stride, uniffiOpts);
                return result switch
                {
                    RustODotnet.Generated.DetectTextResult.Structured structured => new StructuredDetectTextResult(
                        structured.Results.Select(r => new TextResult
                        {
                            Text = r.Text,
                            Score = r.Score,
                            BoxPoints = r.BoxPoints.Select(p => new Point2D(p.X, p.Y)).ToArray(),
                            Frame = new Frame(r.Frame.Width, r.Frame.Height, r.Frame.Top, r.Frame.Left)
                        }).ToArray()
                    ),
                    RustODotnet.Generated.DetectTextResult.Spatial spatial => new SpatialDetectTextResult(spatial.Text),
                    _ => throw new RustOException("Unexpected detection result format")
                };
            }
            catch (Exception ex)
            {
                throw new RustOException($"Text detection failed: {ex.Message}");
            }
        }

        private static RustODotnet.Generated.OcrRunOptions BuildUniffiOptions(OcrRunOptions options)
        {
            var defaultOpts = RustODotnet.Generated.RustoMethods.DefaultOcrRunOptions();
            var uniffiOutput = options.Output switch
            {
                OutputGranularity.Words => RustODotnet.Generated.OutputGranularity.Words,
                OutputGranularity.Spatial => RustODotnet.Generated.OutputGranularity.Spatial,
                _ => RustODotnet.Generated.OutputGranularity.Lines
            };

            return defaultOpts with
            {
                Output = uniffiOutput,
                LineYThreshold = options.LineYThreshold,
                WordXThreshold = options.WordXThreshold,
                TextScore = options.TextScore,
                Classification = options.Classification,
                Orientation = options.Orientation,
                MinHeight = options.MinHeight,
                MaxSideLen = options.MaxSideLen,
                MinSideLen = options.MinSideLen,
                WidthHeightRatio = options.WidthHeightRatio,
                Detection = options.Detection != null ? new RustODotnet.Generated.DetectionRunOptions(
                    options.Detection.LimitSideLen,
                    options.Detection.LimitType,
                    options.Detection.Mean,
                    options.Detection.Std,
                    options.Detection.Postprocess != null ? new RustODotnet.Generated.PostprocessRunOptions(
                        options.Detection.Postprocess.Threshold,
                        options.Detection.Postprocess.BoxThreshold,
                        options.Detection.Postprocess.MaxCandidates,
                        options.Detection.Postprocess.UnclipRatio,
                        options.Detection.Postprocess.UseDilation,
                        options.Detection.Postprocess.ScoreMode
                    ) : null
                ) : defaultOpts.Detection,
                Postprocess = options.Postprocess != null ? new RustODotnet.Generated.PostprocessRunOptions(
                    options.Postprocess.Threshold,
                    options.Postprocess.BoxThreshold,
                    options.Postprocess.MaxCandidates,
                    options.Postprocess.UnclipRatio,
                    options.Postprocess.UseDilation,
                    options.Postprocess.ScoreMode
                ) : defaultOpts.Postprocess
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
            ValidateNonNegative(options.LineYThreshold, nameof(options.LineYThreshold));
            ValidateNonNegative(options.WordXThreshold, nameof(options.WordXThreshold));
            if (options.TextScore.HasValue && (!float.IsFinite(options.TextScore.Value) || options.TextScore.Value < 0 || options.TextScore.Value > 1))
            {
                throw new RustOException("TextScore must be finite and between 0 and 1.");
            }
            ValidatePositive(options.MinHeight, nameof(options.MinHeight));
            ValidatePositive(options.MaxSideLen, nameof(options.MaxSideLen));
            ValidatePositive(options.MinSideLen, nameof(options.MinSideLen));
            if (options.WidthHeightRatio.HasValue && (!float.IsFinite(options.WidthHeightRatio.Value) || (options.WidthHeightRatio.Value <= 0 && options.WidthHeightRatio.Value != -1))) throw new RustOException("WidthHeightRatio must be positive or -1.");
            if (options.MinSideLen.HasValue && options.MaxSideLen.HasValue && options.MinSideLen.Value > options.MaxSideLen.Value) throw new RustOException("MinSideLen must not exceed MaxSideLen.");
            var detection = options.Detection;
            if (detection != null)
            {
                if (detection.LimitSideLen.HasValue && (detection.LimitSideLen.Value < 1 || detection.LimitSideLen.Value > 32767)) throw new RustOException("LimitSideLen must be between 1 and 32767.");
                if (detection.LimitType != null && detection.LimitType != "min" && detection.LimitType != "max") throw new RustOException("LimitType must be min or max.");
                ValidateVector(detection.Mean, false, "Mean");
                ValidateVector(detection.Std, true, "Std");
                ValidatePostprocess(detection.Postprocess);
            }
            ValidatePostprocess(options.Postprocess);
        }

        private static void ValidatePostprocess(PostprocessRunOptions postprocess)
        {
            if (postprocess == null) return;
            ValidateProbability(postprocess.Threshold, "Threshold");
            ValidateProbability(postprocess.BoxThreshold, "BoxThreshold");
            if (postprocess.MaxCandidates.HasValue && postprocess.MaxCandidates.Value < 1) throw new RustOException("MaxCandidates must be at least one.");
            ValidatePositive(postprocess.UnclipRatio, "UnclipRatio");
            if (postprocess.ScoreMode != null && postprocess.ScoreMode != "fast" && postprocess.ScoreMode != "slow") throw new RustOException("ScoreMode must be fast or slow.");
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
