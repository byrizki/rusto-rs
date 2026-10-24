using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace RustODotnet
{
    public struct Point2D
    {
        public float X { get; set; }
        public float Y { get; set; }
        
        public Point2D(float x, float y) { X = x; Y = y; }
    }

    public struct Frame
    {
        public float Width { get; set; }
        public float Height { get; set; }
        public float Top { get; set; }
        public float Left { get; set; }

        public Frame(float width, float height, float top, float left)
        {
            Width = width;
            Height = height;
            Top = top;
            Left = left;
        }
    }

    public class TextResult
    {
        public string Text { get; set; }
        public float Score { get; set; }
        public Point2D[] BoxPoints { get; set; }
        public Frame Frame { get; set; }
    }

    [JsonConverter(typeof(JsonStringEnumConverter))]
    public enum OutputGranularity
    {
        Lines,
        Words,
        Spatial
    }

    public class DetectionConfig
    {
        [JsonPropertyName("model_path")]
        public string ModelPath { get; set; }
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

    public class InitializeConfig
    {
        [JsonPropertyName("template")]
        public string Template { get; set; }

        [JsonPropertyName("detection")]
        public DetectionConfig Detection { get; set; }

        [JsonPropertyName("recognition")]
        public RecognitionConfig Recognition { get; set; }

        [JsonPropertyName("classification")]
        public ClassificationConfig Classification { get; set; }

        [JsonPropertyName("orientation")]
        public OrientationConfig Orientation { get; set; }

        [JsonPropertyName("unwarp")]
        public UnwarpConfig Unwarp { get; set; }

        public static InitializeConfig Ppv3() => new() { Template = "ppv3" };
        public static InitializeConfig Ppv4() => new() { Template = "ppv4" };
        public static InitializeConfig Ppv5() => new() { Template = "ppv5" };
        public static InitializeConfig Ppv6() => new() { Template = "ppv6" };
    }

    public class PostprocessRunOptions
    {
        [JsonPropertyName("thresh")] public float? Threshold { get; set; }
        [JsonPropertyName("boxThresh")] public float? BoxThreshold { get; set; }
        [JsonPropertyName("maxCandidates")] public int? MaxCandidates { get; set; }
        [JsonPropertyName("unclipRatio")] public float? UnclipRatio { get; set; }
        [JsonPropertyName("useDilation")] public bool? UseDilation { get; set; }
    }

    public class DetectionRunOptions
    {
        [JsonPropertyName("limitSideLen")] public int? LimitSideLen { get; set; }
        [JsonPropertyName("limitType")] public string LimitType { get; set; }
        [JsonPropertyName("mean")] public float[] Mean { get; set; }
        [JsonPropertyName("std")] public float[] Std { get; set; }
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

    public abstract class ImageSource { }
    public sealed class UriImageSource : ImageSource
    {
        public string Value { get; }
        public UriImageSource(string value) => Value = value ?? throw new ArgumentNullException(nameof(value));
    }
    public sealed class BytesImageSource : ImageSource
    {
        public byte[] Value { get; }
        public BytesImageSource(byte[] value) => Value = value ?? throw new ArgumentNullException(nameof(value));
    }

    public abstract class DetectTextResult { }
    public sealed class StructuredDetectTextResult : DetectTextResult
    {
        public IReadOnlyList<TextResult> Items { get; }
        public StructuredDetectTextResult(IReadOnlyList<TextResult> items) => Items = items;
    }
    public sealed class SpatialDetectTextResult : DetectTextResult
    {
        public string Text { get; }
        public SpatialDetectTextResult(string text) => Text = text;
    }

    public class RustOException : Exception
    {
        public RustOException(string message) : base(message) { }
    }

    public class RustO : IDisposable
    {
        private readonly uniffi.rusto.RustO _inner;
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

        public static string Version => uniffi.rusto.RustoMethods.RustoVersion();

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
                _inner = uniffi.rusto.RustO.InitializeFromJson(json);
            }
            catch (Exception ex)
            {
                throw new RustOException($"Failed to initialize RustO with config: {ex.Message}");
            }
        }

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

        public DetectTextResult DetectText(ImageSource source, OcrRunOptions options = null)
        {
            ThrowIfDisposed();
            options ??= new OcrRunOptions();
            ValidateOptions(options, allowSpatial: true);

            uniffi.rusto.ImageSource uniffiSource = source switch
            {
                UriImageSource uri => new uniffi.rusto.ImageSource.Path(uri.Value),
                BytesImageSource bytes when bytes.Value != null => new uniffi.rusto.ImageSource.Bytes(bytes.Value),
                _ => throw new ArgumentException("ImageSource must contain a URI or image bytes.", nameof(source))
            };

            var uniffiOpts = BuildUniffiOptions(options);

            try
            {
                var result = _inner.DetectText(uniffiSource, uniffiOpts);
                return result switch
                {
                    uniffi.rusto.DetectTextResult.Structured structured => new StructuredDetectTextResult(
                        structured.Results.Select(r => new TextResult
                        {
                            Text = r.Text,
                            Score = r.Score,
                            BoxPoints = r.BoxPoints.Select(p => new Point2D(p.X, p.Y)).ToArray(),
                            Frame = new Frame(r.Frame.Width, r.Frame.Height, r.Frame.Top, r.Frame.Left)
                        }).ToArray()
                    ),
                    uniffi.rusto.DetectTextResult.Spatial spatial => new SpatialDetectTextResult(spatial.Text),
                    _ => throw new RustOException("Unexpected detection result format")
                };
            }
            catch (Exception ex)
            {
                throw new RustOException($"Text detection failed: {ex.Message}");
            }
        }

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
                    uniffi.rusto.DetectTextResult.Structured structured => new StructuredDetectTextResult(
                        structured.Results.Select(r => new TextResult
                        {
                            Text = r.Text,
                            Score = r.Score,
                            BoxPoints = r.BoxPoints.Select(p => new Point2D(p.X, p.Y)).ToArray(),
                            Frame = new Frame(r.Frame.Width, r.Frame.Height, r.Frame.Top, r.Frame.Left)
                        }).ToArray()
                    ),
                    uniffi.rusto.DetectTextResult.Spatial spatial => new SpatialDetectTextResult(spatial.Text),
                    _ => throw new RustOException("Unexpected detection result format")
                };
            }
            catch (Exception ex)
            {
                throw new RustOException($"Text detection failed: {ex.Message}");
            }
        }

        private static uniffi.rusto.OcrRunOptions BuildUniffiOptions(OcrRunOptions options)
        {
            var defaultOpts = uniffi.rusto.RustoMethods.DefaultOcrRunOptions();
            var uniffiOutput = options.Output switch
            {
                OutputGranularity.Words => uniffi.rusto.OutputGranularity.Words,
                OutputGranularity.Spatial => uniffi.rusto.OutputGranularity.Spatial,
                _ => uniffi.rusto.OutputGranularity.Lines
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
                Detection = options.Detection != null ? new uniffi.rusto.DetectionRunOptions(
                    options.Detection.LimitSideLen,
                    options.Detection.LimitType,
                    options.Detection.Mean,
                    options.Detection.Std,
                    null
                ) : defaultOpts.Detection,
                Postprocess = options.Postprocess != null ? new uniffi.rusto.PostprocessRunOptions(
                    options.Postprocess.Threshold,
                    options.Postprocess.BoxThreshold,
                    options.Postprocess.MaxCandidates,
                    options.Postprocess.UnclipRatio,
                    options.Postprocess.UseDilation
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
            if (detection == null) { ValidatePostprocess(options.Postprocess); return; }
            if (detection.LimitSideLen.HasValue && (detection.LimitSideLen.Value < 1 || detection.LimitSideLen.Value > 32767)) throw new RustOException("LimitSideLen must be between 1 and 32767.");
            if (detection.LimitType != null && detection.LimitType != "min" && detection.LimitType != "max") throw new RustOException("LimitType must be min or max.");
            ValidateVector(detection.Mean, false, "Mean");
            ValidateVector(detection.Std, true, "Std");
            ValidatePostprocess(options.Postprocess);
        }

        private static void ValidatePostprocess(PostprocessRunOptions postprocess)
        {
            if (postprocess == null) return;
            ValidateProbability(postprocess.Threshold, "Threshold");
            ValidateProbability(postprocess.BoxThreshold, "BoxThreshold");
            if (postprocess.MaxCandidates.HasValue && postprocess.MaxCandidates.Value < 1) throw new RustOException("MaxCandidates must be at least one.");
            ValidatePositive(postprocess.UnclipRatio, "UnclipRatio");
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
