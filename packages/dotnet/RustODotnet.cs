using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace RustODotnet
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct CTextResult
    {
        public IntPtr Text;
        public float Score;
        public float BoxX1, BoxY1, BoxX2, BoxY2;
        public float BoxX3, BoxY3, BoxX4, BoxY4;
        public float FrameWidth, FrameHeight, FrameTop, FrameLeft;
    }

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

        internal static TextResult FromNative(CTextResult native)
        {
            return new TextResult
            {
                Text = Marshal.PtrToStringUTF8(native.Text) ?? string.Empty,
                Score = native.Score,
                BoxPoints = new[]
                {
                    new Point2D(native.BoxX1, native.BoxY1),
                    new Point2D(native.BoxX2, native.BoxY2),
                    new Point2D(native.BoxX3, native.BoxY3),
                    new Point2D(native.BoxX4, native.BoxY4),
                },
                Frame = new Frame(native.FrameWidth, native.FrameHeight, native.FrameTop, native.FrameLeft)
            };
        }
    }

    public class DetectionConfig
    {
        [JsonPropertyName("enabled")]
        public bool? Enabled { get; set; }
        [JsonPropertyName("modelPath")]
        public string ModelPath { get; set; }
        [JsonPropertyName("thresh")]
        public float? Thresh { get; set; }
        [JsonPropertyName("boxThresh")]
        public float? BoxThresh { get; set; }
        [JsonPropertyName("unclipRatio")]
        public float? UnclipRatio { get; set; }
        [JsonPropertyName("limitSideLen")]
        public int? LimitSideLen { get; set; }
        [JsonPropertyName("limitType")]
        public string LimitType { get; set; }
        [JsonPropertyName("useDilation")]
        public bool? UseDilation { get; set; }
    }

    public class RecognitionConfig
    {
        [JsonPropertyName("enabled")]
        public bool? Enabled { get; set; }
        [JsonPropertyName("modelPath")]
        public string ModelPath { get; set; }
        [JsonPropertyName("dictPath")]
        public string DictPath { get; set; }
        [JsonPropertyName("scoreThresh")]
        public float? ScoreThresh { get; set; }
        [JsonPropertyName("returnWordBox")]
        public bool? ReturnWordBox { get; set; }
        [JsonPropertyName("returnSingleCharBox")]
        public bool? ReturnSingleCharBox { get; set; }
    }

    /// <summary>
    /// Line Classification Configuration (CLS)
    /// NOTE: Text line orientation classifier (180° rotation) is ONLY available on PP-OCRv4 and PP-OCRv5.
    /// </summary>
    public class ClassificationConfig
    {
        [JsonPropertyName("enabled")]
        public bool? Enabled { get; set; }
        [JsonPropertyName("modelPath")]
        public string ModelPath { get; set; }
        [JsonPropertyName("thresh")]
        public float? Thresh { get; set; }
    }

    public class OrientationConfig
    {
        [JsonPropertyName("enabled")]
        public bool? Enabled { get; set; }
        [JsonPropertyName("modelPath")]
        public string ModelPath { get; set; }
        [JsonPropertyName("thresh")]
        public float? Thresh { get; set; }
    }

    public class UnwarpConfig
    {
        [JsonPropertyName("enabled")]
        public bool? Enabled { get; set; }
        [JsonPropertyName("modelPath")]
        public string ModelPath { get; set; }
    }

    public class LayoutConfig
    {
        [JsonPropertyName("yThresholdMultiplier")]
        public float? YThresholdMultiplier { get; set; }
        [JsonPropertyName("xThresholdMultiplier")]
        public float? XThresholdMultiplier { get; set; }
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
        [JsonPropertyName("layout")]
        public LayoutConfig Layout { get; set; }

        public InitializeConfig() { }

        public static InitializeConfig Ppv6(DetectionConfig detection = null, RecognitionConfig recognition = null) =>
            new InitializeConfig { Template = "ppv6", Detection = detection, Recognition = recognition };

        public static InitializeConfig Ppv5(DetectionConfig detection = null, RecognitionConfig recognition = null) =>
            new InitializeConfig { Template = "ppv5", Detection = detection, Recognition = recognition };

        public static InitializeConfig Ppv4(DetectionConfig detection = null, RecognitionConfig recognition = null) =>
            new InitializeConfig { Template = "ppv4", Detection = detection, Recognition = recognition };

        public static InitializeConfig Ppv3(DetectionConfig detection = null, RecognitionConfig recognition = null) =>
            new InitializeConfig { Template = "ppv3", Detection = detection, Recognition = recognition };
    }

    /// <summary>
    /// Per-request OCR output and pipeline options. These options do not alter
    /// engine configuration or later requests.
    /// </summary>
    public enum OutputGranularity
    {
        Lines,
        Words,
        Spatial,
    }

    public class PostprocessRunOptions
    {
        [JsonPropertyName("threshold")] public float? Threshold { get; set; }
        [JsonPropertyName("boxThreshold")] public float? BoxThreshold { get; set; }
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
        [JsonPropertyName("postprocess")] public PostprocessRunOptions Postprocess { get; set; }
    }

    public class PreprocessingRunOptions
    {
        [JsonPropertyName("minHeight")] public float? MinHeight { get; set; }
        [JsonPropertyName("maxSideLen")] public float? MaxSideLen { get; set; }
        [JsonPropertyName("minSideLen")] public float? MinSideLen { get; set; }
        [JsonPropertyName("widthHeightRatio")] public float? WidthHeightRatio { get; set; }
        [JsonPropertyName("detection")] public DetectionRunOptions Detection { get; set; }
    }

    /// <summary>Per-request OCR output and pipeline options.</summary>
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
        [JsonPropertyName("preprocessing")]
        public PreprocessingRunOptions Preprocessing { get; set; }
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
        private IntPtr _handle;
        private bool _disposed;

        public static string Version
        {
            get
            {
                IntPtr versionPtr = rocr_version();
                return Marshal.PtrToStringUTF8(versionPtr) ?? "unknown";
            }
        }

        private RustO(InitializeConfig config)
        {
            config ??= new InitializeConfig();
            config.Detection ??= new DetectionConfig();
            config.Recognition ??= new RecognitionConfig();

            var detName = config.Detection.ModelPath ?? "det.mnn";
            var recName = config.Recognition.ModelPath ?? "rec.mnn";
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
            _handle = rocr_initialize(json);
            if (_handle == IntPtr.Zero)
            {
                throw new RustOException("Failed to initialize RustO with config");
            }
        }
        public static RustO Initialize(InitializeConfig config = null) => new RustO(config);

        private const string LibName = "rusto";

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rocr_initialize(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string configJson
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int rocr_detect_text_file(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string imagePath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string optionsJson,
            out IntPtr results,
            out nuint count
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rocr_detect_text_file_spatial(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string imagePath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string optionsJson
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int rocr_detect_text_data(
            IntPtr handle,
            byte[] imageData,
            nuint imageLen,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string optionsJson,
            out IntPtr results,
            out nuint count
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rocr_detect_text_data_spatial(
            IntPtr handle,
            byte[] imageData,
            nuint imageLen,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string optionsJson
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void rocr_free_string(IntPtr str);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void rocr_free_results(IntPtr results, nuint count);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void rocr_free(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rocr_version();

        private static string ResolveModelPath(string path)
        {
            if (Path.IsPathRooted(path) && File.Exists(path))
            {
                return path;
            }

            var appDir = AppContext.BaseDirectory;
            var modelsPath = Path.Combine(appDir, "models", path);
            if (File.Exists(modelsPath))
            {
                return modelsPath;
            }

            var directPath = Path.Combine(appDir, path);
            if (File.Exists(directPath))
            {
                return directPath;
            }

            if (File.Exists(path))
            {
                return Path.GetFullPath(path);
            }

            return path;
        }

        public DetectTextResult DetectText(ImageSource source, OcrRunOptions options = null)
        {
            ThrowIfDisposed();
            options ??= new OcrRunOptions();
            return options.Output == OutputGranularity.Spatial
                ? new SpatialDetectTextResult(DetectSpatialText(source, options))
                : new StructuredDetectTextResult(DetectStructuredText(source, options));
        }

        private List<TextResult> DetectStructuredText(ImageSource source, OcrRunOptions options)
        {
            string optionsJson = SerializeStructuredOptions(options);
            int status;
            IntPtr resultsPtr;
            nuint count;
            switch (source)
            {
                case UriImageSource uri:
                    status = rocr_detect_text_file(_handle, uri.Value, optionsJson, out resultsPtr, out count);
                    break;
                case BytesImageSource bytes when bytes.Value != null:
                    status = rocr_detect_text_data(_handle, bytes.Value, (nuint)bytes.Value.Length, optionsJson, out resultsPtr, out count);
                    break;
                default:
                    throw new ArgumentException("ImageSource must contain a URI or image bytes.", nameof(source));
            }
            if (status != 0) throw new RustOException($"Text detection failed with status code: {status}");
            try { return MarshalResults(resultsPtr, checked((int)count)); }
            finally { rocr_free_results(resultsPtr, count); }
        }

        private string DetectSpatialText(ImageSource source, OcrRunOptions options)
        {
            string optionsJson = SerializeSpatialOptions(options);
            IntPtr textPtr = source switch
            {
                UriImageSource uri => rocr_detect_text_file_spatial(_handle, uri.Value, optionsJson),
                BytesImageSource bytes when bytes.Value != null => rocr_detect_text_data_spatial(_handle, bytes.Value, (nuint)bytes.Value.Length, optionsJson),
                _ => throw new ArgumentException("ImageSource must contain a URI or image bytes.", nameof(source)),
            };
            if (textPtr == IntPtr.Zero) throw new RustOException("Text detection failed while producing spatial text.");
            try { return Marshal.PtrToStringUTF8(textPtr) ?? string.Empty; }
            finally { rocr_free_string(textPtr); }
        }

        private static string SerializeStructuredOptions(OcrRunOptions options)
        {
            ValidateOptions(options, allowSpatial: false);
            return SerializeOptions(options, options.Output);
        }

        private static string SerializeSpatialOptions(OcrRunOptions options)
        {
            options ??= new OcrRunOptions();
            ValidateOptions(options, allowSpatial: true);
            return SerializeOptions(options, OutputGranularity.Spatial);
        }

        private static string SerializeOptions(OcrRunOptions options, OutputGranularity output)
        {
            var serialized = new OcrRunOptions
            {
                Output = output,
                LineYThreshold = options.LineYThreshold,
                WordXThreshold = options.WordXThreshold,
                TextScore = options.TextScore,
                Classification = options.Classification,
                Orientation = options.Orientation,
                Preprocessing = options.Preprocessing,
            };
            return JsonSerializer.Serialize(serialized, new JsonSerializerOptions
            {
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            });
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
            ValidatePreprocessing(options.Preprocessing);
        }

        private static void ValidatePreprocessing(PreprocessingRunOptions options)
        {
            if (options == null) return;
            ValidatePositive(options.MinHeight, nameof(options.MinHeight));
            ValidatePositive(options.MaxSideLen, nameof(options.MaxSideLen));
            ValidatePositive(options.MinSideLen, nameof(options.MinSideLen));
            if (options.WidthHeightRatio.HasValue && (!float.IsFinite(options.WidthHeightRatio.Value) || (options.WidthHeightRatio.Value <= 0 && options.WidthHeightRatio.Value != -1))) throw new RustOException("WidthHeightRatio must be positive or -1.");
            if (options.MinSideLen.HasValue && options.MaxSideLen.HasValue && options.MinSideLen.Value > options.MaxSideLen.Value) throw new RustOException("MinSideLen must not exceed MaxSideLen.");
            var detection = options.Detection;
            if (detection == null) return;
            if (detection.LimitSideLen.HasValue && (detection.LimitSideLen.Value < 1 || detection.LimitSideLen.Value > 32767)) throw new RustOException("LimitSideLen must be between 1 and 32767.");
            if (detection.LimitType != null && detection.LimitType != "min" && detection.LimitType != "max") throw new RustOException("LimitType must be min or max.");
            ValidateVector(detection.Mean, false, "Mean");
            ValidateVector(detection.Std, true, "Std");
            var postprocess = detection.Postprocess;
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

        private List<TextResult> MarshalResults(IntPtr resultsPtr, int count)
        {
            var results = new List<TextResult>(count);
            int structSize = Marshal.SizeOf<CTextResult>();

            for (int i = 0; i < count; i++)
            {
                IntPtr itemPtr = IntPtr.Add(resultsPtr, i * structSize);
                var nativeResult = Marshal.PtrToStructure<CTextResult>(itemPtr);
                results.Add(TextResult.FromNative(nativeResult));
            }

            return results;
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
                if (_handle != IntPtr.Zero)
                {
                    rocr_free(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        ~RustO()
        {
            Dispose();
        }
    }
}
