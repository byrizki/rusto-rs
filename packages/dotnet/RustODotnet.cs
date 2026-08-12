using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Collections.Generic;
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

    public class RustOConfig
    {
        [JsonPropertyName("template")]
        public string Template { get; set; } = "ppv6";

        [JsonPropertyName("detModelPath")]
        public string DetModelPath { get; set; } = "det.mnn";

        [JsonPropertyName("recModelPath")]
        public string RecModelPath { get; set; } = "rec.mnn";

        [JsonPropertyName("dictPath")]
        public string DictPath { get; set; } = "dict.txt";

        [JsonPropertyName("clsModelPath")]
        public string ClsModelPath { get; set; }

        [JsonPropertyName("orientModelPath")]
        public string OrientModelPath { get; set; }

        [JsonPropertyName("unwarpModelPath")]
        public string UnwarpModelPath { get; set; }

        [JsonPropertyName("orientThreshold")]
        public float? OrientThreshold { get; set; }

        [JsonPropertyName("clsThreshold")]
        public float? ClsThreshold { get; set; }

        [JsonPropertyName("textScore")]
        public float TextScore { get; set; } = 0.5f;

        [JsonPropertyName("detThresh")]
        public float DetThresh { get; set; } = 0.3f;

        [JsonPropertyName("detBoxThresh")]
        public float DetBoxThresh { get; set; } = 0.6f;

        [JsonPropertyName("limitSideLen")]
        public int LimitSideLen { get; set; } = 736;

        [JsonPropertyName("limitType")]
        public string LimitType { get; set; } = "min";

        [JsonPropertyName("unclipRatio")]
        public float UnclipRatio { get; set; } = 2.0f;

        [JsonPropertyName("useDilation")]
        public bool UseDilation { get; set; } = true;

        [JsonPropertyName("useDet")]
        public bool UseDet { get; set; } = true;

        [JsonPropertyName("useRec")]
        public bool UseRec { get; set; } = true;

        [JsonPropertyName("useCls")]
        public bool UseCls { get; set; } = false;

        [JsonPropertyName("useOrient")]
        public bool UseOrient { get; set; } = false;

        [JsonPropertyName("useUnwarp")]
        public bool UseUnwarp { get; set; } = false;

        [JsonPropertyName("debugImages")]
        public bool DebugImages { get; set; } = false;

        [JsonPropertyName("minHeight")]
        public float MinHeight { get; set; } = 30.0f;

        [JsonPropertyName("maxSideLen")]
        public float MaxSideLen { get; set; } = 2000.0f;

        [JsonPropertyName("minSideLen")]
        public float MinSideLen { get; set; } = 30.0f;

        [JsonPropertyName("returnWordBox")]
        public bool ReturnWordBox { get; set; } = false;

        [JsonPropertyName("returnSingleCharBox")]
        public bool ReturnSingleCharBox { get; set; } = false;

        [JsonPropertyName("yThresholdMultiplier")]
        public float? YThresholdMultiplier { get; set; }

        [JsonPropertyName("xThresholdMultiplier")]
        public float? XThresholdMultiplier { get; set; }

        public RustOConfig() { }

        public RustOConfig(string detModelPath, string recModelPath, string dictPath, string template = "ppv6")
        {
            Template = template;
            DetModelPath = detModelPath;
            RecModelPath = recModelPath;
            DictPath = dictPath;
        }

        public static RustOConfig Ppv6(string det = "det.mnn", string rec = "rec.mnn", string dict = "dict.txt") =>
            new RustOConfig(det, rec, dict, "ppv6") { DetBoxThresh = 0.6f };

        public static RustOConfig Ppv5(string det = "det.mnn", string rec = "rec.mnn", string dict = "dict.txt") =>
            new RustOConfig(det, rec, dict, "ppv5");

        public static RustOConfig Ppv4(string det = "det.mnn", string rec = "rec.mnn", string dict = "dict.txt") =>
            new RustOConfig(det, rec, dict, "ppv4") { LimitSideLen = 960, LimitType = "max", UnclipRatio = 1.5f, UseDilation = false, DetBoxThresh = 0.6f };

        public static RustOConfig Ppv3(string det = "det.mnn", string rec = "rec.mnn", string dict = "dict.txt") =>
            new RustOConfig(det, rec, dict, "ppv3") { LimitSideLen = 960, LimitType = "max", UnclipRatio = 1.5f, UseDilation = false, DetBoxThresh = 0.6f };
    }

    public class RustOException : Exception
    {
        public RustOException(string message) : base(message) { }
    }

    public sealed class RustO : IDisposable
    {
        private const string LibName = "rusto";

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rocr_new(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string detModelPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string recModelPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string dictPath
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rocr_new_with_config(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string configJson
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int rocr_ocr_file(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string imagePath,
            out IntPtr results,
            out nuint count
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int rocr_ocr_file_with_output(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string imagePath,
            out IntPtr output
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int rocr_ocr_data(
            IntPtr handle,
            byte[] imageData,
            nuint imageLen,
            out IntPtr results,
            out nuint count
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int rocr_ocr_data_with_output(
            IntPtr handle,
            byte[] imageData,
            nuint imageLen,
            out IntPtr output
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rocr_output_to_raw(IntPtr output);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rocr_output_to_csv(IntPtr output);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rocr_output_to_text_with_position(IntPtr output);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rocr_output_to_spatial_text(
            IntPtr output,
            float yThresholdMultiplier,
            float xThresholdMultiplier
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int rocr_output_get_results(
            IntPtr output,
            out IntPtr results,
            out nuint count
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void rocr_free_output(IntPtr output);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void rocr_free_string(IntPtr str);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void rocr_free_results(IntPtr results, nuint count);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void rocr_free(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rocr_version();

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

        public RustO(RustOConfig config)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));

            config.DetModelPath = ResolveModelPath(config.DetModelPath ?? "det.mnn");
            config.RecModelPath = ResolveModelPath(config.RecModelPath ?? "rec.mnn");
            config.DictPath = ResolveModelPath(config.DictPath ?? "dict.txt");

            if (config.ClsModelPath != null) config.ClsModelPath = ResolveModelPath(config.ClsModelPath);
            if (config.OrientModelPath != null) config.OrientModelPath = ResolveModelPath(config.OrientModelPath);
            if (config.UnwarpModelPath != null) config.UnwarpModelPath = ResolveModelPath(config.UnwarpModelPath);

            string json = JsonSerializer.Serialize(config);
            _handle = rocr_new_with_config(json);
            if (_handle == IntPtr.Zero)
            {
                throw new RustOException("Failed to initialize RustO with config");
            }
        }

        public RustO(string detModelPath = null, string recModelPath = null, string dictPath = null)
        {
            detModelPath ??= "det.mnn";
            recModelPath ??= "rec.mnn";
            dictPath ??= "dict.txt";

            detModelPath = ResolveModelPath(detModelPath);
            recModelPath = ResolveModelPath(recModelPath);
            dictPath = ResolveModelPath(dictPath);

            _handle = rocr_new(detModelPath, recModelPath, dictPath);
            if (_handle == IntPtr.Zero)
            {
                throw new RustOException($"Failed to initialize RustO with models: {detModelPath}, {recModelPath}, {dictPath}");
            }
        }

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

        public List<TextResult> RecognizeFile(string imagePath)
        {
            ThrowIfDisposed();

            int status = rocr_ocr_file(_handle, imagePath, out IntPtr resultsPtr, out nuint count);
            if (status != 0)
            {
                throw new RustOException($"OCR failed with status code: {status}");
            }

            try
            {
                return MarshalResults(resultsPtr, (int)count);
            }
            finally
            {
                rocr_free_results(resultsPtr, count);
            }
        }

        public List<TextResult> Recognize(byte[] imageData)
        {
            ThrowIfDisposed();

            int status = rocr_ocr_data(_handle, imageData, (nuint)imageData.Length, 
                out IntPtr resultsPtr, out nuint count);
            if (status != 0)
            {
                throw new RustOException($"OCR failed with status code: {status}");
            }

            try
            {
                return MarshalResults(resultsPtr, (int)count);
            }
            finally
            {
                rocr_free_results(resultsPtr, count);
            }
        }

        public string RecognizeFileToRaw(string imagePath)
        {
            ThrowIfDisposed();
            return RecognizeFileWithFormat(imagePath, rocr_output_to_raw);
        }

        public string RecognizeFileToCsv(string imagePath)
        {
            ThrowIfDisposed();
            return RecognizeFileWithFormat(imagePath, rocr_output_to_csv);
        }

        public string RecognizeFileToTextWithPosition(string imagePath)
        {
            ThrowIfDisposed();
            return RecognizeFileWithFormat(imagePath, rocr_output_to_text_with_position);
        }

        public string RecognizeFileToSpatialText(
            string imagePath,
            float yThresholdMultiplier = 0.0f,
            float xThresholdMultiplier = 0.0f)
        {
            ThrowIfDisposed();
            return RecognizeFileWithFormat(imagePath, output => 
                rocr_output_to_spatial_text(output, yThresholdMultiplier, xThresholdMultiplier));
        }

        public string RecognizeToRaw(byte[] imageData)
        {
            ThrowIfDisposed();
            return RecognizeWithFormat(imageData, rocr_output_to_raw);
        }

        public string RecognizeToCsv(byte[] imageData)
        {
            ThrowIfDisposed();
            return RecognizeWithFormat(imageData, rocr_output_to_csv);
        }

        public string RecognizeToTextWithPosition(byte[] imageData)
        {
            ThrowIfDisposed();
            return RecognizeWithFormat(imageData, rocr_output_to_text_with_position);
        }

        public string RecognizeToSpatialText(
            byte[] imageData,
            float yThresholdMultiplier = 0.0f,
            float xThresholdMultiplier = 0.0f)
        {
            ThrowIfDisposed();
            return RecognizeWithFormat(imageData, output => 
                rocr_output_to_spatial_text(output, yThresholdMultiplier, xThresholdMultiplier));
        }

        private string RecognizeFileWithFormat(string imagePath, Func<IntPtr, IntPtr> formatFunc)
        {
            int status = rocr_ocr_file_with_output(_handle, imagePath, out IntPtr outputPtr);
            if (status != 0)
            {
                throw new RustOException($"OCR failed with status code: {status}");
            }

            try
            {
                IntPtr strPtr = formatFunc(outputPtr);
                try
                {
                    return Marshal.PtrToStringUTF8(strPtr) ?? string.Empty;
                }
                finally
                {
                    rocr_free_string(strPtr);
                }
            }
            finally
            {
                rocr_free_output(outputPtr);
            }
        }

        private string RecognizeWithFormat(byte[] imageData, Func<IntPtr, IntPtr> formatFunc)
        {
            int status = rocr_ocr_data_with_output(_handle, imageData, (nuint)imageData.Length, out IntPtr outputPtr);
            if (status != 0)
            {
                throw new RustOException($"OCR failed with status code: {status}");
            }

            try
            {
                IntPtr strPtr = formatFunc(outputPtr);
                try
                {
                    return Marshal.PtrToStringUTF8(strPtr) ?? string.Empty;
                }
                finally
                {
                    rocr_free_string(strPtr);
                }
            }
            finally
            {
                rocr_free_output(outputPtr);
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
