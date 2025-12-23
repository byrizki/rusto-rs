using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace RustODotnet
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct CTextResult
    {
        public IntPtr Text;
        public float Score;
        public float BoxX1, BoxY1, BoxX2, BoxY2;
        public float BoxX3, BoxY3, BoxX4, BoxY4;
    }

    public struct Point2D
    {
        public float X { get; set; }
        public float Y { get; set; }
        
        public Point2D(float x, float y) { X = x; Y = y; }
    }

    public class TextResult
    {
        public string Text { get; set; }
        public float Score { get; set; }
        public Point2D[] BoxPoints { get; set; }

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
                }
            };
        }
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

        public RustO(string detModelPath = null, string recModelPath = null, string dictPath = null)
        {
            // Use default bundled models if not specified
            detModelPath ??= "det.mnn";
            recModelPath ??= "rec.mnn";
            dictPath ??= "dict.txt";

            // Resolve paths - look in models subdirectory of app directory first
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
            // If absolute path exists, use it
            if (Path.IsPathRooted(path) && File.Exists(path))
            {
                return path;
            }

            // Try models subdirectory in application directory
            var appDir = AppContext.BaseDirectory;
            var modelsPath = Path.Combine(appDir, "models", path);
            if (File.Exists(modelsPath))
            {
                return modelsPath;
            }

            // Try directly in application directory
            var directPath = Path.Combine(appDir, path);
            if (File.Exists(directPath))
            {
                return directPath;
            }

            // Try current directory
            if (File.Exists(path))
            {
                return Path.GetFullPath(path);
            }

            // Return original path and let native code handle the error
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
            float yThresholdMultiplier = 0.6f,
            float xThresholdMultiplier = 1.3f)
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
            float yThresholdMultiplier = 0.6f,
            float xThresholdMultiplier = 1.3f)
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
