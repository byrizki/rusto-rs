using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace RustO
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
        private static extern IntPtr rusto_new(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string detModelPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string recModelPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string dictPath
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int rusto_ocr_file(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string imagePath,
            out IntPtr results,
            out nuint count
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int rusto_ocr_data(
            IntPtr handle,
            byte[] imageData,
            nuint imageLen,
            out IntPtr results,
            out nuint count
        );

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void rusto_free_results(IntPtr results, nuint count);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void rusto_free(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr rusto_version();

        private IntPtr _handle;
        private bool _disposed;

        public static string Version
        {
            get
            {
                IntPtr versionPtr = rusto_version();
                return Marshal.PtrToStringUTF8(versionPtr) ?? "unknown";
            }
        }

        public RustO(string detModelPath, string recModelPath, string dictPath)
        {
            _handle = rusto_new(detModelPath, recModelPath, dictPath);
            if (_handle == IntPtr.Zero)
            {
                throw new RustOException("Failed to initialize RustO");
            }
        }

        public List<TextResult> RecognizeFile(string imagePath)
        {
            ThrowIfDisposed();

            int status = rusto_ocr_file(_handle, imagePath, out IntPtr resultsPtr, out nuint count);
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
                rusto_free_results(resultsPtr, count);
            }
        }

        public List<TextResult> Recognize(byte[] imageData)
        {
            ThrowIfDisposed();

            int status = rusto_ocr_data(_handle, imageData, (nuint)imageData.Length, 
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
                rusto_free_results(resultsPtr, count);
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
                    rusto_free(_handle);
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
