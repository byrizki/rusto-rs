import 'dart:ffi' as ffi;
import 'dart:io' show Platform;
import 'package:ffi/ffi.dart';

final class CTextResult extends ffi.Struct {
  external ffi.Pointer<Utf8> text;

  @ffi.Float()
  external double score;

  @ffi.Float()
  external double boxX1;
  @ffi.Float()
  external double boxY1;

  @ffi.Float()
  external double boxX2;
  @ffi.Float()
  external double boxY2;

  @ffi.Float()
  external double boxX3;
  @ffi.Float()
  external double boxY3;

  @ffi.Float()
  external double boxX4;
  @ffi.Float()
  external double boxY4;

  @ffi.Float()
  external double frameWidth;
  @ffi.Float()
  external double frameHeight;
  @ffi.Float()
  external double frameTop;
  @ffi.Float()
  external double frameLeft;
}

typedef RocrInitializeNative = ffi.Pointer<ffi.Void> Function(ffi.Pointer<Utf8> configJson);
typedef RocrInitializeDart = ffi.Pointer<ffi.Void> Function(ffi.Pointer<Utf8> configJson);

typedef RocrDetectTextFileNative = ffi.Int32 Function(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<Utf8> imagePath,
  ffi.Pointer<Utf8> optionsJson,
  ffi.Pointer<ffi.Pointer<CTextResult>> results,
  ffi.Pointer<ffi.Size> count,
);
typedef RocrDetectTextFileDart = int Function(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<Utf8> imagePath,
  ffi.Pointer<Utf8> optionsJson,
  ffi.Pointer<ffi.Pointer<CTextResult>> results,
  ffi.Pointer<ffi.Size> count,
);

typedef RocrDetectTextFileSpatialNative = ffi.Pointer<Utf8> Function(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<Utf8> imagePath,
  ffi.Pointer<Utf8> optionsJson,
);
typedef RocrDetectTextFileSpatialDart = ffi.Pointer<Utf8> Function(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<Utf8> imagePath,
  ffi.Pointer<Utf8> optionsJson,
);

typedef RocrDetectTextDataNative = ffi.Int32 Function(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Uint8> data,
  ffi.Size length,
  ffi.Pointer<Utf8> optionsJson,
  ffi.Pointer<ffi.Pointer<CTextResult>> results,
  ffi.Pointer<ffi.Size> count,
);
typedef RocrDetectTextDataDart = int Function(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Uint8> data,
  int length,
  ffi.Pointer<Utf8> optionsJson,
  ffi.Pointer<ffi.Pointer<CTextResult>> results,
  ffi.Pointer<ffi.Size> count,
);

typedef RocrDetectTextDataSpatialNative = ffi.Pointer<Utf8> Function(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Uint8> data,
  ffi.Size length,
  ffi.Pointer<Utf8> optionsJson,
);
typedef RocrDetectTextDataSpatialDart = ffi.Pointer<Utf8> Function(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Uint8> data,
  int length,
  ffi.Pointer<Utf8> optionsJson,
);

typedef RocrDetectTextRgbNative = ffi.Int32 Function(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Uint8> pixels,
  ffi.Size imageLen,
  ffi.Uint32 width,
  ffi.Uint32 height,
  ffi.Size stride,
  ffi.Pointer<Utf8> optionsJson,
  ffi.Pointer<ffi.Pointer<CTextResult>> results,
  ffi.Pointer<ffi.Size> count,
);
typedef RocrDetectTextRgbDart = int Function(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Uint8> pixels,
  int imageLen,
  int width,
  int height,
  int stride,
  ffi.Pointer<Utf8> optionsJson,
  ffi.Pointer<ffi.Pointer<CTextResult>> results,
  ffi.Pointer<ffi.Size> count,
);

typedef RocrDetectTextRgbSpatialNative = ffi.Pointer<Utf8> Function(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Uint8> pixels,
  ffi.Size imageLen,
  ffi.Uint32 width,
  ffi.Uint32 height,
  ffi.Size stride,
  ffi.Pointer<Utf8> optionsJson,
);
typedef RocrDetectTextRgbSpatialDart = ffi.Pointer<Utf8> Function(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Uint8> pixels,
  int imageLen,
  int width,
  int height,
  int stride,
  ffi.Pointer<Utf8> optionsJson,
);

typedef RocrFreeStringNative = ffi.Void Function(ffi.Pointer<Utf8> s);
typedef RocrFreeStringDart = void Function(ffi.Pointer<Utf8> s);

typedef RocrFreeResultsNative = ffi.Void Function(ffi.Pointer<ffi.Void> results, ffi.Size count);
typedef RocrFreeResultsDart = void Function(ffi.Pointer<ffi.Void> results, int count);

typedef RocrFreeNative = ffi.Void Function(ffi.Pointer<ffi.Void> handle);
typedef RocrFreeDart = void Function(ffi.Pointer<ffi.Void> handle);

class RustONativeBindings {
  final RocrInitializeDart initialize;
  final RocrDetectTextFileDart detectTextFile;
  final RocrDetectTextFileSpatialDart detectTextFileSpatial;
  final RocrDetectTextDataDart detectTextData;
  final RocrDetectTextDataSpatialDart detectTextDataSpatial;
  final RocrDetectTextRgbDart detectTextRgb;
  final RocrDetectTextRgbSpatialDart detectTextRgbSpatial;
  final RocrFreeStringDart freeString;
  final RocrFreeResultsDart freeResults;
  final RocrFreeDart free;

  RustONativeBindings(ffi.DynamicLibrary dylib)
      : initialize = dylib
            .lookup<ffi.NativeFunction<RocrInitializeNative>>('rocr_initialize')
            .asFunction(),
        detectTextFile = dylib
            .lookup<ffi.NativeFunction<RocrDetectTextFileNative>>('rocr_detect_text_file')
            .asFunction(),
        detectTextFileSpatial = dylib
            .lookup<ffi.NativeFunction<RocrDetectTextFileSpatialNative>>('rocr_detect_text_file_spatial')
            .asFunction(),
        detectTextData = dylib
            .lookup<ffi.NativeFunction<RocrDetectTextDataNative>>('rocr_detect_text_data')
            .asFunction(),
        detectTextDataSpatial = dylib
            .lookup<ffi.NativeFunction<RocrDetectTextDataSpatialNative>>('rocr_detect_text_data_spatial')
            .asFunction(),
        detectTextRgb = dylib
            .lookup<ffi.NativeFunction<RocrDetectTextRgbNative>>('rocr_detect_text_rgb')
            .asFunction(),
        detectTextRgbSpatial = dylib
            .lookup<ffi.NativeFunction<RocrDetectTextRgbSpatialNative>>('rocr_detect_text_rgb_spatial')
            .asFunction(),
        freeString = dylib
            .lookup<ffi.NativeFunction<RocrFreeStringNative>>('rocr_free_string')
            .asFunction(),
        freeResults = dylib
            .lookup<ffi.NativeFunction<RocrFreeResultsNative>>('rocr_free_results')
            .asFunction(),
        free = dylib
            .lookup<ffi.NativeFunction<RocrFreeNative>>('rocr_free')
            .asFunction();

  static ffi.DynamicLibrary openDefault() {
    if (Platform.isAndroid) {
      return ffi.DynamicLibrary.open('librusto.so');
    } else if (Platform.isLinux) {
      final envPath = Platform.environment['RUSTO_LIB_PATH'];
      final candidates = [
        if (envPath != null && envPath.isNotEmpty) envPath,
        'librusto.so',
        'target/release/librusto.so',
        'target/debug/librusto.so',
        '../target/release/librusto.so',
        '../target/debug/librusto.so',
        '../../target/release/librusto.so',
        '../../target/debug/librusto.so',
        '../../../target/release/librusto.so',
        '../../../target/debug/librusto.so',
      ];
      for (final p in candidates) {
        try {
          return ffi.DynamicLibrary.open(p);
        } catch (_) {}
      }
      return ffi.DynamicLibrary.open('librusto.so');
    } else if (Platform.isMacOS || Platform.isIOS) {
      try {
        return ffi.DynamicLibrary.process();
      } catch (_) {
        return ffi.DynamicLibrary.open('librusto.dylib');
      }
    } else if (Platform.isWindows) {
      return ffi.DynamicLibrary.open('rusto.dll');
    }
    return ffi.DynamicLibrary.process();
  }
}
