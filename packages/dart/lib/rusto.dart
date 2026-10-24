import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io' show Platform;
import 'dart:typed_data';

import 'src/generated/rusto.dart' as generated;

export 'src/generated/rusto.dart' hide RustO;

ffi.DynamicLibrary _openDefaultDynamicLibrary() {
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
      final envPath = Platform.environment['RUSTO_LIB_PATH'];
      final candidates = [
        if (envPath != null && envPath.isNotEmpty) envPath,
        'librusto.dylib',
        'target/release/librusto.dylib',
        'target/debug/librusto.dylib',
        '../target/release/librusto.dylib',
        '../target/debug/librusto.dylib',
        '../../target/release/librusto.dylib',
        '../../target/debug/librusto.dylib',
        '../../../target/release/librusto.dylib',
        '../../../target/debug/librusto.dylib',
      ];
      for (final p in candidates) {
        try {
          return ffi.DynamicLibrary.open(p);
        } catch (_) {}
      }
      return ffi.DynamicLibrary.open('librusto.dylib');
    }
  } else if (Platform.isWindows) {
    final envPath = Platform.environment['RUSTO_LIB_PATH'];
    final candidates = [
      if (envPath != null && envPath.isNotEmpty) envPath,
      'rusto.dll',
      'target/release/rusto.dll',
      'target/debug/rusto.dll',
      '../target/release/rusto.dll',
      '../target/debug/rusto.dll',
      '../../target/release/rusto.dll',
      '../../target/debug/rusto.dll',
      '../../../target/release/rusto.dll',
      '../../../target/debug/rusto.dll',
    ];
    for (final p in candidates) {
      try {
        return ffi.DynamicLibrary.open(p);
      } catch (_) {}
    }
    return ffi.DynamicLibrary.open('rusto.dll');
  }
  return ffi.DynamicLibrary.process();
}

bool _bindingsConfigured = false;
void _initBindingsIfNeeded() {
  if (!_bindingsConfigured) {
    generated.configureDefaultBindings(dynamicLibrary: _openDefaultDynamicLibrary());
    _bindingsConfigured = true;
  }
}

/// High-level RustO OCR Engine for Dart and Flutter.
class RustO {
  final generated.RustO _inner;

  RustO._(this._inner);

  static void configureBindings({ffi.DynamicLibrary? dynamicLibrary, String? libraryPath}) {
    generated.configureDefaultBindings(
      dynamicLibrary: dynamicLibrary,
      libraryPath: libraryPath,
    );
    _bindingsConfigured = true;
  }

  /// Initialize engine with either JSON config string or structured InitializeConfig.
  static RustO initialize(dynamic config, {Object? bindings}) {
    _initBindingsIfNeeded();
    if (config is generated.InitializeConfig) {
      return RustO._(generated.RustO.initialize(config));
    } else if (config is String) {
      try {
        return RustO._(generated.RustO.initializeFromJson(config));
      } catch (e) {
        throw StateError('Failed to initialize RustO: $e');
      }
    } else {
      throw ArgumentError('Expected String (JSON) or InitializeConfig, got ${config.runtimeType}');
    }
  }

  /// Initialize engine from JSON configuration string.
  static RustO initializeFromJson(String configJson) {
    _initBindingsIfNeeded();
    return RustO._(generated.RustO.initializeFromJson(configJson));
  }

  /// Detect text from file path with structured results.
  List<generated.TextResult> detectTextFile(String imagePath, [dynamic options]) {
    final ocrOpts = _resolveOptions(options, spatial: false);
    final res = _inner.detectText(generated.ImageSourcePath(path: imagePath), ocrOpts);
    if (res is generated.DetectTextResultStructured) {
      return res.results;
    }
    return const [];
  }

  /// Detect text from file path formatted as spatial string.
  String detectTextFileSpatial(String imagePath, [dynamic options]) {
    final ocrOpts = _resolveOptions(options, spatial: true);
    final res = _inner.detectText(generated.ImageSourcePath(path: imagePath), ocrOpts);
    if (res is generated.DetectTextResultSpatial) {
      return res.text;
    }
    return '';
  }

  /// Detect text from memory bytes with structured results.
  List<generated.TextResult> detectTextData(Uint8List bytes, [dynamic options]) {
    final ocrOpts = _resolveOptions(options, spatial: false);
    final res = _inner.detectText(generated.ImageSourceBytes(data: bytes), ocrOpts);
    if (res is generated.DetectTextResultStructured) {
      return res.results;
    }
    return const [];
  }

  /// Detect text from memory bytes formatted as spatial string.
  String detectTextDataSpatial(Uint8List bytes, [dynamic options]) {
    final ocrOpts = _resolveOptions(options, spatial: true);
    final res = _inner.detectText(generated.ImageSourceBytes(data: bytes), ocrOpts);
    if (res is generated.DetectTextResultSpatial) {
      return res.text;
    }
    return '';
  }

  /// Detect text from raw RGB pixel buffer with structured results.
  List<generated.TextResult> detectTextRgb(
    Uint8List pixels,
    int width,
    int height,
    int stride, [
    dynamic options,
  ]) {
    final ocrOpts = _resolveOptions(options, spatial: false);
    final res = _inner.detectTextRgb(pixels, width, height, stride, ocrOpts);
    if (res is generated.DetectTextResultStructured) {
      return res.results;
    }
    return const [];
  }

  /// Detect text from raw RGB pixel buffer formatted as spatial string.
  String detectTextRgbSpatial(
    Uint8List pixels,
    int width,
    int height,
    int stride, [
    dynamic options,
  ]) {
    final ocrOpts = _resolveOptions(options, spatial: true);
    final res = _inner.detectTextRgb(pixels, width, height, stride, ocrOpts);
    if (res is generated.DetectTextResultSpatial) {
      return res.text;
    }
    return '';
  }

  /// Unified UniFFI image detection method.
  generated.DetectTextResult detectText(generated.ImageSource source, generated.OcrRunOptions options) {
    return _inner.detectText(source, options);
  }

  /// Engine version string.
  String version() => _inner.version();

  /// Release native resources.
  void close() => _inner.close();

  /// Alias for close() for backward compatibility.
  void dispose() => close();

  static generated.OcrRunOptions _resolveOptions(dynamic options, {required bool spatial}) {
    generated.OcrRunOptions opts;
    if (options is generated.OcrRunOptions) {
      opts = options;
    } else if (options is String && options.isNotEmpty && options != '{}') {
      try {
        final decoded = json.decode(options) as Map<String, dynamic>;
        opts = generated.OcrRunOptions.fromJson(decoded);
      } catch (_) {
        opts = generated.defaultOcrRunOptions();
      }
    } else {
      opts = generated.defaultOcrRunOptions();
    }

    if (spatial && opts.output != generated.OutputGranularity.spatial) {
      return generated.OcrRunOptions(
        output: generated.OutputGranularity.spatial,
        lineYThreshold: opts.lineYThreshold,
        wordXThreshold: opts.wordXThreshold,
        textScore: opts.textScore,
        classification: opts.classification,
        orientation: opts.orientation,
        minHeight: opts.minHeight,
        maxSideLen: opts.maxSideLen,
        minSideLen: opts.minSideLen,
        widthHeightRatio: opts.widthHeightRatio,
        detection: opts.detection,
        postprocess: opts.postprocess,
        calibration: opts.calibration,
        optimization: opts.optimization,
      );
    }
    return opts;
  }
}
