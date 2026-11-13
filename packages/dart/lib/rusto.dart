/// Pure Rust OCR library bindings for Dart and Flutter based on RapidOCR and PaddleOCR.
library rusto;

import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io' show Platform;
import 'dart:typed_data';

import 'src/generated/rusto.dart' as generated;

export 'src/generated/rusto.dart'
    hide
        RustO,
        RustoFfi,
        RustOFfiCodec,
        DetectTextResultFfiCodec,
        ImageSourceFfiCodec,
        OutputGranularityFfiCodec,
        RustoBindingErrorFfiCodec,
        RustoBindingErrorExceptionFfiCodec,
        RustoBindingErrorExceptionInitialization,
        RustoBindingErrorExceptionPreprocess,
        RustoBindingErrorExceptionInference,
        RustoBindingErrorExceptionPostprocess,
        RustoBindingErrorExceptionInvalidInput;

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
    generated.configureDefaultBindings(
        dynamicLibrary: _openDefaultDynamicLibrary());
    _bindingsConfigured = true;
  }
}

/// High-level RustO OCR Engine for Dart and Flutter.
class RustO {
  final generated.RustO _inner;

  RustO._(this._inner);

  /// Configure custom native library path or pre-opened DynamicLibrary.
  static void configureBindings(
      {ffi.DynamicLibrary? dynamicLibrary, String? libraryPath}) {
    generated.configureDefaultBindings(
      dynamicLibrary: dynamicLibrary,
      libraryPath: libraryPath,
    );
    _bindingsConfigured = true;
  }

  /// Initialize engine with optional JSON config string or structured InitializeConfig.
  /// Defaults to '{}' (PP-OCRv6 Tiny FP32 auto-download).
  static RustO initialize([dynamic config = '{}', Object? bindings]) {
    _initBindingsIfNeeded();
    if (config == null) {
      return RustO._(generated.RustO.initializeFromJson('{}'));
    } else if (config is generated.InitializeConfig) {
      return RustO._(generated.RustO.initialize(config));
    } else if (config is String) {
      try {
        return RustO._(generated.RustO.initializeFromJson(config));
      } catch (e) {
        throw StateError('Failed to initialize RustO: $e');
      }
    } else {
      throw ArgumentError(
          'Expected String (JSON) or InitializeConfig, got ${config.runtimeType}');
    }
  }

  /// Initialize engine from JSON configuration string.
  static RustO initializeFromJson([String configJson = '{}']) {
    _initBindingsIfNeeded();
    return RustO._(generated.RustO.initializeFromJson(configJson));
  }

  /// Downloads model assets for a specific OCR pipeline ahead of time.
  static List<String> downloadModel({
    String? model,
    String? lang,
    String? cacheDir,
  }) {
    _initBindingsIfNeeded();
    return generated.downloadModel(model, lang, cacheDir);
  }

  /// Returns the underlying engine library version string.
  static String version() {
    _initBindingsIfNeeded();
    return generated.rustoVersion();
  }

  /// Returns whether this engine instance is ready for inference.
  bool isReady() => true;

  /// Returns the underlying engine library version string.
  String getVersion() => version();


  /// Detect text from raw RGB pixel buffer.
  generated.DetectTextResult detectTextRgb(
    Uint8List pixels,
    int width,
    int height, [
    int? stride,
    dynamic options,
  ]) {
    final effectiveStride = stride ?? (width * 3);
    final ocrOpts = _resolveOptions(options);
    return _inner.detectTextRgb(pixels, width, height, effectiveStride, ocrOpts);
  }

  /// Unified UniFFI image detection method or fluent builder.
  dynamic detectText([dynamic source, generated.OcrRunOptions? options]) {
    if (source != null && options != null && source is generated.ImageSource) {
      return _inner.detectText(source, options);
    }
    if (source == null) {
      return DetectTextBuilder(this);
    }
    if (source is String) {
      final b = DetectTextBuilder(this).image(source);
      if (options != null) b.withOptions(options);
      return b;
    }
    if (source is Uint8List) {
      final b = DetectTextBuilder(this).data(source);
      if (options != null) b.withOptions(options);
      return b;
    }
    if (source is generated.ImageSource) {
      if (options != null) {
        return _inner.detectText(source, options);
      }
      return DetectTextBuilder(this, source);
    }
    throw ArgumentError(
        'Expected String, Uint8List, or ImageSource, got ${source.runtimeType}');
  }

  /// Creates a fluent DetectTextBuilder for text detection and recognition.
  DetectTextBuilder detectTextBuilder() => DetectTextBuilder(this);

  /// Release native resources.
  void close() => _inner.close();

  static generated.OcrRunOptions _resolveOptions(dynamic options) {
    if (options is generated.OcrRunOptions) {
      return options;
    }
    if (options is String && options.isNotEmpty && options != '{}') {
      try {
        final decoded = json.decode(options) as Map<String, dynamic>;
        return generated.OcrRunOptions.fromJson(decoded);
      } catch (_) {
        return generated.defaultOcrRunOptions();
      }
    }
    return generated.defaultOcrRunOptions();
  }
}

/// Fluent builder for chained text detection configuration and execution.
class DetectTextBuilder {
  final RustO _engine;
  generated.ImageSource? _source;
  Uint8List? _rgbPixels;
  int _rgbWidth = 0;
  int _rgbHeight = 0;
  int _rgbStride = 0;
  generated.OutputGranularity _output = generated.OutputGranularity.lines;
  bool? _autoUnwarp;
  bool? _layoutReading;
  String? _layoutModel;
  bool? _orientationDetection;
  bool? _lineClassification;
  generated.PreDetectionOptions? _preDetection;
  generated.DetectionRunOptions? _detection;
  generated.PostDetectionOptions? _postDetection;
  generated.PreRecognitionOptions? _preRecognition;
  generated.RecognitionRunOptions? _recognition;
  generated.PostRecognitionOptions? _postRecognition;

  DetectTextBuilder(this._engine, [this._source]);

  DetectTextBuilder image(String imagePath) {
    _source = generated.ImageSourcePath(path: imagePath);
    _rgbPixels = null;
    return this;
  }

  DetectTextBuilder data(Uint8List bytes) {
    _source = generated.ImageSourceBytes(data: bytes);
    _rgbPixels = null;
    return this;
  }

  DetectTextBuilder rgb(Uint8List pixels, int width, int height, [int? stride]) {
    _rgbPixels = pixels;
    _rgbWidth = width;
    _rgbHeight = height;
    _rgbStride = stride ?? (width * 3);
    _source = null;
    return this;
  }

  DetectTextBuilder source(generated.ImageSource source) {
    _source = source;
    _rgbPixels = null;
    return this;
  }

  DetectTextBuilder withOptions(generated.OcrRunOptions options) {
    _output = options.output;
    _autoUnwarp = options.autoUnwarp;
    _layoutReading = options.layoutReading;
    _layoutModel = options.layoutModel;
    _orientationDetection = options.orientationDetection;
    _lineClassification = options.lineClassification;
    _preDetection = options.preDetection;
    _detection = options.detection;
    _postDetection = options.postDetection;
    _preRecognition = options.preRecognition;
    _recognition = options.recognition;
    _postRecognition = options.postRecognition;
    return this;
  }

  DetectTextBuilder withAutoUnwarp([bool enabled = true]) {
    _autoUnwarp = enabled;
    return this;
  }

  DetectTextBuilder withLayoutReading([bool enabled = true]) {
    _layoutReading = enabled;
    return this;
  }

  DetectTextBuilder withLayoutModel(String model) {
    _layoutModel = model;
    return this;
  }

  DetectTextBuilder withOrientationDetection([bool enabled = true]) {
    _orientationDetection = enabled;
    return this;
  }

  DetectTextBuilder withLineClassification([bool enabled = true]) {
    _lineClassification = enabled;
    return this;
  }

  DetectTextBuilder withPreDetection({
    double? descreenStrength,
    int? targetMaxSide,
    double? cropPaddingX,
    double? cropPaddingY,
    generated.PreDetectionOptions? options,
  }) {
    _preDetection = options ??
        generated.PreDetectionOptions(
          descreenStrength: descreenStrength,
          targetMaxSide: targetMaxSide,
          cropPaddingX: cropPaddingX,
          cropPaddingY: cropPaddingY,
        );
    return this;
  }

  DetectTextBuilder withDetection({
    int? limitSideLen,
    String? limitType,
    Float32List? mean,
    Float32List? std,
    generated.DetectionRunOptions? options,
  }) {
    _detection = options ??
        generated.DetectionRunOptions(
          limitSideLen: limitSideLen,
          limitType: limitType,
          mean: mean,
          std: std,
        );
    return this;
  }

  DetectTextBuilder withPostDetection({
    double? threshold,
    double? boxThreshold,
    double? unclipRatio,
    bool? useDilation,
    String? scoreMode,
    int? maxCandidates,
    double? minHeight,
    double? minSideLen,
    double? maxSideLen,
    double? widthHeightRatio,
    generated.PostDetectionOptions? options,
  }) {
    _postDetection = options ??
        generated.PostDetectionOptions(
          threshold: threshold,
          boxThreshold: boxThreshold,
          unclipRatio: unclipRatio,
          useDilation: useDilation,
          scoreMode: scoreMode,
          maxCandidates: maxCandidates,
          minHeight: minHeight,
          minSideLen: minSideLen,
          maxSideLen: maxSideLen,
          widthHeightRatio: widthHeightRatio,
        );
    return this;
  }

  DetectTextBuilder withPreRecognition({
    double? lineYThreshold,
    double? wordXThreshold,
    generated.PreRecognitionOptions? options,
  }) {
    _preRecognition = options ??
        generated.PreRecognitionOptions(
          lineYThreshold: lineYThreshold,
          wordXThreshold: wordXThreshold,
        );
    return this;
  }

  /// Configure recognition inference parameters (`batchSize`, `workers`, `maxWorkers`).
  DetectTextBuilder withRecognition({
    int? batchSize,
    int? workers,
    int? maxWorkers,
    generated.RecognitionRunOptions? options,
  }) {
    _recognition = options ??
        generated.RecognitionRunOptions(
          batchSize: batchSize,
          workers: workers,
          maxWorkers: maxWorkers,
        );
    return this;
  }

  DetectTextBuilder withPostRecognition({
    double? scoreThreshold,
    generated.PostRecognitionOptions? options,
  }) {
    _postRecognition = options ??
        generated.PostRecognitionOptions(
          scoreThreshold: scoreThreshold,
        );
    return this;
  }

  DetectTextBuilder withOutput(generated.OutputGranularity granularity) {
    _output = granularity;
    return this;
  }

  DetectTextBuilder asLines() => withOutput(generated.OutputGranularity.lines);
  DetectTextBuilder asWords() => withOutput(generated.OutputGranularity.words);
  DetectTextBuilder asSpatial() => withOutput(generated.OutputGranularity.spatial);
  DetectTextBuilder asTsv() => withOutput(generated.OutputGranularity.tsv);

  generated.OcrRunOptions buildOptions() {
    return generated.OcrRunOptions(
      output: _output,
      autoUnwarp: _autoUnwarp,
      layoutReading: _layoutReading,
      layoutModel: _layoutModel,
      orientationDetection: _orientationDetection,
      lineClassification: _lineClassification,
      preDetection: _preDetection,
      detection: _detection,
      postDetection: _postDetection,
      preRecognition: _preRecognition,
      recognition: _recognition,
      postRecognition: _postRecognition,
    );
  }

  generated.DetectTextResult run() {
    final pixels = _rgbPixels;
    if (pixels != null) {
      return _engine.detectTextRgb(
        pixels,
        _rgbWidth,
        _rgbHeight,
        _rgbStride,
        buildOptions(),
      );
    }
    final src = _source;
    if (src == null) {
      throw StateError('Image source must be specified before calling run().');
    }
    return _engine.detectText(src, buildOptions()) as generated.DetectTextResult;
  }
}
