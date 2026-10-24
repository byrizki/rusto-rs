import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'src/ffi.dart';

export 'src/ffi.dart' show RustONativeBindings;

class Frame {
  final double width;
  final double height;
  final double top;
  final double left;

  const Frame({
    required this.width,
    required this.height,
    required this.top,
    required this.left,
  });

  @override
  String toString() => 'Frame(width: $width, height: $height, top: $top, left: $left)';
}

class TextResult {
  final String text;
  final double score;
  final List<List<double>> boxPoints;
  final Frame frame;

  const TextResult({
    required this.text,
    required this.score,
    required this.boxPoints,
    required this.frame,
  });

  @override
  String toString() => 'TextResult(text: "$text", score: $score, frame: $frame)';
}

class RustO {
  final RustONativeBindings _bindings;
  ffi.Pointer<ffi.Void> _handle;
  bool _isDisposed = false;

  RustO._(this._bindings, this._handle);

  static RustO initialize(String configJson, {RustONativeBindings? bindings}) {
    final native = bindings ?? RustONativeBindings(RustONativeBindings.openDefault());
    final cConfig = configJson.toNativeUtf8();
    try {
      final handle = native.initialize(cConfig);
      if (handle == ffi.nullptr) {
        throw StateError('Failed to initialize RustO OCR engine');
      }
      return RustO._(native, handle);
    } finally {
      calloc.free(cConfig);
    }
  }

  void _ensureNotDisposed() {
    if (_isDisposed || _handle == ffi.nullptr) {
      throw StateError('RustO instance is already disposed');
    }
  }

  List<TextResult> _convertResults(ffi.Pointer<CTextResult> resultsPtr, int count) {
    if (resultsPtr == ffi.nullptr || count <= 0) {
      return const [];
    }
    final list = <TextResult>[];
    for (var i = 0; i < count; i++) {
      final item = (resultsPtr + i).ref;
      final text = item.text != ffi.nullptr ? item.text.toDartString() : '';
      list.add(
        TextResult(
          text: text,
          score: item.score,
          boxPoints: [
            [item.boxX1, item.boxY1],
            [item.boxX2, item.boxY2],
            [item.boxX3, item.boxY3],
            [item.boxX4, item.boxY4],
          ],
          frame: Frame(
            width: item.frameWidth,
            height: item.frameHeight,
            top: item.frameTop,
            left: item.frameLeft,
          ),
        ),
      );
    }
    return list;
  }

  List<TextResult> detectTextFile(String imagePath, [String optionsJson = '{}']) {
    _ensureNotDisposed();
    final cPath = imagePath.toNativeUtf8();
    final cOpts = optionsJson.toNativeUtf8();
    final resultsPtrPtr = calloc<ffi.Pointer<CTextResult>>();
    final countPtr = calloc<ffi.Size>();

    try {
      final status = _bindings.detectTextFile(
        _handle,
        cPath,
        cOpts,
        resultsPtrPtr,
        countPtr,
      );
      if (status != 0) {
        throw StateError('Text detection failed with status: $status');
      }
      final results = resultsPtrPtr.value;
      final count = countPtr.value;
      try {
        return _convertResults(results, count);
      } finally {
        if (results != ffi.nullptr) {
          _bindings.freeResults(results.cast(), count);
        }
      }
    } finally {
      calloc.free(cPath);
      calloc.free(cOpts);
      calloc.free(resultsPtrPtr);
      calloc.free(countPtr);
    }
  }

  List<TextResult> detectTextData(Uint8List bytes, [String optionsJson = '{}']) {
    _ensureNotDisposed();
    if (bytes.isEmpty) return const [];

    final cBytes = calloc<ffi.Uint8>(bytes.length);
    cBytes.asTypedList(bytes.length).setAll(0, bytes);
    final cOpts = optionsJson.toNativeUtf8();
    final resultsPtrPtr = calloc<ffi.Pointer<CTextResult>>();
    final countPtr = calloc<ffi.Size>();

    try {
      final status = _bindings.detectTextData(
        _handle,
        cBytes,
        bytes.length,
        cOpts,
        resultsPtrPtr,
        countPtr,
      );
      if (status != 0) {
        throw StateError('Text detection failed with status: $status');
      }
      final results = resultsPtrPtr.value;
      final count = countPtr.value;
      try {
        return _convertResults(results, count);
      } finally {
        if (results != ffi.nullptr) {
          _bindings.freeResults(results.cast(), count);
        }
      }
    } finally {
      calloc.free(cBytes);
      calloc.free(cOpts);
      calloc.free(resultsPtrPtr);
      calloc.free(countPtr);
    }
  }

  String detectTextFileSpatial(String imagePath, [String optionsJson = '{"output":"spatial"}']) {
    _ensureNotDisposed();
    final cPath = imagePath.toNativeUtf8();
    final cOpts = optionsJson.toNativeUtf8();

    try {
      final strPtr = _bindings.detectTextFileSpatial(_handle, cPath, cOpts);
      if (strPtr == ffi.nullptr) {
        throw StateError('Spatial text detection failed');
      }
      try {
        return strPtr.toDartString();
      } finally {
        _bindings.freeString(strPtr);
      }
    } finally {
      calloc.free(cPath);
      calloc.free(cOpts);
    }
  }

  String detectTextDataSpatial(Uint8List bytes, [String optionsJson = '{"output":"spatial"}']) {
    _ensureNotDisposed();
    if (bytes.isEmpty) return '';

    final cBytes = calloc<ffi.Uint8>(bytes.length);
    cBytes.asTypedList(bytes.length).setAll(0, bytes);
    final cOpts = optionsJson.toNativeUtf8();

    try {
      final strPtr = _bindings.detectTextDataSpatial(_handle, cBytes, bytes.length, cOpts);
      if (strPtr == ffi.nullptr) {
        throw StateError('Spatial text detection failed');
      }
      try {
        return strPtr.toDartString();
      } finally {
        _bindings.freeString(strPtr);
      }
    } finally {
      calloc.free(cBytes);
      calloc.free(cOpts);
    }
  }

  void dispose() {
    if (!_isDisposed && _handle != ffi.nullptr) {
      _bindings.free(_handle);
      _handle = ffi.nullptr;
      _isDisposed = true;
    }
  }
}
