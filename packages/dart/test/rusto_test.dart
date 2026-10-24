import 'dart:ffi' as ffi;
import 'package:test/test.dart';
import 'package:rusto/rusto.dart';

void main() {
  group('RustO Dart Binding Tests', () {
    test('Invalid JSON initialization throws StateError', () {
      final dylib = ffi.DynamicLibrary.open('../../target/debug/librusto.so');
      final bindings = RustONativeBindings(dylib);

      expect(
        () => RustO.initialize('invalid-json-config', bindings: bindings),
        throwsA(isA<StateError>()),
      );
    });

    test('Frame and TextResult structure validation', () {
      const frame = Frame(width: 120, height: 24, top: 10, left: 20);
      expect(frame.width, equals(120));
      expect(frame.height, equals(24));
      expect(frame.top, equals(10));
      expect(frame.left, equals(20));

      final result = TextResult(
        text: 'PaddleOCR RustO',
        score: 0.99,
        boxPoints: [
          [20, 10],
          [140, 10],
          [140, 34],
          [20, 34],
        ],
        frame: frame,
      );

      expect(result.text, equals('PaddleOCR RustO'));
      expect(result.score, equals(0.99));
      expect(result.frame, equals(frame));
    });
  });
}
