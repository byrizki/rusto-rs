import 'dart:io';
import 'package:test/test.dart';
import 'package:rusto/rusto.dart';

void main() {
  group('RustO Dart Binding Tests', () {
    test('Invalid JSON initialization throws StateError', () {
      final bindings = RustONativeBindings(RustONativeBindings.openDefault());

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

    test('Real OCR inference on example1.png', () {
      final det = File('../../models/PPOCR_v6/det.rten');
      final rec = File('../../models/PPOCR_v6/rec.rten');
      final dict = File('../../models/PPOCR_v6/dict.txt');
      final img = File('../../models/test_images/example1.png');

      if (!det.existsSync() || !img.existsSync()) {
        return;
      }

      final config = '{"detection":{"model_path":"${det.absolute.path}"},"recognition":{"model_path":"${rec.absolute.path}","dict_path":"${dict.absolute.path}"}}';
      final engine = RustO.initialize(config);

      final results = engine.detectTextFile(img.absolute.path);
      expect(results, isNotEmpty);
      expect(results.any((r) => r.text.contains('INVOICE')), isTrue);

      final spatial = engine.detectTextFileSpatial(img.absolute.path);
      expect(spatial, contains('INVOICE'));

      engine.dispose();
    });
  });
}
