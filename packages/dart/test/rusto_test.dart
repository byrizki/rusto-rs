import 'dart:io';
import 'package:test/test.dart';
import 'package:rusto/rusto.dart';

void main() {
  group('RustO Dart Binding Tests', () {
    test('Invalid JSON initialization throws StateError', () {
      expect(
        () => RustO.initialize('invalid-json-config'),
        throwsA(isA<StateError>()),
      );
    });

    test('Frame and TextResult structure validation', () {
      const frame = Frame(width: 120, height: 24, top: 10, left: 20);
      expect(frame.width, equals(120));
      expect(frame.height, equals(24));
      expect(frame.top, equals(10));
      expect(frame.left, equals(20));

      const result = TextResult(
        text: 'PaddleOCR RustO',
        score: 0.99,
        boxPoints: [
          Point2D(x: 20, y: 10),
          Point2D(x: 140, y: 10),
          Point2D(x: 140, y: 34),
          Point2D(x: 20, y: 34),
        ],
        frame: frame,
      );

      expect(result.text, equals('PaddleOCR RustO'));
      expect(result.score, equals(0.99));
      expect(result.frame, equals(frame));
      expect(result.boxPoints[0].x, equals(20));
      expect(result.boxPoints[0].y, equals(10));
    });

    test('Real OCR inference on example1.png', () {
      final det = File('../../models/PPOCR_v6/det.rten');
      final rec = File('../../models/PPOCR_v6/rec.rten');
      final dict = File('../../models/PPOCR_v6/dict.txt');
      final img = File('../../models/test_images/example1.png');

      if (!det.existsSync() || !img.existsSync()) {
        return;
      }

      final config =
          '{"detection":{"model_path":"${det.absolute.path}"},"recognition":{"model_path":"${rec.absolute.path}","dict_path":"${dict.absolute.path}"}}';
      final engine = RustO.initialize(config);

      final results = engine.detectTextFile(img.absolute.path);
      expect(results, isNotEmpty);
      expect(results.any((r) => r.text.contains('INVOICE')), isTrue);

      final spatial = engine.detectTextFileSpatial(img.absolute.path);
      expect(spatial, contains('INVOICE'));

      engine.dispose();
    });

    test('UniFFI typed config and options inference', () {
      final det = File('../../models/PPOCR_v6/det.rten');
      final rec = File('../../models/PPOCR_v6/rec.rten');
      final dict = File('../../models/PPOCR_v6/dict.txt');
      final img = File('../../models/test_images/example1.png');

      if (!det.existsSync() || !img.existsSync()) {
        return;
      }

      final config = createInitializeConfigPpv6(
        det.absolute.path,
        rec.absolute.path,
        dict.absolute.path,
      );
      final engine = RustO.initialize(config);

      final options = defaultOcrRunOptions();
      final res =
          engine.detectText(ImageSourcePath(path: img.absolute.path), options);
      expect(res, isA<DetectTextResultStructured>());
      final structured = res as DetectTextResultStructured;
      expect(structured.results, isNotEmpty);
      expect(structured.results.any((r) => r.text.contains('INVOICE')), isTrue);

      engine.close();
    });
  });
}
