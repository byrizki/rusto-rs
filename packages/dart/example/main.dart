import 'package:rusto/rusto.dart';

void main() {
  // Configure and initialize RustO OCR engine
  // using either a JSON string or typed InitializeConfig:
  final config = createInitializeConfigPpv6(
    'models/det.rten',
    'models/rec.rten',
    'models/dict.txt',
  );

  final engine = RustO.initialize(config);
  print('RustO Engine version: ${engine.version()}');

  // Perform text recognition on an image file:
  final options = defaultOcrRunOptions();
  final result = engine.detectText(
    ImageSourcePath(path: 'sample.png'),
    options,
  );

  if (result is DetectTextResultStructured) {
    for (final item in result.results) {
      print('Detected: "${item.text}" (confidence: ${item.score})');
    }
  }

  // Release engine resources when done:
  engine.close();
}
