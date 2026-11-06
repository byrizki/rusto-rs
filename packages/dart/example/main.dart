import 'package:rusto/rusto.dart';

void main() {
  final engine = RustO.initialize();
  print('RustO Engine version: ${engine.version()}');

  final result = engine
      .detectTextBuilder()
      .image('sample.png')
      .withLayoutReading()
      .asLines()
      .run();

  if (result is DetectTextResultStructured) {
    for (final item in result.results) {
      print('Detected: "${item.text}" (confidence: ${item.score})');
    }
  }

  engine.close();
}
