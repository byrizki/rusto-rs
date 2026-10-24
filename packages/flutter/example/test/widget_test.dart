import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:rusto_flutter_example/main.dart';

void main() {
  testWidgets('RustO Flutter OCR UI smoke test and verification trigger', (WidgetTester tester) async {
    await tester.pumpWidget(const RustOExampleApp());

    expect(find.text('RustO! OCR'), findsOneWidget);
    expect(find.byKey(const Key('run-tests-button')), findsOneWidget);
    expect(find.byKey(const Key('test-status-mod')), findsOneWidget);
    expect(find.text('IDLE'), findsNWidgets(5));

    // Tap run tests button to verify state transition
    await tester.tap(find.byKey(const Key('run-tests-button')));
    await tester.pump();

    // Verify button goes to running / disabled
    expect(find.byKey(const Key('run-tests-button')), findsOneWidget);

    // Allow real background file I/O and FFI initialization to complete
    await tester.runAsync(() async {
      await Future<void>.delayed(const Duration(milliseconds: 3500));
    });
    await tester.pump();

    // Assert all 5 verification tests executed and succeeded
    expect(find.text('SUCCESS'), findsNWidgets(5));
    expect(find.byKey(const Key('latest-ocr-card')), findsOneWidget);
    expect(find.textContaining('Boxes Recognized:'), findsOneWidget);
  });
}

