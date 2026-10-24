import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:rusto_flutter_example/main.dart';

void main() {
  testWidgets('RustO Flutter OCR UI smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const RustOExampleApp());

    expect(find.text('RustO! OCR'), findsOneWidget);
    expect(find.byKey(const Key('run-tests-button')), findsOneWidget);
    expect(find.byKey(const Key('test-status-mod')), findsOneWidget);
    expect(find.text('IDLE'), findsNWidgets(5));
  });
}
