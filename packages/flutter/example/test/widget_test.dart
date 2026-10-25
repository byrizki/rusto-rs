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

    // Poll until all verification steps complete (up to 60 seconds for slower CI runners)
    final stopwatch = Stopwatch()..start();
    while (stopwatch.elapsed < const Duration(seconds: 60)) {
      await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 500)));
      await tester.pump();
      if (find.text('SUCCESS').evaluate().length == 5 ||
          find.text('FAILURE').evaluate().isNotEmpty) {
        break;
      }
    }

    final failureTexts = <String>[];
    for (final element in find.byType(Text).evaluate()) {
      final widget = element.widget as Text;
      if (widget.key != null &&
          widget.key.toString().contains('test-details-') &&
          widget.data != null) {
        failureTexts.add('${widget.key}: ${widget.data}');
      }
    }

    // Assert all 5 verification tests executed and succeeded
    expect(find.text('FAILURE'), findsNothing,
        reason:
            'One or more verification steps reported FAILURE:\n${failureTexts.join('\n')}');
    expect(find.text('SUCCESS'), findsNWidgets(5),
        reason:
            'Verification did not complete with 5 SUCCESS states within timeout');
    expect(find.byKey(const Key('latest-ocr-card')), findsOneWidget);
    expect(find.textContaining('Boxes Recognized: 45'), findsOneWidget);
  });
}

