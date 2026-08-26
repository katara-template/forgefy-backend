import 'package:flutter/material.dart' hide Spacer, Wrap;
import 'package:flutter_test/flutter_test.dart';
import 'package:forgefy_ui/forgefy_ui.dart';

/// Pump [child] inside a sized, directional context so layout widgets resolve.
Future<void> _pump(WidgetTester tester, Widget child, {Size size = const Size(1200, 800)}) async {
  tester.view.physicalSize = size;
  tester.view.devicePixelRatio = 1.0;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(
    MediaQuery(
      data: MediaQueryData(size: size),
      child: Directionality(textDirection: TextDirection.ltr, child: child),
    ),
  );
}

void main() {
  group('layout', () {
    testWidgets('VStack inserts a gap between children only', (tester) async {
      await _pump(
        tester,
        const VStack(spacing: 10, children: [Text('a'), Text('b'), Text('c')]),
      );
      // 3 children → exactly 2 gaps.
      expect(find.byType(SizedBox), findsNWidgets(2));
      expect(find.text('a'), findsOneWidget);
      expect(find.text('c'), findsOneWidget);
    });

    testWidgets('VStack with a single child inserts no gap', (tester) async {
      await _pump(tester, const VStack(spacing: 10, children: [Text('solo')]));
      expect(find.byType(SizedBox), findsNothing);
    });

    testWidgets('Grid sizes children to width / columns', (tester) async {
      await _pump(
        tester,
        const Grid(
          columns: 2,
          children: [SizedBox(height: 20), SizedBox(height: 20)],
        ),
        size: const Size(400, 800),
      );
      // No spacing → each of 2 columns is 200 wide within a 400 viewport.
      final firstCell = tester.widgetList<SizedBox>(find.byType(SizedBox)).first;
      expect(firstCell.width, 200);
    });

    testWidgets('Responsive.value picks the breakpoint bucket', (tester) async {
      late int cols;
      await _pump(
        tester,
        Builder(builder: (context) {
          cols = Responsive.value(context, mobile: 1, tablet: 2, desktop: 4);
          return const SizedBox();
        }),
        size: const Size(1200, 800), // desktop
      );
      expect(cols, 4);
    });

    testWidgets('Spacer(size:) is a fixed box; bare Spacer expands', (tester) async {
      await _pump(tester, const HStack(children: [Text('x'), Spacer(size: 24), Text('y')]));
      expect(find.byType(SizedBox), findsWidgets);

      await _pump(tester, const Row(children: [Text('x'), Spacer(), Text('y')]));
      expect(find.byType(Expanded), findsOneWidget);
    });
  });

  group('animation', () {
    testWidgets('FadeIn animates opacity 0 → 1 over its duration', (tester) async {
      await _pump(
        tester,
        const FadeIn(duration: Duration(milliseconds: 200), child: Text('hi')),
      );
      // At t=0 the child is transparent…
      Opacity opacity() => tester.widget<Opacity>(find.byType(Opacity));
      expect(opacity().opacity, 0.0);

      await tester.pump(const Duration(milliseconds: 200));
      expect(opacity().opacity, 1.0);
    });

    testWidgets('SlideIn respects its delay before starting', (tester) async {
      await _pump(
        tester,
        const SlideIn(delay: Duration(milliseconds: 100), child: Text('later')),
      );
      double opacity() => tester.widget<Opacity>(find.byType(Opacity)).opacity;
      expect(opacity(), 0.0);

      await tester.pump(const Duration(milliseconds: 50)); // still within delay
      expect(opacity(), 0.0);

      await tester.pump(const Duration(milliseconds: 350)); // past delay + duration
      expect(opacity(), 1.0);
      await tester.pumpAndSettle();
    });

    testWidgets('Stagger renders one wrapper per child', (tester) async {
      await _pump(
        tester,
        const Stagger(
          effect: StaggerEffect.fade,
          children: [Text('1'), Text('2'), Text('3')],
        ),
      );
      expect(find.byType(FadeIn), findsNWidgets(3));
      await tester.pumpAndSettle();
    });
  });

  group('slivers', () {
    testWidgets('SliverScreen builds a CustomScrollView with a lazy list', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SliverScreen(
              gutter: 16,
              slivers: [
                SliverListView(
                  itemCount: 3,
                  spacing: 8,
                  itemBuilder: (context, i) => Text('row $i'),
                ),
              ],
            ),
          ),
        ),
      );
      expect(find.byType(CustomScrollView), findsOneWidget);
      expect(find.text('row 0'), findsOneWidget);
      expect(find.text('row 2'), findsOneWidget);
    });

    testWidgets('SliverHeader renders as a pinned SliverAppBar', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: SliverScreen(
              appBar: SliverHeader(title: Text('Feed'), pinned: true),
              slivers: [SliverGap(24)],
            ),
          ),
        ),
      );
      expect(find.byType(SliverAppBar), findsOneWidget);
      expect(find.text('Feed'), findsOneWidget);
    });

    testWidgets('SliverStagger wraps each item in an entrance', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SliverScreen(
              slivers: [
                SliverStagger(
                  itemCount: 3,
                  effect: StaggerEffect.fade,
                  itemBuilder: (context, i) => SizedBox(height: 40, child: Text('item $i')),
                ),
              ],
            ),
          ),
        ),
      );
      expect(find.byType(FadeIn), findsWidgets);
      await tester.pumpAndSettle();
      expect(find.text('item 0'), findsOneWidget);
    });

    testWidgets('SliverFill centers empty-state content', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: SliverScreen(
              slivers: [SliverFill(child: Text('No posts yet'))],
            ),
          ),
        ),
      );
      expect(find.text('No posts yet'), findsOneWidget);
      expect(find.byType(Center), findsWidgets);
    });
  });
}
