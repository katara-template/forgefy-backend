import 'package:flutter/widgets.dart';

/// A fixed-column grid that sizes its children to the available width — so it
/// works inside a [Column]/[Scroll] without the bounded-height ceremony a raw
/// `GridView` needs.
///
/// ```dart
/// Grid(
///   columns: 2,
///   spacing: 12,
///   runSpacing: 12,
///   children: products.map(ProductCard.new).toList(),
/// )
/// ```
class Grid extends StatelessWidget {
  const Grid({
    super.key,
    required this.columns,
    this.spacing = 0,
    this.runSpacing = 0,
    this.children = const <Widget>[],
  }) : assert(columns > 0, 'Grid.columns must be greater than 0');

  /// Number of items per row.
  final int columns;

  /// Horizontal gap between items in a row.
  final double spacing;

  /// Vertical gap between rows.
  final double runSpacing;
  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final totalGaps = spacing * (columns - 1);
        final available = constraints.maxWidth - totalGaps;
        final itemWidth = available <= 0 ? 0.0 : available / columns;
        return Wrap(
          spacing: spacing,
          runSpacing: runSpacing,
          children: [
            for (final child in children)
              SizedBox(width: itemWidth, child: child),
          ],
        );
      },
    );
  }
}
