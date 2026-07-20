// Flutter ships its own `Wrap`; we expose the same widget under the Forgefy
// primitive vocabulary. Hide the framework's name here and reach it via the
// prefixed import so our class can delegate to it.
import 'package:flutter/widgets.dart' hide Wrap;
import 'package:flutter/widgets.dart' as widgets show Wrap;

/// Flow layout: lays children out along [direction], wrapping to the next run
/// when they overflow. A thin, consistently-named pass-through to Flutter's
/// `Wrap` so chips/tags/filters read in the same vocabulary as the other
/// primitives.
///
/// ```dart
/// Wrap(
///   spacing: 8,
///   runSpacing: 8,
///   children: tags.map(TagChip.new).toList(),
/// )
/// ```
class Wrap extends StatelessWidget {
  const Wrap({
    super.key,
    this.spacing = 0,
    this.runSpacing = 0,
    this.direction = Axis.horizontal,
    this.alignment = WrapAlignment.start,
    this.runAlignment = WrapAlignment.start,
    this.crossAxisAlignment = WrapCrossAlignment.start,
    this.children = const <Widget>[],
  });

  /// Gap between children within a run.
  final double spacing;

  /// Gap between runs.
  final double runSpacing;
  final Axis direction;
  final WrapAlignment alignment;
  final WrapAlignment runAlignment;
  final WrapCrossAlignment crossAxisAlignment;
  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    return widgets.Wrap(
      spacing: spacing,
      runSpacing: runSpacing,
      direction: direction,
      alignment: alignment,
      runAlignment: runAlignment,
      crossAxisAlignment: crossAxisAlignment,
      children: children,
    );
  }
}
