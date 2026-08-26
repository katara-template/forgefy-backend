import 'package:flutter/widgets.dart';

/// A vertical stack — a [Column] with a [spacing] gap inserted between children,
/// so you never hand-place `SizedBox`es for even spacing.
///
/// ```dart
/// VStack(
///   spacing: 12,
///   children: [Title(), Body(), Actions()],
/// )
/// ```
class VStack extends StatelessWidget {
  const VStack({
    super.key,
    this.spacing = 0,
    this.alignment = CrossAxisAlignment.start,
    this.mainAxisAlignment = MainAxisAlignment.start,
    this.mainAxisSize = MainAxisSize.min,
    this.children = const <Widget>[],
  });

  /// Logical-pixel gap inserted between adjacent children (not before the first
  /// or after the last).
  final double spacing;

  /// Horizontal alignment of children (the cross axis of a column).
  final CrossAxisAlignment alignment;
  final MainAxisAlignment mainAxisAlignment;
  final MainAxisSize mainAxisSize;
  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: alignment,
      mainAxisAlignment: mainAxisAlignment,
      mainAxisSize: mainAxisSize,
      children: gapped(children, spacing, Axis.vertical),
    );
  }
}

/// A horizontal stack — a [Row] with a [spacing] gap between children.
///
/// ```dart
/// HStack(
///   spacing: 8,
///   alignment: CrossAxisAlignment.center,
///   children: [Icon(...), Label()],
/// )
/// ```
class HStack extends StatelessWidget {
  const HStack({
    super.key,
    this.spacing = 0,
    this.alignment = CrossAxisAlignment.center,
    this.mainAxisAlignment = MainAxisAlignment.start,
    this.mainAxisSize = MainAxisSize.min,
    this.children = const <Widget>[],
  });

  /// Logical-pixel gap inserted between adjacent children.
  final double spacing;

  /// Vertical alignment of children (the cross axis of a row).
  final CrossAxisAlignment alignment;
  final MainAxisAlignment mainAxisAlignment;
  final MainAxisSize mainAxisSize;
  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: alignment,
      mainAxisAlignment: mainAxisAlignment,
      mainAxisSize: mainAxisSize,
      children: gapped(children, spacing, Axis.horizontal),
    );
  }
}

/// Interleave fixed-size gaps between [children]. Shared by [VStack]/[HStack]
/// and [Scroll]; not part of the public API.
List<Widget> gapped(List<Widget> children, double spacing, Axis axis) {
  if (spacing <= 0 || children.length < 2) return children;
  final gap = axis == Axis.vertical
      ? SizedBox(height: spacing)
      : SizedBox(width: spacing);
  final out = <Widget>[];
  for (var i = 0; i < children.length; i++) {
    out.add(children[i]);
    if (i != children.length - 1) out.add(gap);
  }
  return out;
}
