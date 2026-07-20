import 'package:flutter/widgets.dart';

import 'stacks.dart';

/// A scrollable stack — a [SingleChildScrollView] wrapping a [VStack]/[HStack]
/// of [children] with even [spacing]. The common "scrolling column of widgets"
/// in one primitive.
///
/// ```dart
/// Scroll(
///   padding: const EdgeInsets.all(16),
///   spacing: 12,
///   children: [Header(), Card(), Card(), Footer()],
/// )
/// ```
class Scroll extends StatelessWidget {
  const Scroll({
    super.key,
    this.axis = Axis.vertical,
    this.spacing = 0,
    this.padding,
    this.crossAxisAlignment = CrossAxisAlignment.start,
    this.controller,
    this.physics,
    this.reverse = false,
    this.children = const <Widget>[],
  });

  final Axis axis;

  /// Gap between children along [axis].
  final double spacing;
  final EdgeInsetsGeometry? padding;
  final CrossAxisAlignment crossAxisAlignment;
  final ScrollController? controller;
  final ScrollPhysics? physics;
  final bool reverse;
  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    final content = axis == Axis.vertical
        ? VStack(spacing: spacing, alignment: crossAxisAlignment, children: children)
        : HStack(spacing: spacing, alignment: crossAxisAlignment, children: children);
    return SingleChildScrollView(
      scrollDirection: axis,
      padding: padding,
      controller: controller,
      physics: physics,
      reverse: reverse,
      child: content,
    );
  }
}
