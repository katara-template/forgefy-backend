// Flutter ships its own `Spacer`; ours is a superset (flexible *or* fixed), so
// we hide the framework's to avoid an ambiguous reference in this library.
import 'package:flutter/widgets.dart' hide Spacer;

/// Space inside a [VStack]/[HStack]/[Row]/[Column].
///
/// - Default (no [size]): flexible — expands to push siblings apart, like
///   Flutter's `Spacer` (honours [flex]).
/// - With [size]: a fixed square gap, like a `SizedBox(width/height: size)` that
///   works in either axis.
///
/// ```dart
/// HStack(children: [Back(), Spacer(), Save()]);   // pushes Save to the end
/// VStack(children: [Header(), Spacer(size: 24), Body()]);
/// ```
class Spacer extends StatelessWidget {
  const Spacer({super.key, this.size, this.flex = 1})
      : assert(size == null || size >= 0, 'Spacer.size must be non-negative');

  /// Fixed extent in logical pixels. When null, the spacer is flexible.
  final double? size;

  /// Flex weight when flexible (ignored when [size] is set).
  final int flex;

  @override
  Widget build(BuildContext context) {
    if (size != null) return SizedBox(width: size, height: size);
    return Expanded(flex: flex, child: const SizedBox.shrink());
  }
}
