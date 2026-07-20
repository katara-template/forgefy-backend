/// Forgefy layout & animation primitives for Flutter.
///
/// Declarative building blocks that remove the bulk of manual layout and
/// entrance-animation code from generated apps:
///
/// - Layout: [VStack], [HStack], [Grid], [Responsive], [Spacer], [Wrap], [Scroll]
/// - Slivers (recommended for screens): [SliverScreen], [SliverHeader],
///   [SliverListView], [SliverGridView], [SliverStack], [SliverBox],
///   [SliverGap], [SliverStagger], [SliverFill]
/// - Animation: [FadeIn], [SlideIn], [ScaleIn], [AnimatedVisibility], [Stagger]
///
/// Note: [Spacer] and [Wrap] share a name with Flutter's own widgets. In a file
/// that also imports `package:flutter/material.dart`, hide the two you don't
/// want to disambiguate:
///
/// ```dart
/// import 'package:flutter/material.dart' hide Spacer, Wrap;
/// import 'package:forgefy_ui/forgefy_ui.dart';
/// ```
library;

export 'src/animation/animations.dart';
export 'src/grid.dart';
export 'src/responsive.dart';
export 'src/scroll.dart';
export 'src/slivers.dart';
export 'src/spacer.dart';
export 'src/stacks.dart' hide gapped;
export 'src/wrap.dart';
