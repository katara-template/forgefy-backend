import 'dart:async';

import 'package:flutter/widgets.dart';

import '../stacks.dart';

/// Signature for turning a 0→1 entrance progress into a transformed [child].
typedef _EntranceBuilder = Widget Function(BuildContext context, double t, Widget child);

/// One-shot entrance driver shared by [FadeIn], [SlideIn] and [ScaleIn].
///
/// Runs a single forward animation on mount (after an optional [delay]) and
/// rebuilds the child through [builder]. Private — callers use the primitives.
class _Entrance extends StatefulWidget {
  const _Entrance({
    required this.child,
    required this.builder,
    required this.duration,
    required this.delay,
    required this.curve,
  });

  final Widget child;
  final _EntranceBuilder builder;
  final Duration duration;
  final Duration delay;
  final Curve curve;

  @override
  State<_Entrance> createState() => _EntranceState();
}

class _EntranceState extends State<_Entrance> with SingleTickerProviderStateMixin {
  late final AnimationController _controller =
      AnimationController(vsync: this, duration: widget.duration);
  late final Animation<double> _animation =
      CurvedAnimation(parent: _controller, curve: widget.curve);
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    if (widget.delay == Duration.zero) {
      _controller.forward();
    } else {
      _timer = Timer(widget.delay, () {
        if (mounted) _controller.forward();
      });
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) => widget.builder(context, _animation.value, child!),
      // The child is built once and reused across ticks — only the transform
      // wrapper rebuilds.
      child: widget.child,
    );
  }
}

/// Fades its [child] in on mount.
///
/// ```dart
/// FadeIn(delay: Duration(milliseconds: 100), child: Card());
/// ```
class FadeIn extends StatelessWidget {
  const FadeIn({
    super.key,
    required this.child,
    this.duration = const Duration(milliseconds: 300),
    this.delay = Duration.zero,
    this.curve = Curves.easeOut,
  });

  final Widget child;
  final Duration duration;
  final Duration delay;
  final Curve curve;

  @override
  Widget build(BuildContext context) {
    return _Entrance(
      duration: duration,
      delay: delay,
      curve: curve,
      builder: (context, t, child) => Opacity(opacity: t, child: child),
      child: child,
    );
  }
}

/// Slides (and, by default, fades) its [child] in from [begin] — a pixel offset
/// relative to the resting position.
///
/// ```dart
/// SlideIn(begin: Offset(0, 24), child: ListItem());  // rise up + fade
/// ```
class SlideIn extends StatelessWidget {
  const SlideIn({
    super.key,
    required this.child,
    this.begin = const Offset(0, 24),
    this.duration = const Duration(milliseconds: 300),
    this.delay = Duration.zero,
    this.curve = Curves.easeOut,
    this.fade = true,
  });

  final Widget child;

  /// Starting offset in logical pixels; animates to `Offset.zero`.
  final Offset begin;
  final Duration duration;
  final Duration delay;
  final Curve curve;
  final bool fade;

  @override
  Widget build(BuildContext context) {
    return _Entrance(
      duration: duration,
      delay: delay,
      curve: curve,
      builder: (context, t, child) {
        final offset = Offset(begin.dx * (1 - t), begin.dy * (1 - t));
        final moved = Transform.translate(offset: offset, child: child);
        return fade ? Opacity(opacity: t, child: moved) : moved;
      },
      child: child,
    );
  }
}

/// Scales (and, by default, fades) its [child] in from [begin] to 1.0.
///
/// ```dart
/// ScaleIn(begin: 0.9, child: Avatar());
/// ```
class ScaleIn extends StatelessWidget {
  const ScaleIn({
    super.key,
    required this.child,
    this.begin = 0.95,
    this.duration = const Duration(milliseconds: 300),
    this.delay = Duration.zero,
    this.curve = Curves.easeOut,
    this.fade = true,
  });

  final Widget child;

  /// Starting scale factor; animates to 1.0.
  final double begin;
  final Duration duration;
  final Duration delay;
  final Curve curve;
  final bool fade;

  @override
  Widget build(BuildContext context) {
    return _Entrance(
      duration: duration,
      delay: delay,
      curve: curve,
      builder: (context, t, child) {
        final scaled = Transform.scale(scale: begin + (1 - begin) * t, child: child);
        return fade ? Opacity(opacity: t, child: scaled) : scaled;
      },
      child: child,
    );
  }
}

/// Shows/hides [child] with an animated fade + collapse driven by [visible].
/// Unlike a bare `Visibility`, the space it occupies animates too.
///
/// ```dart
/// AnimatedVisibility(visible: hasError, child: ErrorBanner());
/// ```
class AnimatedVisibility extends StatelessWidget {
  const AnimatedVisibility({
    super.key,
    required this.visible,
    required this.child,
    this.duration = const Duration(milliseconds: 250),
    this.curve = Curves.easeInOut,
    this.axis = Axis.vertical,
  });

  final bool visible;
  final Widget child;
  final Duration duration;
  final Curve curve;

  /// Axis along which the widget collapses when hidden.
  final Axis axis;

  @override
  Widget build(BuildContext context) {
    return AnimatedSize(
      duration: duration,
      curve: curve,
      alignment: Alignment.topCenter,
      child: AnimatedOpacity(
        opacity: visible ? 1 : 0,
        duration: duration,
        curve: curve,
        child: visible
            ? child
            : SizedBox(
                width: axis == Axis.horizontal ? 0 : null,
                height: axis == Axis.vertical ? 0 : null,
              ),
      ),
    );
  }
}

/// Which entrance effect [Stagger] applies to each child.
enum StaggerEffect { fade, slideUp, scale }

/// Lays [children] out in a [VStack]/[HStack] and animates them in one after
/// another, each delayed by [interval] more than the last.
///
/// ```dart
/// Stagger(
///   spacing: 12,
///   children: todos.map(TodoCard.new).toList(),
/// )
/// ```
class Stagger extends StatelessWidget {
  const Stagger({
    super.key,
    required this.children,
    this.axis = Axis.vertical,
    this.spacing = 0,
    this.effect = StaggerEffect.slideUp,
    this.interval = const Duration(milliseconds: 60),
    this.duration = const Duration(milliseconds: 300),
    this.initialDelay = Duration.zero,
    this.crossAxisAlignment = CrossAxisAlignment.start,
  });

  final List<Widget> children;
  final Axis axis;

  /// Gap between children along [axis].
  final double spacing;
  final StaggerEffect effect;

  /// Extra delay added per child.
  final Duration interval;

  /// Duration of each child's own entrance.
  final Duration duration;

  /// Delay before the first child animates.
  final Duration initialDelay;
  final CrossAxisAlignment crossAxisAlignment;

  @override
  Widget build(BuildContext context) {
    final animated = <Widget>[
      for (var i = 0; i < children.length; i++)
        staggerEntrance(
          effect,
          delay: initialDelay + interval * i,
          duration: duration,
          child: children[i],
        ),
    ];
    return axis == Axis.vertical
        ? VStack(spacing: spacing, alignment: crossAxisAlignment, children: animated)
        : HStack(spacing: spacing, alignment: crossAxisAlignment, children: animated);
  }
}

/// Wrap [child] in the entrance transition matching [effect]. Shared by
/// [Stagger] and the sliver staggered list so both animate identically.
Widget staggerEntrance(
  StaggerEffect effect, {
  required Duration delay,
  required Duration duration,
  required Widget child,
}) =>
    switch (effect) {
      StaggerEffect.fade => FadeIn(delay: delay, duration: duration, child: child),
      StaggerEffect.slideUp => SlideIn(delay: delay, duration: duration, child: child),
      StaggerEffect.scale => ScaleIn(delay: delay, duration: duration, child: child),
    };
