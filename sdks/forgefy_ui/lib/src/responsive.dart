import 'package:flutter/widgets.dart';

/// Picks a widget for the current screen width. [tablet] and [desktop] are
/// optional and fall back to the next-smaller layout.
///
/// ```dart
/// Responsive(
///   mobile: SingleColumn(),
///   tablet: TwoColumn(),
///   desktop: ThreeColumn(),
/// )
/// ```
///
/// For non-widget values (a column count, a padding), use [Responsive.value].
class Responsive extends StatelessWidget {
  const Responsive({
    super.key,
    required this.mobile,
    this.tablet,
    this.desktop,
    this.tabletBreakpoint = 600,
    this.desktopBreakpoint = 1024,
  });

  final Widget mobile;
  final Widget? tablet;
  final Widget? desktop;

  /// Width at/above which [tablet] is used (default 600).
  final double tabletBreakpoint;

  /// Width at/above which [desktop] is used (default 1024).
  final double desktopBreakpoint;

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.sizeOf(context).width;
    if (width >= desktopBreakpoint) return desktop ?? tablet ?? mobile;
    if (width >= tabletBreakpoint) return tablet ?? mobile;
    return mobile;
  }

  /// Resolve a plain value by breakpoint — handy for column counts, spacing,
  /// font sizes, etc.
  ///
  /// ```dart
  /// final cols = Responsive.value(context, mobile: 1, tablet: 2, desktop: 4);
  /// ```
  static T value<T>(
    BuildContext context, {
    required T mobile,
    T? tablet,
    T? desktop,
    double tabletBreakpoint = 600,
    double desktopBreakpoint = 1024,
  }) {
    final width = MediaQuery.sizeOf(context).width;
    if (width >= desktopBreakpoint) return desktop ?? tablet ?? mobile;
    if (width >= tabletBreakpoint) return tablet ?? mobile;
    return mobile;
  }
}
