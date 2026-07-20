// Sliver-first primitives — the recommended way to build screens in forgefy_ui.
//
// A screen is a `SliverScreen` (a CustomScrollView) whose body is a list of
// slivers: SliverHeader (collapsing app bar), SliverListView / SliverGridView
// (lazy), SliverStack / SliverBox (fixed content), SliverStagger (animated
// list), and SliverFill (empty/error/loading states). SliverAppBar &
// FlexibleSpaceBar are Material, so this file imports material; the rest of the
// package stays widgets-only.
import 'package:flutter/material.dart';

import 'animation/animations.dart';
import 'stacks.dart';

/// A sliver-based screen: a [CustomScrollView] assembled from [slivers], with an
/// optional [appBar] pinned first and a horizontal [gutter] applied to every
/// body sliver. Use [SliverGap] for vertical rhythm between sections (applying a
/// full-inset padding per sliver would double the gaps).
///
/// ```dart
/// SliverScreen(
///   gutter: 16,
///   appBar: SliverHeader(title: Text('Feed'), pinned: true),
///   slivers: [
///     SliverGap(12),
///     SliverStagger(itemCount: posts.length, itemBuilder: (_, i) => PostCard(posts[i])),
///   ],
/// )
/// ```
class SliverScreen extends StatelessWidget {
  const SliverScreen({
    super.key,
    this.slivers = const <Widget>[],
    this.appBar,
    this.gutter = 0,
    this.controller,
    this.physics,
    this.reverse = false,
  });

  /// Body slivers, rendered after [appBar].
  final List<Widget> slivers;

  /// An optional leading sliver app bar — typically a [SliverHeader].
  final Widget? appBar;

  /// Horizontal padding applied to every body sliver (side gutters).
  final double gutter;
  final ScrollController? controller;
  final ScrollPhysics? physics;
  final bool reverse;

  @override
  Widget build(BuildContext context) {
    final pad = EdgeInsets.symmetric(horizontal: gutter);
    return CustomScrollView(
      controller: controller,
      physics: physics,
      reverse: reverse,
      slivers: [
        if (appBar != null) appBar!,
        for (final sliver in slivers)
          if (gutter > 0) SliverPadding(padding: pad, sliver: sliver) else sliver,
      ],
    );
  }
}

/// A collapsing/pinned app bar. Wraps [SliverAppBar]; provide [expandedHeight]
/// + [background] for a header that collapses as you scroll.
class SliverHeader extends StatelessWidget {
  const SliverHeader({
    super.key,
    this.title,
    this.pinned = true,
    this.floating = false,
    this.expandedHeight,
    this.background,
    this.actions,
    this.leading,
    this.centerTitle,
  });

  final Widget? title;
  final bool pinned;
  final bool floating;

  /// Height when fully expanded (enables the collapse effect with [background]).
  final double? expandedHeight;

  /// Content shown behind the title when expanded (e.g. a hero image).
  final Widget? background;
  final List<Widget>? actions;
  final Widget? leading;
  final bool? centerTitle;

  @override
  Widget build(BuildContext context) {
    return SliverAppBar(
      title: title,
      pinned: pinned,
      floating: floating,
      expandedHeight: expandedHeight,
      actions: actions,
      leading: leading,
      centerTitle: centerTitle,
      flexibleSpace: background == null
          ? null
          : FlexibleSpaceBar(background: background),
    );
  }
}

/// Drop a single box widget into the scroll — wraps [SliverToBoxAdapter].
class SliverBox extends StatelessWidget {
  const SliverBox({super.key, required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) => SliverToBoxAdapter(child: child);
}

/// Vertical space between sliver sections.
class SliverGap extends StatelessWidget {
  const SliverGap(this.size, {super.key});

  final double size;

  @override
  Widget build(BuildContext context) =>
      SliverToBoxAdapter(child: SizedBox(height: size));
}

/// A small, fixed group of box widgets as one sliver, with even [spacing]. For
/// long/lazy lists prefer [SliverListView].
class SliverStack extends StatelessWidget {
  const SliverStack({super.key, this.spacing = 0, this.children = const <Widget>[]});

  final double spacing;
  final List<Widget> children;

  @override
  Widget build(BuildContext context) =>
      SliverList.list(children: gapped(children, spacing, Axis.vertical));
}

/// A lazily-built list sliver with even [spacing] between items.
///
/// ```dart
/// SliverListView(itemCount: items.length, spacing: 12, itemBuilder: (_, i) => Row(items[i]));
/// ```
class SliverListView extends StatelessWidget {
  const SliverListView({
    super.key,
    required this.itemCount,
    required this.itemBuilder,
    this.spacing = 0,
  });

  final int itemCount;
  final IndexedWidgetBuilder itemBuilder;

  /// Vertical gap inserted after every item except the last.
  final double spacing;

  @override
  Widget build(BuildContext context) {
    return SliverList.builder(
      itemCount: itemCount,
      itemBuilder: (context, index) {
        final item = itemBuilder(context, index);
        if (spacing > 0 && index != itemCount - 1) {
          return Padding(padding: EdgeInsets.only(bottom: spacing), child: item);
        }
        return item;
      },
    );
  }
}

/// A lazily-built fixed-column grid sliver.
class SliverGridView extends StatelessWidget {
  const SliverGridView({
    super.key,
    required this.columns,
    required this.itemCount,
    required this.itemBuilder,
    this.spacing = 0,
    this.runSpacing = 0,
    this.childAspectRatio = 1,
  }) : assert(columns > 0, 'SliverGridView.columns must be greater than 0');

  final int columns;
  final int itemCount;
  final IndexedWidgetBuilder itemBuilder;

  /// Horizontal gap between columns.
  final double spacing;

  /// Vertical gap between rows.
  final double runSpacing;
  final double childAspectRatio;

  @override
  Widget build(BuildContext context) {
    return SliverGrid.builder(
      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: columns,
        mainAxisSpacing: runSpacing,
        crossAxisSpacing: spacing,
        childAspectRatio: childAspectRatio,
      ),
      itemCount: itemCount,
      itemBuilder: itemBuilder,
    );
  }
}

/// A lazily-built list sliver whose items animate in as they are built. Reuses
/// the [FadeIn]/[SlideIn]/[ScaleIn] entrances via [effect]. The per-item delay
/// grows with index but is capped at [maxStagger] so items deep in a long list
/// don't wait forever.
class SliverStagger extends StatelessWidget {
  const SliverStagger({
    super.key,
    required this.itemCount,
    required this.itemBuilder,
    this.effect = StaggerEffect.slideUp,
    this.interval = const Duration(milliseconds: 60),
    this.duration = const Duration(milliseconds: 300),
    this.spacing = 0,
    this.maxStagger = 10,
  });

  final int itemCount;
  final IndexedWidgetBuilder itemBuilder;
  final StaggerEffect effect;
  final Duration interval;
  final Duration duration;
  final double spacing;

  /// Highest index that still gets an incremental delay.
  final int maxStagger;

  @override
  Widget build(BuildContext context) {
    return SliverList.builder(
      itemCount: itemCount,
      itemBuilder: (context, index) {
        final steps = index > maxStagger ? maxStagger : index;
        Widget item = staggerEntrance(
          effect,
          delay: interval * steps,
          duration: duration,
          child: itemBuilder(context, index),
        );
        if (spacing > 0 && index != itemCount - 1) {
          item = Padding(padding: EdgeInsets.only(bottom: spacing), child: item);
        }
        return item;
      },
    );
  }
}

/// Fills the remaining viewport with centered [child] — for empty, error, and
/// loading states inside a [SliverScreen].
///
/// ```dart
/// SliverFill(child: EmptyState(message: 'No posts yet'))
/// ```
class SliverFill extends StatelessWidget {
  const SliverFill({
    super.key,
    required this.child,
    this.hasScrollBody = false,
    this.fillOverscroll = false,
    this.center = true,
  });

  final Widget child;
  final bool hasScrollBody;
  final bool fillOverscroll;

  /// Center the child in the remaining space (typical for empty states).
  final bool center;

  @override
  Widget build(BuildContext context) {
    return SliverFillRemaining(
      hasScrollBody: hasScrollBody,
      fillOverscroll: fillOverscroll,
      child: center ? Center(child: child) : child,
    );
  }
}
