# forgefy_ui

Layout & animation **primitives** for Forgefy-generated Flutter apps — the
declarative building blocks that remove the bulk of manual layout and
entrance-animation code, so the build agent (and you) write intent, not
`SizedBox`/`AnimationController` boilerplate.

Standalone Flutter package — separate from [`forgefy_client`](../dart_client)
(data/auth), which stays dependency-free. `forgefy_ui` depends only on Flutter.

## Primitives

| Layout | What it is |
|---|---|
| `VStack` / `HStack` | `Column`/`Row` with an even `spacing` gap between children |
| `Grid` | Fixed `columns`, children auto-sized to width — works inside a `Column`/`Scroll` |
| `Responsive` | Widget per breakpoint, plus `Responsive.value(...)` for numbers/spacing |
| `Spacer` | Flexible (expands) or fixed (`Spacer(size: 24)`) space |
| `Wrap` | Flow layout that wraps to the next run |
| `Scroll` | A `SingleChildScrollView` wrapping a gap-stack of children |

| Slivers (recommended for screens) | What it is |
|---|---|
| `SliverScreen` | A `CustomScrollView` assembled from slivers, with an optional app bar + side `gutter` |
| `SliverHeader` | Collapsing/pinned app bar (`SliverAppBar`); add `expandedHeight` + `background` to collapse |
| `SliverListView` | Lazily-built list sliver with even `spacing` |
| `SliverGridView` | Lazily-built fixed-column grid sliver |
| `SliverStagger` | Lazy list whose items animate in (delay capped by `maxStagger`) |
| `SliverStack` / `SliverBox` | A small group / a single box widget as a sliver |
| `SliverGap` | Vertical space between sliver sections |
| `SliverFill` | Fills the remaining viewport (centered) — empty / error / loading states |

| Animation (baked in) | What it does |
|---|---|
| `FadeIn` | Fades a child in on mount (`delay`, `duration`, `curve`) |
| `SlideIn` | Slides + fades in from a pixel `begin` offset |
| `ScaleIn` | Scales + fades in from a `begin` factor |
| `AnimatedVisibility` | Animated show/hide with a size collapse driven by `visible` |
| `Stagger` | Lays children in a stack and animates them in one-by-one |

## Import (note the name overlap)

`Spacer` and `Wrap` share a name with Flutter's own widgets. In a file that also
imports Material, hide those two so the reference is unambiguous:

```dart
import 'package:flutter/material.dart' hide Spacer, Wrap;
import 'package:forgefy_ui/forgefy_ui.dart';
```

## Examples

```dart
// A scrolling column with even spacing — no manual SizedBoxes.
Scroll(
  padding: const EdgeInsets.all(16),
  spacing: 16,
  children: [
    HStack(children: [Title(), Spacer(), SettingsButton()]), // push to the ends
    Wrap(spacing: 8, runSpacing: 8, children: tags),
    Grid(
      columns: Responsive.value(context, mobile: 2, tablet: 3, desktop: 4),
      spacing: 12,
      runSpacing: 12,
      children: products,
    ),
  ],
)
```

```dart
// Animate a list in, one card at a time.
Stagger(
  spacing: 12,
  effect: StaggerEffect.slideUp,
  children: todos.map(TodoCard.new).toList(),
)

// Show an error banner that animates its own space open/closed.
AnimatedVisibility(visible: hasError, child: ErrorBanner(message))
```

## Sliver-first screens (recommended)

Build screens as a `SliverScreen` (a `CustomScrollView`) so lists stay lazy,
headers collapse, and empty states fill the viewport — the performant default
for scrolling screens.

```dart
SliverScreen(
  gutter: 16, // horizontal side padding on every body sliver
  appBar: SliverHeader(
    title: const Text('Feed'),
    pinned: true,
    expandedHeight: 200,
    background: Image.asset('assets/hero.jpg', fit: BoxFit.cover),
  ),
  slivers: [
    const SliverGap(12),
    if (isEmpty)
      SliverFill(child: EmptyState(message: 'No posts yet'))
    else
      SliverStagger(
        itemCount: posts.length,
        spacing: 12,
        itemBuilder: (context, i) => PostCard(posts[i]),
      ),
  ],
)
```

Use `SliverGap` for vertical rhythm between sections (a full-inset padding per
sliver would double the gaps — `gutter` handles the horizontal side).

## Development

```bash
flutter pub get
flutter analyze
flutter test
```
