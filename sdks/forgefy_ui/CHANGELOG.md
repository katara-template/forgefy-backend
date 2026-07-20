# Changelog

## 0.1.0

Initial release — layout & animation primitives for Forgefy-generated Flutter
apps.

- **Layout**: `VStack`, `HStack` (gap-aware Column/Row), `Grid` (fixed columns,
  auto-sized to width), `Responsive` (widget + `Responsive.value`), `Spacer`
  (flexible or fixed), `Wrap`, `Scroll` (scrolling gap-stack).
- **Slivers** (recommended for screens): `SliverScreen`, `SliverHeader`,
  `SliverListView`, `SliverGridView`, `SliverStack`, `SliverBox`, `SliverGap`,
  `SliverStagger` (lazy animated list), `SliverFill`.
- **Animation** (baked in): `FadeIn`, `SlideIn`, `ScaleIn`, `AnimatedVisibility`
  (animated show/hide + collapse), and `Stagger` (sequenced list entrance).
