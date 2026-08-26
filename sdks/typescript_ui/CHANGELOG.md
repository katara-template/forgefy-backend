# Changelog

## 0.1.0

Initial release — layout & animation primitives for React, the TypeScript
sibling of the Flutter `forgefy_ui` package. One package, two entry points:
`@forgefy/ui/web` (Next.js / React DOM) and `@forgefy/ui/native` (React Native),
sharing prop types and flexbox mapping.

- **Layout**: `VStack`, `HStack`, `Grid`, `Responsive` (+ `useResponsiveValue`),
  `Spacer`, `Wrap`, `Scroll`, `List` (FlatList on native, mapped on web), `Fill`.
- **Animation**: `FadeIn`, `SlideIn`, `ScaleIn`, `AnimatedVisibility`
  (animated show/hide + collapse), `Stagger`. Web uses CSS transitions; native
  uses the RN `Animated` API.
