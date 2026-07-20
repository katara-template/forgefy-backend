# @forgefy/ui

Layout & animation **primitives** for Forgefy-generated React apps — the
declarative building blocks that remove the bulk of manual flexbox/StyleSheet
and entrance-animation code. The TypeScript sibling of the Flutter
[`forgefy_ui`](../forgefy_ui) package.

One package, **two entry points** (React Native and web render with different
primitives, so the component is chosen at import time):

```ts
import { VStack, Grid, Stagger } from "@forgefy/ui/web";     // Next.js / React DOM
import { VStack, Grid, Stagger } from "@forgefy/ui/native";  // React Native
```

Both expose the **same component API**; only the import path changes. Web uses
`div` + flexbox/grid + CSS transitions; native uses `View`/`FlatList` +
`Animated`. Zero runtime dependencies (React is a peer; React Native is an
optional peer for the native entry).

## Primitives

| Layout | What it is |
|---|---|
| `VStack` / `HStack` | Flex stack with an even `gap` |
| `Grid` | Fixed `columns` (CSS grid on web; measured flex-wrap on native) |
| `Responsive` | Node per breakpoint, plus `useResponsiveValue(...)` for numbers |
| `Spacer` | Flexible (expands) or fixed (`size`) space |
| `Wrap` | Flow layout that wraps |
| `Scroll` | A scrolling gap-stack |
| `List` | Lazy list — `FlatList` on native, mapped scroll on web |
| `Fill` | Fills the parent and centers content (empty/error/loading states) |

| Animation | What it does |
|---|---|
| `FadeIn` / `SlideIn` / `ScaleIn` | Entrance transitions on mount (`delay`, `duration`) |
| `AnimatedVisibility` | Animated show/hide with a height collapse, driven by `visible` |
| `Stagger` | Lays children in a column and animates them in one-by-one |

## Examples

```tsx
import { VStack, HStack, Spacer, Grid, Wrap, useResponsiveValue } from "@forgefy/ui/web";

function Dashboard() {
  const columns = useResponsiveValue({ mobile: 2, tablet: 3, desktop: 4 });
  return (
    <VStack gap={16}>
      <HStack>
        <Title />
        <Spacer />        {/* pushes Settings to the end */}
        <Settings />
      </HStack>
      <Wrap gap={8}>{tags.map((t) => <Tag key={t} label={t} />)}</Wrap>
      <Grid columns={columns} gap={12}>{products.map((p) => <Card key={p.id} {...p} />)}</Grid>
    </VStack>
  );
}
```

```tsx
// React Native — a lazy list whose rows animate in.
import { List, Stagger, Fill } from "@forgefy/ui/native";

todos.length === 0
  ? <Fill><EmptyState message="No todos yet" /></Fill>
  : <List data={todos} gap={12} renderItem={(t) => <TodoCard todo={t} />} />;

// Sequenced entrance for a small group (any platform):
<Stagger gap={12} effect="slideUp">
  {cards.map((c) => <Card key={c.id} {...c} />)}
</Stagger>
```

## Notes

- **`gap`** uses native flexbox `gap`, which requires **React Native ≥ 0.71**.
- **Web `List`** is not virtualized — it maps every item. For very long feeds,
  wrap a windowing library; the native `List` (`FlatList`) is already lazy.
- **SSR** — web `Responsive`/`useResponsiveValue` render `mobile` on the server
  and update after mount, so hydration stays stable.

## Development

```bash
npm install
npm run typecheck   # tsc over both entry points
npm run build       # emits dist/web, dist/native, dist (shared types)
```
