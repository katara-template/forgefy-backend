import type { CSSProperties } from "react";

import {
  type FillBaseProps,
  type GridBaseProps,
  type ScrollBaseProps,
  type SpacerBaseProps,
  type StackBaseProps,
  toAlign,
  toJustify,
  type WrapBaseProps,
} from "../shared";

type Styled<P> = P & { style?: CSSProperties };

/** Vertical flex stack with an even `gap`. */
export function VStack({ gap = 0, align, justify, style, children }: Styled<StackBaseProps>) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap,
        alignItems: toAlign(align),
        justifyContent: toJustify(justify),
        ...style,
      }}
    >
      {children}
    </div>
  );
}

/** Horizontal flex stack with an even `gap`. */
export function HStack({ gap = 0, align = "center", justify, style, children }: Styled<StackBaseProps>) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        gap,
        alignItems: toAlign(align),
        justifyContent: toJustify(justify),
        ...style,
      }}
    >
      {children}
    </div>
  );
}

/** Fixed-column CSS grid. */
export function Grid({ columns, gap = 0, runGap, style, children }: Styled<GridBaseProps>) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
        columnGap: gap,
        rowGap: runGap ?? gap,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

/** Flexible (expands) or fixed-`size` space. */
export function Spacer({ size, style }: Styled<SpacerBaseProps>) {
  if (size != null) return <div style={{ width: size, height: size, flex: "0 0 auto", ...style }} />;
  return <div style={{ flex: "1 1 0%", ...style }} />;
}

/** Flow layout that wraps to the next line. */
export function Wrap({ gap = 0, runGap, align, style, children }: Styled<WrapBaseProps>) {
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        columnGap: gap,
        rowGap: runGap ?? gap,
        alignItems: toAlign(align),
        ...style,
      }}
    >
      {children}
    </div>
  );
}

/** A scrolling flex stack of children. */
export function Scroll({ horizontal, gap = 0, align, style, children }: Styled<ScrollBaseProps>) {
  return (
    <div
      style={{
        overflowX: horizontal ? "auto" : "hidden",
        overflowY: horizontal ? "hidden" : "auto",
        display: "flex",
        flexDirection: horizontal ? "row" : "column",
        gap,
        alignItems: toAlign(align),
        ...style,
      }}
    >
      {children}
    </div>
  );
}

/** Fills its flex parent and centers `children` — for empty/error/loading states. */
export function Fill({ children, style }: Styled<FillBaseProps>) {
  return (
    <div
      style={{ flex: "1 1 0%", display: "flex", alignItems: "center", justifyContent: "center", ...style }}
    >
      {children}
    </div>
  );
}
