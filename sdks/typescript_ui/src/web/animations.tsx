import { Children, type CSSProperties, type ReactNode, useEffect, useState } from "react";

import type {
  AnimatedVisibilityBaseProps,
  EntranceProps,
  ScaleInBaseProps,
  SlideInBaseProps,
  StaggerBaseProps,
  StaggerEffect,
} from "../shared";

/** Flip to `true` `delay` ms after mount, so a CSS transition can run. */
function useEntrance(delay: number): boolean {
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const id = setTimeout(() => setShown(true), delay);
    return () => clearTimeout(id);
  }, [delay]);
  return shown;
}

/** Fade a child in on mount. */
export function FadeIn({ children, duration = 300, delay = 0, style }: EntranceProps & { style?: CSSProperties }) {
  const shown = useEntrance(delay);
  return (
    <div style={{ opacity: shown ? 1 : 0, transition: `opacity ${duration}ms ease-out`, ...style }}>
      {children}
    </div>
  );
}

/** Slide (and, by default, fade) a child in from an `x`/`y` offset. */
export function SlideIn({
  children,
  duration = 300,
  delay = 0,
  x = 0,
  y = 24,
  fade = true,
  style,
}: SlideInBaseProps & { style?: CSSProperties }) {
  const shown = useEntrance(delay);
  return (
    <div
      style={{
        transform: shown ? "none" : `translate(${x}px, ${y}px)`,
        opacity: fade ? (shown ? 1 : 0) : 1,
        transition: `transform ${duration}ms ease-out, opacity ${duration}ms ease-out`,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

/** Scale (and, by default, fade) a child in from `from` to 1. */
export function ScaleIn({
  children,
  duration = 300,
  delay = 0,
  from = 0.95,
  fade = true,
  style,
}: ScaleInBaseProps & { style?: CSSProperties }) {
  const shown = useEntrance(delay);
  return (
    <div
      style={{
        transform: shown ? "scale(1)" : `scale(${from})`,
        opacity: fade ? (shown ? 1 : 0) : 1,
        transition: `transform ${duration}ms ease-out, opacity ${duration}ms ease-out`,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

/**
 * Animated show/hide with a height collapse, using the `grid-template-rows:
 * 0fr → 1fr` technique (no height measurement needed).
 */
export function AnimatedVisibility({ visible, children, duration = 250 }: AnimatedVisibilityBaseProps) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateRows: visible ? "1fr" : "0fr",
        opacity: visible ? 1 : 0,
        transition: `grid-template-rows ${duration}ms ease, opacity ${duration}ms ease`,
      }}
    >
      <div style={{ overflow: "hidden", minHeight: 0 }}>{children}</div>
    </div>
  );
}

function entrance(effect: StaggerEffect, node: ReactNode, delay: number, duration: number): ReactNode {
  switch (effect) {
    case "fade":
      return (
        <FadeIn delay={delay} duration={duration}>
          {node}
        </FadeIn>
      );
    case "slideUp":
      return (
        <SlideIn delay={delay} duration={duration}>
          {node}
        </SlideIn>
      );
    case "scale":
      return (
        <ScaleIn delay={delay} duration={duration}>
          {node}
        </ScaleIn>
      );
  }
}

/** Lay children in a column and animate them in one after another. */
export function Stagger({
  effect = "slideUp",
  interval = 60,
  duration = 300,
  gap = 0,
  children,
  style,
}: StaggerBaseProps & { style?: CSSProperties }) {
  const items = Children.toArray(children);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap, ...style }}>
      {items.map((child, i) => (
        <div key={i}>{entrance(effect, child, interval * i, duration)}</div>
      ))}
    </div>
  );
}
