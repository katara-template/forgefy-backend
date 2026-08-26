/**
 * Cross-platform types and flexbox helpers shared by the web and native
 * entry points of @forgefy/ui.
 *
 * The flexbox value names (`flex-start`, `space-between`, …) are identical
 * across CSS and React Native's style system, so the mapping lives here once.
 */
import type { ReactNode } from "react";

export type Align = "start" | "center" | "end" | "stretch";
export type Justify = "start" | "center" | "end" | "between" | "around" | "evenly";

/** Values valid for both CSS `align-items` and RN `alignItems`. */
export type FlexAlign = "flex-start" | "center" | "flex-end" | "stretch";
/** Values valid for both CSS `justify-content` and RN `justifyContent`. */
export type FlexJustify =
  | "flex-start"
  | "center"
  | "flex-end"
  | "space-between"
  | "space-around"
  | "space-evenly";

const ALIGN: Record<Align, FlexAlign> = {
  start: "flex-start",
  center: "center",
  end: "flex-end",
  stretch: "stretch",
};

const JUSTIFY: Record<Justify, FlexJustify> = {
  start: "flex-start",
  center: "center",
  end: "flex-end",
  between: "space-between",
  around: "space-around",
  evenly: "space-evenly",
};

export const toAlign = (a: Align = "stretch"): FlexAlign => ALIGN[a];
export const toJustify = (j: Justify = "start"): FlexJustify => JUSTIFY[j];

// ── Shared prop shapes (platform files add a typed `style`) ─────────────────

export interface StackBaseProps {
  /** Gap between children, in px/dp (native flex `gap`, requires RN >= 0.71). */
  gap?: number;
  /** Cross-axis alignment. */
  align?: Align;
  /** Main-axis distribution. */
  justify?: Justify;
  children?: ReactNode;
}

export interface GridBaseProps {
  columns: number;
  /** Horizontal gap between columns. */
  gap?: number;
  /** Vertical gap between rows (defaults to [gap]). */
  runGap?: number;
  children?: ReactNode;
}

export interface ResponsiveBaseProps {
  mobile: ReactNode;
  tablet?: ReactNode;
  desktop?: ReactNode;
  tabletBreakpoint?: number;
  desktopBreakpoint?: number;
}

export interface ResponsiveValueOptions<T> {
  mobile: T;
  tablet?: T;
  desktop?: T;
  tabletBreakpoint?: number;
  desktopBreakpoint?: number;
}

export interface SpacerBaseProps {
  /** Fixed extent in px/dp. When omitted, the spacer is flexible (expands). */
  size?: number;
}

export interface WrapBaseProps {
  gap?: number;
  runGap?: number;
  align?: Align;
  children?: ReactNode;
}

export interface ScrollBaseProps {
  horizontal?: boolean;
  gap?: number;
  align?: Align;
  children?: ReactNode;
}

export interface ListBaseProps<T> {
  data: readonly T[];
  renderItem: (item: T, index: number) => ReactNode;
  /** Gap between items. */
  gap?: number;
  horizontal?: boolean;
  keyExtractor?: (item: T, index: number) => string;
}

export interface FillBaseProps {
  children?: ReactNode;
}

export interface EntranceProps {
  children?: ReactNode;
  duration?: number;
  delay?: number;
}

export interface SlideInBaseProps extends EntranceProps {
  /** Starting X offset in px/dp (animates to 0). */
  x?: number;
  /** Starting Y offset in px/dp (animates to 0). */
  y?: number;
  fade?: boolean;
}

export interface ScaleInBaseProps extends EntranceProps {
  /** Starting scale factor (animates to 1). */
  from?: number;
  fade?: boolean;
}

export interface AnimatedVisibilityBaseProps {
  visible: boolean;
  children?: ReactNode;
  duration?: number;
}

export type StaggerEffect = "fade" | "slideUp" | "scale";

export interface StaggerBaseProps {
  effect?: StaggerEffect;
  /** Extra delay added per child. */
  interval?: number;
  /** Duration of each child's entrance. */
  duration?: number;
  gap?: number;
  children?: ReactNode;
}

/** Resolve a value by viewport width against the given breakpoints. */
export function pickResponsive<T>(width: number, opts: ResponsiveValueOptions<T>): T {
  const tablet = opts.tabletBreakpoint ?? 600;
  const desktop = opts.desktopBreakpoint ?? 1024;
  if (width >= desktop) return opts.desktop ?? opts.tablet ?? opts.mobile;
  if (width >= tablet) return opts.tablet ?? opts.mobile;
  return opts.mobile;
}
