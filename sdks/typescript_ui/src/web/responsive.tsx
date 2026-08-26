import { type ReactNode, useEffect, useState } from "react";

import { pickResponsive, type ResponsiveBaseProps, type ResponsiveValueOptions } from "../shared";

/**
 * Current viewport width. SSR- and hydration-safe: starts at 0 (→ mobile) so
 * the first client render matches the server, then updates after mount.
 */
export function useViewportWidth(): number {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const onResize = () => setWidth(window.innerWidth);
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  return width;
}

/** Resolve a value by breakpoint — handy for column counts, spacing, etc. */
export function useResponsiveValue<T>(opts: ResponsiveValueOptions<T>): T {
  return pickResponsive(useViewportWidth(), opts);
}

/** Render a different node per breakpoint. */
export function Responsive(props: ResponsiveBaseProps): ReactNode {
  const node = useResponsiveValue({
    mobile: props.mobile,
    tablet: props.tablet,
    desktop: props.desktop,
    tabletBreakpoint: props.tabletBreakpoint,
    desktopBreakpoint: props.desktopBreakpoint,
  });
  return <>{node}</>;
}
