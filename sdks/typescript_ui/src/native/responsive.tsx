import type { ReactNode } from "react";
import { useWindowDimensions } from "react-native";

import { pickResponsive, type ResponsiveBaseProps, type ResponsiveValueOptions } from "../shared";

/** Resolve a value by screen width — handy for column counts, spacing, etc. */
export function useResponsiveValue<T>(opts: ResponsiveValueOptions<T>): T {
  const { width } = useWindowDimensions();
  return pickResponsive(width, opts);
}

/** Render a different node per screen width. */
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
