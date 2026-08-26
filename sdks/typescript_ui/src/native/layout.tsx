import { Children, useState } from "react";
import {
  type LayoutChangeEvent,
  ScrollView,
  type StyleProp,
  View,
  type ViewStyle,
} from "react-native";

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

type Styled<P> = P & { style?: StyleProp<ViewStyle> };

/** Vertical stack with an even `gap` (RN >= 0.71). */
export function VStack({ gap = 0, align, justify, style, children }: Styled<StackBaseProps>) {
  return (
    <View style={[{ flexDirection: "column", gap, alignItems: toAlign(align), justifyContent: toJustify(justify) }, style]}>
      {children}
    </View>
  );
}

/** Horizontal stack with an even `gap`. */
export function HStack({ gap = 0, align = "center", justify, style, children }: Styled<StackBaseProps>) {
  return (
    <View style={[{ flexDirection: "row", gap, alignItems: toAlign(align), justifyContent: toJustify(justify) }, style]}>
      {children}
    </View>
  );
}

/**
 * Fixed-column grid. RN has no CSS grid, so this measures its width on layout
 * and sizes each cell to `(width - gaps) / columns`.
 */
export function Grid({ columns, gap = 0, runGap, style, children }: Styled<GridBaseProps>) {
  const [width, setWidth] = useState(0);
  const items = Children.toArray(children);
  const itemWidth = width > 0 ? (width - gap * (columns - 1)) / columns : 0;
  return (
    <View
      onLayout={(e: LayoutChangeEvent) => setWidth(e.nativeEvent.layout.width)}
      style={[{ flexDirection: "row", flexWrap: "wrap", columnGap: gap, rowGap: runGap ?? gap }, style]}
    >
      {items.map((child, i) => (
        <View key={i} style={{ width: itemWidth }}>
          {child}
        </View>
      ))}
    </View>
  );
}

/** Flexible (expands) or fixed-`size` space. */
export function Spacer({ size, style }: Styled<SpacerBaseProps>) {
  if (size != null) return <View style={[{ width: size, height: size }, style]} />;
  return <View style={[{ flex: 1 }, style]} />;
}

/** Flow layout that wraps to the next line. */
export function Wrap({ gap = 0, runGap, align, style, children }: Styled<WrapBaseProps>) {
  return (
    <View style={[{ flexDirection: "row", flexWrap: "wrap", columnGap: gap, rowGap: runGap ?? gap, alignItems: toAlign(align) }, style]}>
      {children}
    </View>
  );
}

/** A `ScrollView` whose content is a gap-stack of children. */
export function Scroll({ horizontal, gap = 0, align, style, children }: Styled<ScrollBaseProps>) {
  return (
    <ScrollView
      horizontal={horizontal}
      style={style}
      contentContainerStyle={{ flexDirection: horizontal ? "row" : "column", gap, alignItems: toAlign(align) }}
    >
      {children}
    </ScrollView>
  );
}

/** Fills its flex parent and centers `children` — for empty/error/loading states. */
export function Fill({ children, style }: Styled<FillBaseProps>) {
  return <View style={[{ flex: 1, alignItems: "center", justifyContent: "center" }, style]}>{children}</View>;
}
