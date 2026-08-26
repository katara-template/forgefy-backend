import { Children, type ReactNode, useEffect, useRef, useState } from "react";
import { Animated, Easing, type LayoutChangeEvent, type StyleProp, View, type ViewStyle } from "react-native";

import type {
  AnimatedVisibilityBaseProps,
  EntranceProps,
  ScaleInBaseProps,
  SlideInBaseProps,
  StaggerBaseProps,
  StaggerEffect,
} from "../shared";

/** A 0→1 value that runs once on mount after `delay`. */
function useEntranceValue(duration: number, delay: number): Animated.Value {
  const value = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    const anim = Animated.timing(value, {
      toValue: 1,
      duration,
      delay,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    });
    anim.start();
    return () => anim.stop();
  }, [value, duration, delay]);
  return value;
}

/** Fade a child in on mount. */
export function FadeIn({ children, duration = 300, delay = 0 }: EntranceProps) {
  const t = useEntranceValue(duration, delay);
  return <Animated.View style={{ opacity: t }}>{children}</Animated.View>;
}

/** Slide (and, by default, fade) a child in from an `x`/`y` offset. */
export function SlideIn({ children, duration = 300, delay = 0, x = 0, y = 24, fade = true }: SlideInBaseProps) {
  const t = useEntranceValue(duration, delay);
  const translateX = t.interpolate({ inputRange: [0, 1], outputRange: [x, 0] });
  const translateY = t.interpolate({ inputRange: [0, 1], outputRange: [y, 0] });
  return (
    <Animated.View style={{ opacity: fade ? t : 1, transform: [{ translateX }, { translateY }] }}>
      {children}
    </Animated.View>
  );
}

/** Scale (and, by default, fade) a child in from `from` to 1. */
export function ScaleIn({ children, duration = 300, delay = 0, from = 0.95, fade = true }: ScaleInBaseProps) {
  const t = useEntranceValue(duration, delay);
  const scale = t.interpolate({ inputRange: [0, 1], outputRange: [from, 1] });
  return <Animated.View style={{ opacity: fade ? t : 1, transform: [{ scale }] }}>{children}</Animated.View>;
}

/**
 * Animated show/hide with a height collapse. Measures the child's natural
 * height once, then animates height + opacity between 0 and it. Height can't use
 * the native driver, so this runs on the JS thread.
 */
export function AnimatedVisibility({ visible, children, duration = 250 }: AnimatedVisibilityBaseProps) {
  const [measured, setMeasured] = useState(0);
  const progress = useRef(new Animated.Value(visible ? 1 : 0)).current;

  useEffect(() => {
    Animated.timing(progress, {
      toValue: visible ? 1 : 0,
      duration,
      easing: Easing.inOut(Easing.ease),
      useNativeDriver: false,
    }).start();
  }, [visible, duration, progress]);

  const height =
    measured === 0 ? undefined : progress.interpolate({ inputRange: [0, 1], outputRange: [0, measured] });

  return (
    <Animated.View style={{ height, opacity: progress, overflow: "hidden" }}>
      <View
        onLayout={(e: LayoutChangeEvent) => {
          const h = e.nativeEvent.layout.height;
          if (h > 0 && h !== measured) setMeasured(h);
        }}
      >
        {children}
      </View>
    </Animated.View>
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
}: StaggerBaseProps & { style?: StyleProp<ViewStyle> }) {
  const items = Children.toArray(children);
  return (
    <View style={[{ flexDirection: "column", gap }, style]}>
      {items.map((child, i) => (
        <View key={i}>{entrance(effect, child, interval * i, duration)}</View>
      ))}
    </View>
  );
}
