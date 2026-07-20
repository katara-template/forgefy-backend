import { FlatList, type StyleProp, View, type ViewStyle } from "react-native";

import type { ListBaseProps } from "../shared";

/**
 * A lazily-rendered list backed by `FlatList` (the RN analog of a lazy
 * sliver list). `gap` is applied via an item separator.
 */
export function List<T>({
  data,
  renderItem,
  gap = 0,
  horizontal,
  keyExtractor,
  style,
}: ListBaseProps<T> & { style?: StyleProp<ViewStyle> }) {
  return (
    <FlatList
      data={data}
      horizontal={horizontal}
      style={style}
      keyExtractor={(item, index) => (keyExtractor ? keyExtractor(item, index) : String(index))}
      renderItem={({ item, index }) => <>{renderItem(item, index)}</>}
      ItemSeparatorComponent={
        gap > 0
          ? () => <View style={{ width: horizontal ? gap : 0, height: horizontal ? 0 : gap }} />
          : undefined
      }
    />
  );
}
