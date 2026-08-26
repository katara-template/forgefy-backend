import type { CSSProperties } from "react";

import type { ListBaseProps } from "../shared";

/**
 * A scrolling list. On web this maps every item (not windowed) — for very long
 * feeds, wrap a virtualization lib. API-compatible with the native `List`
 * (which is a `FlatList`).
 */
export function List<T>({
  data,
  renderItem,
  gap = 0,
  horizontal,
  keyExtractor,
  style,
}: ListBaseProps<T> & { style?: CSSProperties }) {
  return (
    <div
      style={{
        overflowX: horizontal ? "auto" : "hidden",
        overflowY: horizontal ? "hidden" : "auto",
        display: "flex",
        flexDirection: horizontal ? "row" : "column",
        gap,
        ...style,
      }}
    >
      {data.map((item, i) => (
        <div key={keyExtractor ? keyExtractor(item, i) : i}>{renderItem(item, i)}</div>
      ))}
    </div>
  );
}
