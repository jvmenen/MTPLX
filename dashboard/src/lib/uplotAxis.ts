import type uPlot from "uplot";

type SizedAxis = uPlot.Axis & {
  _size?: number;
  font?: string | [string, number, number];
};

/**
 * uPlot sizes an axis at a fixed 50px by default, which clips wider tick
 * labels such as "52 tok/s" or "1250". Size the axis to its longest label
 * instead (uPlot's axis-autosize recipe).
 */
export function autoAxisSize(
  self: uPlot,
  values: string[] | null,
  axisIdx: number,
  cycleNum: number,
): number {
  const axis = self.axes[axisIdx] as SizedAxis;
  // uPlot re-runs sizing until it converges; keep the first result.
  if (cycleNum > 1) return axis._size ?? 50;
  let size = (axis.ticks?.size ?? 10) + (axis.gap ?? 5);
  const longest = (values ?? []).reduce(
    (acc, value) => (value.length > acc.length ? value : acc),
    "",
  );
  if (longest !== "") {
    const font = Array.isArray(axis.font) ? axis.font[0] : axis.font;
    if (font) self.ctx.font = font;
    size += self.ctx.measureText(longest).width / devicePixelRatio;
  }
  return Math.ceil(size);
}
