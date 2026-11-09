/** KDA: every earlier key contributes, scaled by the cumulative per-channel decay ∏α, so weight fades with distance from the diagonal. */
export function KdaDecayGradient({ id }: { id: string }) {
  return (
    <linearGradient id={id} gradientUnits="objectBoundingBox" x1="0.5" y1="0.5" x2="0" y2="1">
      <stop offset="0" className="kda-stop" stopOpacity={1} />
      <stop offset="0.25" className="kda-stop" stopOpacity={0.45} />
      <stop offset="0.5" className="kda-stop" stopOpacity={0.22} />
      <stop offset="1" className="kda-stop" stopOpacity={0.08} />
    </linearGradient>
  )
}
