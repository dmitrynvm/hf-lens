import { useLayoutEffect, useRef, useState } from 'react'
import type { GptOssConfig } from '../model/config'
import type { LayerRoute } from '../model/routing'

interface Props {
  config: GptOssConfig
  /** Simulated, seeded router choices per layer. */
  routes: LayerRoute[]
  selectedLayer: number
  flowStep: number | null
}

/** Square cells with a 1px gap, sized so the grid fills the panel width (no scrolling); the pitch is capped so few-expert configs stay compact. */
const GAP = 1
const MIN_PITCH = 2
const MAX_PITCH = 12
const LABEL_W = 35
const TOP = 16
const MIN_ROW = 11 // a 9px row label fits

export function ExpertRouting({ config: c, routes, selectedLayer, flowStep }: Props) {
  const L = c.num_hidden_layers
  const E = c.num_local_experts
  const k = c.num_experts_per_tok
  const frame = useRef<HTMLDivElement>(null)
  const [avail, setAvail] = useState(0)
  useLayoutEffect(() => {
    const el = frame.current!
    const measure = () => setAvail(el.clientWidth)
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  const PITCH = Math.min(MAX_PITCH, Math.max(MIN_PITCH, (avail - LABEL_W + GAP) / E))
  const CELL = PITCH - GAP
  // Rows get their own pitch so every layer's label fits; cells grow taller than wide when the columns are narrow.
  const ROW = Math.max(PITCH, MIN_ROW)
  const CELL_H = ROW - GAP
  const gridW = E * PITCH - GAP
  const width = LABEL_W + gridW
  const height = TOP + L * ROW
  const active = L * k
  const total = L * E
  return (
    <section className="panel routing" aria-label="Expert routing, simulated">
      <h2 className="section-title">Expert routing for one token <span className="section-count">simulated · {active} of {total.toLocaleString('en-US')} experts ({((active / total) * 100).toFixed(1)}%)</span></h2>
      <div className="routing-frame" ref={frame}>
        <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} className="routing-svg" role="img"
          aria-label={`Top-${k} expert choices (simulated) for each of ${L} layers`}>
          <title>{`Each row is one layer's own ${E} experts; brightness = weight relative to the layer's top expert`}</title>
          <defs>
            <pattern id="routing-cells" width={PITCH} height={ROW} patternUnits="userSpaceOnUse">
              <rect width={CELL} height={CELL_H} className="routing-bg" />
            </pattern>
          </defs>
          <text x={LABEL_W} y={10} className="t">expert 0</text>
          <text x={width} y={10} className="t" textAnchor="end">{E - 1}</text>
          <g transform={`translate(0,${TOP})`}>
            {routes.map((r, i) => {
              const passed = flowStep !== null && flowStep >= i
              const rowClass = `routing-row${i === selectedLayer ? ' selected' : ''}${flowStep === i ? ' current' : ''}`
              const y = i * ROW
              return (
                <g key={i} className={rowClass}>
                  <text x={LABEL_W - 5} y={y + CELL_H - 1} className="t" textAnchor="end">L{i}</text>
                  <rect x={LABEL_W - 1} y={y - 1} width={gridW + 2} height={ROW + 1} className="routing-band" />
                  <rect x={LABEL_W} y={y} width={gridW} height={CELL_H} fill="url(#routing-cells)" />
                  {r.experts.map((e, j) => (
                    <rect
                      key={e}
                      x={LABEL_W + e * PITCH}
                      y={y}
                      width={CELL}
                      height={CELL_H}
                      className="routing-cell"
                      style={{ opacity: (passed || flowStep === null ? 1 : 0.4) * (0.35 + 0.65 * r.weights[j] / r.weights[0]) }}
                    >
                      <title>layer {i} · expert {e} · w={r.weights[j].toFixed(3)}</title>
                    </rect>
                  ))}
                </g>
              )
            })}
          </g>
        </svg>
      </div>
    </section>
  )
}
