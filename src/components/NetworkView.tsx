import { memo, useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import type { GptOssConfig } from '../model/config'
import type { Dtype } from '../model/params'
import {
  BRACKET_Y, BY, CAP_PITCH, CY, DX, DY, GROUP_BG_BOTTOM, groupForStep, JUNCTION_R, LABEL_Y, networkLayout,
  SUBLABEL_Y,
  type BlockSquares, type Branch, type ExpertGrid, type Group, type MaskSquare, type Part, type Slab, type TapeRow, type Target,
} from '../model/network'
import type { LayerRoute } from '../model/routing'
import { KdaDecayGradient } from './KdaDecay'

/** Scroll animation when the focused group changes, however far it moves. */
const SCROLL_MS = 250

interface Props {
  config: GptOssConfig
  modelName: string
  routes: LayerRoute[]
  /** The selected layer or end; the diagram scrolls to it. */
  selected: Target
  flowStep: number | null
  /** Playback step duration; the squares' fill animation spans one step. */
  stepMs: number
  onSelect: (target: Target) => void
  /** Storage dtype picked in the sidebar: dtype captions of tensors stored in it turn red; null marks nothing. */
  highlight: Dtype | null
}

/** Oblique box: front face, lighter top, darker side, dashed hidden back edges. */
function SlabBox({ s, scale = 1 }: { s: Slab; scale?: number }) {
  const dx = DX * scale
  const dy = DY * scale
  const { x, t } = s
  const h = s.h
  const fy = s.y - h / 2 + dy / 2
  return (
    <g className={`slab ${s.kind}`}>
      <title>{s.title}</title>
      <polyline className="hidden-edge" fill="none"
        points={`${x},${fy + h} ${x + dx},${fy + h - dy} ${x + dx},${fy - dy} ${x + dx},${fy + h - dy} ${x + t + dx},${fy + h - dy}`} />
      <rect className="face front" x={x} y={fy} width={t} height={h} />
      <polygon className="face top" points={`${x},${fy} ${x + t},${fy} ${x + t + dx},${fy - dy} ${x + dx},${fy - dy}`} />
      <polygon className="face side" points={`${x + t},${fy} ${x + t + dx},${fy - dy} ${x + t + dx},${fy + h - dy} ${x + t},${fy + h}`} />
    </g>
  )
}

/** The dtype caption of a part whose weights are stored in the picked type; weightless ops (SwiGLU, sum, softmax) never are. */
const picked = (p: Part, line: string, highlight: Dtype | null) => line === highlight && (p.weights?.[highlight] ?? 0) > 0

function PartView({ p, y, prevRight, highlight }: { p: Part; y: number; prevRight: number; highlight: Dtype | null }) {
  // Parallel slabs stack into the page: draw the farthest first so the front one covers them.
  const slabs = p.parallel ? [...p.slabs].reverse() : p.slabs
  const capX = (p.x0 + p.x1) / 2
  return (
    <g>
      <line className="net-connector" x1={prevRight} y1={y} x2={p.x0} y2={y} />
      {slabs.map((s, i) => <SlabBox key={i} s={s} />)}
      {p.op && <text className={`net-op${p.op === 'SWA' || p.op === 'KDA' ? ' sliding' : ''}`} x={p.opX} y={p.opY} textAnchor="middle">{p.op}</text>}
      {p.captions.map((line, i) => {
        const isDtype = line === 'int64' || line === 'bf16' || line === 'mxfp4'
        return (
          <text key={i} className={`net-cap${isDtype ? ` dtype ${line}` : ''}${picked(p, line, highlight) ? ' picked' : ''}`} x={capX} y={p.captionY + i * CAP_PITCH} textAnchor="middle">{line}</text>
        )
      })}
    </g>
  )
}

function BranchView({ b, highlight }: { b: Branch; highlight: Dtype | null }) {
  const first = b.parts[0]
  const last = b.parts[b.parts.length - 1]
  return (
    <g className="net-branch">
      <circle className="net-tap" cx={b.tapX} cy={CY} r={3} />
      <polyline className="net-wire" fill="none" markerEnd="url(#net-arrow)" points={`${b.tapX},${CY} ${b.tapX},${BY} ${first.x0 - 1},${BY}`} />
      {b.parts.map((p, i) => (
        <PartView key={i} p={p} y={BY} prevRight={i === 0 ? b.tapX : b.parts[i - 1].frontX1} highlight={highlight} />
      ))}
      <polyline className="net-wire" fill="none" markerEnd="url(#net-arrow)" points={`${last.frontX1},${BY} ${b.joinX},${BY} ${b.joinX},${CY - JUNCTION_R - 1}`} />
      <g className="junction">
        <circle cx={b.joinX} cy={CY} r={JUNCTION_R} />
        <path d={`M${b.joinX - 5},${CY} h10 M${b.joinX},${CY - 5} v10`} />
      </g>
    </g>
  )
}

/** Linear time 0 → 1 over one flow step (`ms`) while `active` (the token is in this block), 1 otherwise. */
function useStepTime(active: boolean, ms: number): number {
  const [p, setP] = useState(active ? 0 : 1)
  const [wasActive, setWasActive] = useState(active)
  if (active !== wasActive) {
    setWasActive(active)
    setP(active ? 0 : 1)
  }
  // Read in the frame loop, so moving the speed slider mid-step changes the pace without restarting the fill.
  const msRef = useRef(ms)
  useEffect(() => void (msRef.current = ms), [ms])
  useEffect(() => {
    if (!active) return
    let last = performance.now()
    let k = 0
    let id = requestAnimationFrame(function tick(t) {
      k = Math.min(1, k + Math.max(0, t - last) / msRef.current) // a frame's timestamp can precede `last`
      last = t
      setP(k)
      if (k < 1) id = requestAnimationFrame(tick)
    })
    return () => cancelAnimationFrame(id)
  }, [active])
  return p
}

const clamp01 = (x: number) => Math.min(1, Math.max(0, x))
const easeInOut = (x: number) => (x < 0.5 ? 4 * x ** 3 : 1 - (-2 * x + 2) ** 3 / 2)
/** Overshoots slightly before settling at 1: a lit cell pops in. */
const easeOutBack = (x: number) => 1 + 2.7 * (x - 1) ** 3 + 1.7 * (x - 1) ** 2

/** One step's timeline, attention before MoE as in the block: the mask fills over the first half, then the routed experts light up one after another over the second, heaviest first. */
const MASK_END = 0.5
const EXPERTS_START = 0.5
const EXPERTS_STAGGER = 0.3
const EXPERT_FADE = 0.2
const maskFill = (t: number) => easeInOut(clamp01(t / MASK_END))
const expertIn = (t: number, j: number, k: number) => clamp01((t - EXPERTS_START - (EXPERTS_STAGGER * j) / Math.max(1, k - 1)) / EXPERT_FADE)

/** The attention mask (sink bar on the left), clipped to the top `fill` share of its square with a scan line at the edge while it moves; the caption counts the keys revealed so far. */
function MaskView({ m, layer, fill, captionY }: { m: MaskSquare; layer: number; fill: number; captionY: number }) {
  const clip = `net-mask-clip-${layer}`
  return (
    <>
      <g className={`net-mask ${m.kind}`} transform={`translate(${m.x},${m.y})`}>
        <title>{m.title}</title>
        <clipPath id={clip}><rect x={-6} y={0} width={m.side + 6} height={fill * m.side} /></clipPath>
        <rect className="net-mask-bg" width={m.side} height={m.side} />
        <g clipPath={`url(#${clip})`}>
          <polygon className="net-mask-on" points={m.points} fill={m.kind === 'linear' ? 'url(#net-kda-decay)' : undefined} />
          {m.kind !== 'linear' && <rect className="net-mask-sink" x={-6} width={3} height={m.side} />}
        </g>
        {fill > 0 && fill < 1 && <rect className="net-mask-scan" x={-6} y={fill * m.side - 0.75} width={m.side + 6} height={1.5} />}
      </g>
      <text className="net-cap" x={m.x + m.side / 2} y={captionY} textAnchor="middle">{`${Math.min(m.keys, Math.round(fill * m.shown))}/${m.shown}`}</text>
    </>
  )
}

/** Cell `e` of an expert grid, row-major, square and centred in its pitch with half a gap on every side. */
function cellAt(gr: ExpertGrid, e: number) {
  const s = gr.pitch - gr.gap
  return { x: gr.gap / 2 + (e % gr.cols) * gr.pitch, y: gr.gap / 2 + Math.floor(e / gr.cols) * gr.pitch, rx: s * 0.2, width: s, height: s }
}

/** A routed expert lighting up: `a` 0 → 1 fades it in with a small pop and an expanding ring. */
function LitExpert({ gr, e, weight, top, a }: { gr: ExpertGrid; e: number; weight: number; top: number; a: number }) {
  const r = cellAt(gr, e)
  const cx = r.x + r.width / 2
  const cy = r.y + r.height / 2
  const s = 0.4 + 0.6 * easeOutBack(a)
  const ring = 1 + 1.4 * a
  return (
    <>
      {a < 1 && (
        <rect className="net-expert-ring" x={cx - (r.width * ring) / 2} y={cy - (r.height * ring) / 2} width={r.width * ring} height={r.height * ring}
          rx={r.rx * ring} style={{ opacity: 1 - a }} />
      )}
      <rect className="net-expert on" {...r} transform={a < 1 ? `translate(${cx},${cy}) scale(${s}) translate(${-cx},${-cy})` : undefined}
        style={{ opacity: Math.min(1, a * 1.5) * (0.35 + (0.65 * weight) / top) }}>
        <title>{`expert ${e} · w=${weight.toFixed(3)}`}</title>
      </rect>
    </>
  )
}

/** Under a block: its attention mask and, beside it, its expert grid in a frame that hugs the cells. Both stay empty while idle and until the token arrives; over that step the mask fills top-down, then the routed experts light up one by one, heaviest first. */
function BlockSquaresView({ s, layer, route, reached, current, stepMs }: {
  s: BlockSquares; layer: number; route?: LayerRoute; reached: boolean; current: boolean; stepMs: number
}) {
  const gr = s.moe.grid
  const time = useStepTime(current, stepMs)
  const t = reached ? time : 0
  // The dim cells don't change while the squares animate; build them once per grid, not every frame.
  const cells = useMemo(() => gr && Array.from({ length: gr.cols * gr.rows }, (_, e) => (
    <rect key={e} className={e < gr.experts ? 'net-expert' : 'net-expert pad'} {...cellAt(gr, e)} />
  )), [gr])
  const k = route?.experts.length ?? 0
  const lit = route ? route.experts.filter((_, j) => expertIn(t, j, k) >= 0.5).length : 0
  return (
    <g className="net-squares">
      <MaskView m={s.mask} layer={layer} fill={maskFill(t)} captionY={s.captionY} />
      {gr && (
        <g className="net-experts" transform={`translate(${gr.x},${gr.y})`}>
          <title>{`${s.moe.title}: ${route ? route.experts.join(' · ') : ''}`}</title>
          <rect className="net-mask-bg" width={gr.width} height={gr.height} />
          {cells}
          {route?.experts.map((e, j) => {
            const a = expertIn(t, j, k)
            return a > 0 && <LitExpert key={e} gr={gr} e={e} weight={route.weights[j]} top={route.weights[0]} a={a} />
          })}
        </g>
      )}
      <g>
        {!gr && <title>{s.moe.title}</title>}
        <text className="net-cap" x={s.moe.cx} y={s.captionY} textAnchor="middle">{gr && route ? `${lit}/${gr.experts}` : !gr || reached ? s.moe.caption : `0/${gr.experts}`}</text>
      </g>
    </g>
  )
}

/** Memoized: a flow step or selection change re-renders only the groups whose flags flip, not every expert cell of every block. */
const GroupView = memo(function GroupView({ g, current, selected, route, reached, stepMs, onSelect, highlight }: {
  g: Group
  current: boolean
  selected: boolean
  route?: LayerRoute
  /** The token has reached this block: the mask is drawn and the routed experts lit. False while idle. */
  reached: boolean
  stepMs: number
  onSelect: (target: Target) => void
  highlight: Dtype | null
}) {
  const target: Target = g.kind === 'embed' || g.kind === 'head' ? g.kind : g.layers[0]
  const onKeyDown = (e: KeyboardEvent<SVGGElement>) => {
    if (e.key !== 'Enter' && e.key !== ' ') return
    e.preventDefault()
    onSelect(target)
  }
  const mid = (g.x0 + g.x1) / 2
  return (
    <g
      className={`net-group ${g.kind}${current ? ' now' : ''}${selected ? ' selected' : ''} clickable`}
      role="button"
      tabIndex={0}
      aria-label={`${g.label}${g.attention ? ` (${g.attention === 'sliding' ? 'SWA' : 'FA'})` : ''}: select ${typeof target === 'number' ? `layer ${target}` : g.label}`}
      aria-pressed={selected}
      onClick={() => onSelect(target)}
      onKeyDown={onKeyDown}
    >
      <rect className="net-bg" x={g.x0 - 4} y={2} width={g.x1 - g.x0 + 8} height={GROUP_BG_BOTTOM - 2} rx={2} />
      {g.branches.length > 0 && (
        <line
          className="net-group-stream"
          x1={g.branches[0].tapX}
          y1={CY}
          x2={g.branches[g.branches.length - 1].joinX}
          y2={CY}
        />
      )}
      {g.parts.map((p, i) => (
        <PartView key={i} p={p} y={CY} prevRight={i === 0 ? p.x0 : g.parts[i - 1].frontX1} highlight={highlight} />
      ))}
      {g.branches.map((b, i) => <BranchView key={i} b={b} highlight={highlight} />)}
      {g.squares && <BlockSquaresView s={g.squares} layer={g.layers[0]} route={route} reached={reached} current={current} stepMs={stepMs} />}
      <path className="net-bracket" fill="none" d={`M${g.x0},${BRACKET_Y + 6} v-6 H${g.x1} v6`} />
      <polygon className="net-pointer" points={`${mid - 6},${BRACKET_Y} ${mid + 6},${BRACKET_Y} ${mid},${BRACKET_Y + 7}`} />
      <text className="net-label" x={mid} y={LABEL_Y} textAnchor="middle">{g.label}</text>
      {g.sublabel && (
        <text className={`net-sublabel${g.attention === 'sliding' ? ' sliding' : ''}`} x={mid} y={SUBLABEL_Y} textAnchor="middle">{g.sublabel}</text>
      )}
    </g>
  )
})

/** The block weight tape: one bar per group coloured by attention type (as the layer minimap), separators and label. */
function TapeRowView({ row }: { row: TapeRow }) {
  return (
    <g className="net-tape" aria-label="Block weights">
      {row.segments.map((s, i) => {
        const isBlockEdge = i > 0 && row.segments[i - 1].groupId !== s.groupId
        return (
          <g key={s.id}>
            <title>{s.title}</title>
            <rect className={`net-bar ${s.kind}`} x={s.x0} y={row.barY0} width={s.x1 - s.x0} height={row.barY1 - row.barY0} />
            {i > 0 && (
              <line
                className={`net-bar-sep${isBlockEdge ? ' block-sep' : ''}`}
                x1={s.x0}
                y1={isBlockEdge ? row.barY0 - 2 : row.barY0}
                x2={s.x0}
                y2={isBlockEdge ? row.barY1 + 2 : row.barY1}
              />
            )}
            <text className="net-cap net-tape-label" x={s.labelX} y={row.labelY} textAnchor="middle" dominantBaseline="central">{s.label}</text>
          </g>
        )
      })}
    </g>
  )
}

export function NetworkView({ config: c, modelName, routes, selected, flowStep, stepMs, onSelect, highlight }: Props) {
  const L = c.num_hidden_layers
  const layout = useMemo(() => networkLayout(c), [c])
  const now = flowStep === null ? undefined : groupForStep(layout.groups, flowStep, L)
  const scroller = useRef<HTMLDivElement>(null)
  const prevLayoutRef = useRef<typeof layout | null>(null)
  const scrollAnim = useRef(0)

  // Center the selected group (which also follows the token while it flows), whether chosen here or in the sidebar.
  const focus = layout.groups.find((g) => (typeof selected === 'number' ? g.layers.includes(selected) : g.kind === selected))
  const focusId = focus?.id
  useEffect(() => {
    const el = scroller.current
    const g = layout.groups.find((x) => x.id === focusId)
    if (!el || !g) return
    const scale = el.scrollWidth / layout.width
    const [a, b] = [(g.x0 - 8) * scale, (g.x1 + 8) * scale]
    const maxScroll = Math.max(0, el.scrollWidth - el.clientWidth)
    const target = Math.max(0, Math.min(maxScroll, (a + b) / 2 - el.clientWidth / 2))

    const isInitialOrNewLayout = prevLayoutRef.current !== layout
    prevLayoutRef.current = layout

    if (isInitialOrNewLayout) {
      el.scrollTo({ left: target, behavior: 'instant' })
      return
    }
    // Fixed duration whatever the distance: native smooth scroll stretches with distance, so the wrap from lm_head back to the embedding crawled.
    const from = el.scrollLeft
    const t0 = performance.now()
    const tick = (t: number) => {
      const k = Math.min(1, (t - t0) / SCROLL_MS)
      el.scrollLeft = from + (target - from) * (1 - (1 - k) ** 3)
      if (k < 1) scrollAnim.current = requestAnimationFrame(tick)
    }
    scrollAnim.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(scrollAnim.current)
  }, [focusId, layout])

  return (
    <section className="panel network" aria-label="Architecture">
      <div className="table-head">
        <h2 className="section-title">Architecture <span className="section-count">{L} blocks</span></h2>
      </div>
      <div className="diagram-scroll" ref={scroller}>
        <svg viewBox={`0 0 ${layout.width} ${layout.height}`} className="net-svg" role="group" style={{ minWidth: layout.width }}
          aria-label={`${modelName}: input embedding, ${L} decoder blocks (attention + MoE each), output norm, lm_head and softmax`}>
          <defs>
            <marker id="net-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" className="net-arrowhead" />
            </marker>
            <marker id="net-arrow-stream" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" className="net-arrowhead-stream" />
            </marker>
            <KdaDecayGradient id="net-kda-decay" />
          </defs>
          <line className="net-stream" x1={layout.stream[0]} y1={CY} x2={layout.stream[1]} y2={CY} markerEnd="url(#net-arrow-stream)" />
          <text className="net-stream-label" x={layout.stream[0] + DX + 8} y={CY + 20}>{layout.streamLabel}</text>
          {layout.groups.map((g) => {
            const isSelected = typeof selected === 'number' ? g.layers.includes(selected) : g.kind === selected
            return (
              <GroupView
                key={g.id}
                g={g}
                current={now === g}
                selected={isSelected}
                route={g.kind === 'block' ? routes[g.layers[0]] : undefined}
                reached={flowStep !== null && g.kind === 'block' && flowStep >= g.layers[0]}
                stepMs={stepMs}
                onSelect={onSelect}
                highlight={highlight}
              />
            )
          })}
          <TapeRowView row={layout.blockTape} />
        </svg>
      </div>
    </section>
  )
}
