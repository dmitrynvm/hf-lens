import { useEffect, useMemo, useRef, type KeyboardEvent } from 'react'
import type { GptOssConfig } from '../model/config'
import {
  BAR_H, BAR_Y, BRACKET_Y, BY, CAP_PITCH, CY, DX, DY, GROUP_BG_BOTTOM, groupForStep, JUNCTION_R, LABEL_Y, networkLayout,
  SIZE_LABEL_Y, SUB_Y, SUBLAYER_Y, TAPE_KEY_Y,
  type Branch, type Group, type Part, type Slab, type Target, type WeightBar,
} from '../model/network'
import type { LayerRoute } from '../model/routing'

interface Props {
  config: GptOssConfig
  modelName: string
  routes: LayerRoute[]
  /** The selected layer or end; the diagram scrolls to it. */
  selected: Target
  flowStep: number | null
  onSelect: (target: Target) => void
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

/** `extra`: the routed expert ids, appended under the captions once the token has passed. */
function PartView({ p, y, prevRight, extra }: { p: Part; y: number; prevRight: number; extra: string | null }) {
  // Parallel slabs stack into the page: draw the farthest first so the front one covers them.
  const slabs = p.parallel ? [...p.slabs].reverse() : p.slabs
  const capX = (p.x0 + p.x1) / 2
  return (
    <g>
      <line className="net-connector" x1={prevRight} y1={y} x2={p.x0} y2={y} />
      {slabs.map((s, i) => <SlabBox key={i} s={s} />)}
      {p.op && <text className="net-op" x={p.opX} y={p.opY} textAnchor="middle">{p.op}</text>}
      {p.captions.map((line, i) => (
        <text key={i} className="net-cap" x={capX} y={p.captionY + i * CAP_PITCH} textAnchor="middle">{line}</text>
      ))}
      {extra && <text className="net-cap routed" x={capX} y={p.captionY + p.captions.length * CAP_PITCH} textAnchor="middle">{extra}</text>}
    </g>
  )
}

function BranchView({ b, route }: { b: Branch; route: LayerRoute | null }) {
  const first = b.parts[0]
  const last = b.parts[b.parts.length - 1]
  return (
    <g className="net-branch">
      <text className={`net-sublayer${b.attention ? ` ${b.attention}` : ''}`} x={(b.tapX + b.joinX) / 2} y={SUBLAYER_Y} textAnchor="middle">{b.label}</text>
      <circle className="net-tap" cx={b.tapX} cy={CY} r={3} />
      <polyline className="net-wire" fill="none" markerEnd="url(#net-arrow)" points={`${b.tapX},${CY} ${b.tapX},${BY} ${first.x0 - 1},${BY}`} />
      {b.parts.map((p, i) => (
        <PartView key={i} p={p} y={BY} prevRight={i === 0 ? b.tapX : b.parts[i - 1].frontX1}
          extra={p.experts && route ? `experts ${route.experts.join('·')}` : null} />
      ))}
      <polyline className="net-wire" fill="none" markerEnd="url(#net-arrow)" points={`${last.frontX1},${BY} ${b.joinX},${BY} ${b.joinX},${CY - JUNCTION_R - 1}`} />
      <g className="junction">
        <circle cx={b.joinX} cy={CY} r={JUNCTION_R} />
        <path d={`M${b.joinX - 6},${CY} h12 M${b.joinX},${CY - 6} v12`} />
      </g>
    </g>
  )
}

/** What clicking a group selects: the ends select themselves, a block its layer. */
const targetOf = (g: Group): Target => (g.kind === 'embed' || g.kind === 'head' ? g.kind : g.layers[0])

function GroupView({ g, current, selected, passed, route, onSelect }: {
  g: Group
  current: boolean
  selected: boolean
  passed: boolean
  route: LayerRoute | null
  onSelect: (target: Target) => void
}) {
  const target = targetOf(g)
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
      aria-label={`${g.label}${g.attention ? ` ${g.attention} attention,` : ''} ${g.sub}: select ${typeof target === 'number' ? `layer ${target}` : g.label}`}
      aria-pressed={selected}
      onClick={() => onSelect(target)}
      onKeyDown={onKeyDown}
    >
      <rect className="net-bg" x={g.x0 - 8} y={2} width={g.x1 - g.x0 + 16} height={GROUP_BG_BOTTOM - 2} rx={2} />
      {g.parts.map((p, i) => (
        <PartView key={i} p={p} y={CY} prevRight={i === 0 ? p.x0 : g.parts[i - 1].frontX1} extra={null} />
      ))}
      {g.branches.map((b, i) => <BranchView key={i} b={b} route={passed ? route : null} />)}
      <path className="net-bracket" fill="none" d={`M${g.x0},${BRACKET_Y + 6} v-6 H${g.x1} v6`} />
      <text className="net-label" x={mid} y={LABEL_Y} textAnchor="middle">{g.label}</text>
      <text className="net-sub" x={mid} y={SUB_Y} textAnchor="middle">{g.sub}</text>
    </g>
  )
}

/** Names the slab encodings on the embedding slab itself: height = width d, thickness = params, depth = tokens T. */
function Anatomy({ s }: { s: Slab }) {
  const fy = s.y - s.h / 2 + DY / 2 // front face top, as in SlabBox
  const hx = s.x - 6 // height bar, left of the front face
  const ty = fy - DY - 6 // thickness bar, above the top face's back edge
  return (
    <g className="net-anno">
      <title>slab height ∝ log width d · thickness ∝ log params of the op that made it · depth = tokens T</title>
      <path d={`M${hx - 2},${fy} h4 M${hx},${fy} V${fy + s.h} M${hx - 2},${fy + s.h} h4`} />
      <text x={hx} y={fy + s.h + 14} textAnchor="middle">d</text>
      <path d={`M${s.x + DX},${ty - 2} v4 M${s.x + DX},${ty} H${s.x + DX + s.t} M${s.x + DX + s.t},${ty - 2} v4`} />
      <text x={s.x + DX + s.t / 2} y={ty - 6} textAnchor="middle">params</text>
      <text x={s.x + DX / 2 - 6} y={fy - DY - 28} textAnchor="end">T tokens</text>
    </g>
  )
}

/** What each tape segment is, written on it: the block's attention type, or in / out for the ends. */
const tapeType = (b: WeightBar) => (b.kind !== 'end' ? b.kind : b.groupId === 'embed' ? 'in' : 'out')

/** The weight tape: one segment per group in its tile colour, its type written on it and its size below. */
function TapeView({ bars }: { bars: WeightBar[] }) {
  return (
    <g className="net-weights" aria-label="Weight bytes per group">
      {bars.map((b) => (
        <g key={b.groupId}>
          <title>{b.title}</title>
          <rect className={`net-bar ${b.kind}`} x={b.bf16[0]} y={BAR_Y} width={b.mxfp4[1] - b.bf16[0]} height={BAR_H} />
          <text className="net-bar-type" x={b.labelX} y={BAR_Y + BAR_H / 2 + 4} textAnchor="middle">{tapeType(b)}</text>
          <text className="net-cap" x={b.labelX} y={SIZE_LABEL_Y} textAnchor="middle">{b.label}</text>
        </g>
      ))}
      <g className="net-tape-key">
        <text x={bars[0].bf16[0]} y={TAPE_KEY_Y}>width scales with weights · hover a segment for its breakdown</text>
      </g>
    </g>
  )
}

export function NetworkView({ config: c, modelName, routes, selected, flowStep, onSelect }: Props) {
  const L = c.num_hidden_layers
  const layout = useMemo(() => networkLayout(c), [c])
  const now = flowStep === null ? undefined : groupForStep(layout.groups, flowStep, L)
  const scroller = useRef<HTMLDivElement>(null)

  // Keep the token's group in view while it flows (ends included), otherwise the selection (from here or the sidebar).
  // Nearby moves glide; a jump longer than the viewport is instant, so the view never trails the token across steps.
  const focus = now ?? layout.groups.find((g) => (typeof selected === 'number' ? g.layers.includes(selected) : g.kind === selected))
  const focusId = focus?.id
  useEffect(() => {
    const el = scroller.current
    const g = layout.groups.find((g) => g.id === focusId)
    if (!el || !g) return
    const scale = el.scrollWidth / layout.width
    const [a, b] = [(g.x0 - 8) * scale, (g.x1 + 8) * scale]
    if (a >= el.scrollLeft && b <= el.scrollLeft + el.clientWidth) return
    const left = (a + b) / 2 - el.clientWidth / 2
    const far = Math.abs(left - el.scrollLeft) > el.clientWidth
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches
    el.scrollTo({ left, behavior: reduce || far ? 'auto' : 'smooth' })
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
            <marker id="net-arrow-stream" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="12" markerHeight="12" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" className="net-arrowhead-stream" />
            </marker>
          </defs>
          {/* The token's group is washed below the stream line, so the opaque wash never hides it. */}
          {now && <rect className="net-now" x={now.x0 - 8} y={2} width={now.x1 - now.x0 + 16} height={GROUP_BG_BOTTOM - 2} rx={2} />}
          <line className="net-stream" x1={layout.stream[0]} y1={CY} x2={layout.stream[1]} y2={CY} markerEnd="url(#net-arrow-stream)" />
          <text className="net-stream-label" x={layout.stream[0] + DX + 8} y={CY + 20}>{layout.streamLabel}</text>
          {layout.groups.map((g) => {
            const layer = g.kind === 'block' ? g.layers[0] : -1
            const isSelected = typeof selected === 'number' ? g.layers.includes(selected) : g.kind === selected
            return (
              <GroupView
                key={g.id}
                g={g}
                current={now === g}
                selected={isSelected}
                passed={layer >= 0 && flowStep !== null && flowStep >= layer}
                route={layer >= 0 ? routes[layer] : null}
                onSelect={onSelect}
              />
            )
          })}
          <Anatomy s={layout.groups[0].parts[1].slabs[0]} />
          <TapeView bars={layout.tape} />
        </svg>
      </div>
    </section>
  )
}
