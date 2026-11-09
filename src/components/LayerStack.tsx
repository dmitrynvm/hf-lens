import { useRef, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from 'react'
import type { GptOssConfig } from '../model/config'
import type { Target } from '../model/network'
import { formatBytes, layerMemory } from '../model/params'
import type { LayerRoute } from '../model/routing'

interface RodProps {
  config: GptOssConfig
  /** Selected bead: a decoder layer or one of the ends; the arrow keys step from it. */
  selected: Target
  onSelect: (target: Target) => void
  flowStep: number | null
}

interface DetailProps extends RodProps {
  routes: LayerRoute[]
}
/** Rod order: the embedding, the decoder layers, lm_head. A bead's flow step is its position - 1 (embedding -1, lm_head L). */
const positionOf = (t: Target, L: number) => (t === 'embed' ? 0 : t === 'head' ? L + 1 : t + 1)
const targetAt = (p: number, L: number): Target => (p === 0 ? 'embed' : p === L + 1 ? 'head' : p - 1)
const kindOf = (t: Target, c: GptOssConfig) =>
  t === 'embed' || t === 'head' ? 'end' : c.layer_types[t] === 'sliding_attention' || c.layer_types[t] === 'linear_attention' ? 'sliding' : 'full'
const idOf = (t: Target) => (t === 'embed' ? 'In' : t === 'head' ? 'Out' : `L${t}`)
const valueText = (t: Target, c: GptOssConfig) =>
  t === 'embed' ? 'In · embedding' : t === 'head' ? 'Out · lm_head' : `L${t} · ${kindOf(t, c)} attention`
/** Stored weight bytes of a layer or end, the same figures the detail panel shows. */
const weightsOf = (t: Target, c: GptOssConfig) =>
  t === 'embed' ? c.vocab_size * c.hidden_size * 2 : t === 'head' ? (c.vocab_size * c.hidden_size + c.hidden_size) * 2 : layerMemory(c, t).bytes

/** Keys that move along the rod, by how many beads; Home / End jump to the ends. */
const KEY_STEP: Record<string, number> = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1, PageDown: -6, PageUp: 6 }

/** The layers as beads on one rod, embedded in the main header bar: click, drag or use the arrow keys; the token's path is the played stretch. */
export function LayerRod({ config, selected, onSelect, flowStep }: RodProps) {
  const L = config.num_hidden_layers
  const N = L + 2
  const inner = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)
  const [hover, setHover] = useState<number | null>(null)
  const sel = positionOf(selected, L)
  const nowPos = flowStep === null ? null : Math.min(flowStep + 1, N - 1)
  const pct = (p: number) => `${((p + 0.5) / N) * 100}%`
  const posAt = (clientX: number) => {
    const r = inner.current!.getBoundingClientRect()
    return Math.min(N - 1, Math.max(0, Math.floor(((clientX - r.left) / r.width) * N)))
  }
  const pick = (p: number) => {
    if (p !== sel) onSelect(targetAt(p, L))
  }

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const next = e.key === 'Home' ? 0 : e.key === 'End' ? N - 1 : e.key in KEY_STEP ? sel + KEY_STEP[e.key] : null
    if (next === null) return
    e.preventDefault()
    pick(Math.min(N - 1, Math.max(0, next)))
  }

  const shown = hover ?? sel
  return (
    <div
      className="rod"
      role="slider"
      tabIndex={0}
      aria-label="Layer"
      aria-orientation="horizontal"
      aria-valuemin={0}
      aria-valuemax={N - 1}
      aria-valuenow={sel}
      aria-valuetext={valueText(selected, config)}
      style={{ '--beads': N } as CSSProperties}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId)
        dragging.current = true
        pick(posAt(e.clientX))
      }}
      onPointerMove={(e) => {
        const p = posAt(e.clientX)
        if (dragging.current) pick(p)
        else setHover(p)
      }}
      onPointerUp={() => void (dragging.current = false)}
      onPointerCancel={() => void (dragging.current = false)}
      onPointerLeave={() => setHover(null)}
      onKeyDown={onKeyDown}
    >
      <div className="rod-inner" ref={inner}>
        <span className={`rod-label${shown !== sel ? ' preview' : ''}`} style={{ left: pct(shown) }}>{idOf(targetAt(shown, L))}</span>
        {nowPos !== null && <span className="rod-played" style={{ width: pct(nowPos) }} />}
        {Array.from({ length: N }, (_, p) => (
          <span
            key={p}
            className={`bead ${kindOf(targetAt(p, L), config)}${p === sel ? ' selected' : ''}${p === nowPos ? ' now' : ''}${p === hover ? ' hover' : ''}`}
          />
        ))}
      </div>
      <span className="rod-readout">{idOf(targetAt(shown, L))}</span>
    </div>
  )
}

/** The selected layer or end in plain words: what it sees, which experts the token used, its bytes. */
export function LayerDetail({ config, selected, flowStep, routes }: Omit<DetailProps, 'onSelect'>) {
  const mem = layerMemory(config, typeof selected === 'number' ? selected : undefined)
  const H = config.hidden_size
  const V = config.vocab_size
  const v = V.toLocaleString('en-US')
  let title: ReactNode
  let facts: [string, ReactNode][]
  if (selected === 'embed') {
    title = 'Embedding'
    facts = [
      ['Table', `${v} tokens × ${H} numbers`],
      ['Weights', formatBytes(weightsOf('embed', config))],
      ['Read per token', formatBytes(H * 2)],
    ]
  } else if (selected === 'head') {
    const bytes = formatBytes(weightsOf('head', config))
    title = 'lm_head'
    facts = [
      ['Scores', `all ${v} tokens`],
      ['Weights', bytes],
      ['Read per token', bytes],
    ]
  } else {
    const i = selected
    const rawType = config.layer_types[i]
    const kind = rawType === 'sliding_attention' || rawType === 'linear_attention' ? 'sliding' : 'full'
    const attnText = rawType === 'linear_attention' ? 'linear attention' : `${kind} attention`
    const seesText = rawType === 'linear_attention'
      ? `recurrent state (${config.head_dim}×${config.head_dim})`
      : kind === 'sliding'
        ? `the last ${config.sliding_window} tokens`
        : 'all tokens so far'
    const isDense = i < (config.first_k_dense_replace ?? 0)
    const r = routes[i]
    const shownExperts = r ? (r.experts.length > 6 ? r.experts.slice(0, 6) : r.experts) : []
    title = <>L{i}<span className={`attn-kind ${kind}`}>{attnText}</span></>
    facts = [
      ['Sees', seesText],
      ['Experts used', isDense ? 'dense MLP (all weights active)' : flowStep === null ? 'press ▶ to send a token' : flowStep < i ? 'token not here yet' : (
        <>
          {shownExperts.map((e, j) => `${e} (${Math.round(r.weights[j] * 100)}%)`).join(' · ')}
          {r.experts.length > 6 ? ' …' : ''}
          <span className="muted"> of {config.num_local_experts}, simulated</span>
        </>
      )],
      ['Weights', formatBytes(mem.bytes)],
      ['Read per token', formatBytes(mem.activeBytes)],
    ]
  }

  return (
    <div className="layer-detail" aria-live="polite">
      <h3 className="detail-title">{title}</h3>
      <dl className="detail-facts">
        {facts.map(([dt, dd]) => (
          <div key={dt}><dt>{dt}</dt><dd>{dd}</dd></div>
        ))}
      </dl>
    </div>
  )
}
