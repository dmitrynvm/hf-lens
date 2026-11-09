import { useRef, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from 'react'
import type { GptOssConfig } from '../model/config'
import type { Target } from '../model/network'
import { formatBytes, layerMemory } from '../model/params'
import type { LayerRoute } from '../model/routing'

interface Props {
  config: GptOssConfig
  /** Selected bead: a decoder layer or one of the ends; the arrow keys step from it. */
  selected: Target
  onSelect: (target: Target) => void
  flowStep: number | null
  routes: LayerRoute[]
}

/** Rod order: the embedding, the decoder layers, lm_head. A bead's flow step is its position - 1 (embedding -1, lm_head L). */
const positionOf = (t: Target, L: number) => (t === 'embed' ? 0 : t === 'head' ? L + 1 : t + 1)
const targetAt = (p: number, L: number): Target => (p === 0 ? 'embed' : p === L + 1 ? 'head' : p - 1)
const kindOf = (t: Target, c: GptOssConfig) => (t === 'embed' || t === 'head' ? 'end' : c.layer_types[t] === 'sliding_attention' ? 'sliding' : 'full')
const idOf = (t: Target) => (t === 'embed' ? 'In' : t === 'head' ? 'Out' : `L${t}`)
const valueText = (t: Target, c: GptOssConfig) =>
  t === 'embed' ? 'In · embedding' : t === 'head' ? 'Out · lm_head' : `L${t} · ${kindOf(t, c)} attention`

/** Keys that move along the rod, by how many beads; Home / End jump to the ends. */
const KEY_STEP: Record<string, number> = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1, PageDown: -6, PageUp: 6 }

/** The layers as beads on one rod, like frames on a video timeline: click, drag or use the arrow keys; the token's path is the played stretch. */
export function LayerStack({ config, selected, onSelect, flowStep, routes }: Props) {
  const L = config.num_hidden_layers
  const N = L + 2
  const inner = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)
  const [hover, setHover] = useState<number | null>(null)
  const sel = positionOf(selected, L)
  const nowPos = flowStep === null ? null : Math.min(flowStep + 1, N - 1)
  const pct = (p: number) => `${(p / (N - 1)) * 100}%`
  const posAt = (clientX: number) => {
    const r = inner.current!.getBoundingClientRect()
    return Math.min(N - 1, Math.max(0, Math.round(((clientX - r.left) / r.width) * (N - 1))))
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
    <section className="stack" aria-label="Layer stack">
      <div className="stack-head">
        <h2 className="section-title">Layers<span className="section-count">{L}</span></h2>
        <span className="stack-legend">
          <span className="legend-item"><span className="swatch end" aria-hidden="true" />In · embedding</span>
          <span className="legend-item"><span className="swatch sliding" aria-hidden="true" />sliding · last {config.sliding_window} tokens</span>
          <span className="legend-item"><span className="swatch full" aria-hidden="true" />full · all tokens</span>
          <span className="legend-item"><span className="swatch end" aria-hidden="true" />Out · lm_head</span>
        </span>
      </div>
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
        style={{ '--gaps': N - 1 } as CSSProperties}
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
          <span className="rod-track" />
          {nowPos !== null && <span className="rod-played" style={{ width: pct(nowPos) }} />}
          {Array.from({ length: N }, (_, p) => (
            <span
              key={p}
              className={`bead ${kindOf(targetAt(p, L), config)}${p === sel ? ' selected' : ''}${p === nowPos ? ' now' : ''}${p === hover ? ' hover' : ''}`}
              style={{ left: pct(p) }}
            />
          ))}
        </div>
      </div>
      <LayerDetail config={config} selected={selected} flowStep={flowStep} routes={routes} />
    </section>
  )
}

/** The selected layer or end in plain words: what it sees, which experts the token used, its bytes. */
function LayerDetail({ config, selected, flowStep, routes }: { config: GptOssConfig; selected: Target; flowStep: number | null; routes: LayerRoute[] }) {
  const mem = layerMemory(config)
  const H = config.hidden_size
  const V = config.vocab_size
  const v = V.toLocaleString('en-US')
  let title: ReactNode
  let facts: [string, ReactNode][]
  if (selected === 'embed') {
    title = 'Embedding'
    facts = [
      ['Table', `${v} tokens × ${H} numbers`],
      ['Weights', formatBytes(V * H * 2)],
      ['Read per token', formatBytes(H * 2)],
    ]
  } else if (selected === 'head') {
    const bytes = formatBytes((V * H + H) * 2)
    title = 'lm_head'
    facts = [
      ['Scores', `all ${v} tokens`],
      ['Weights', bytes],
      ['Read per token', bytes],
    ]
  } else {
    const i = selected
    const kind = config.layer_types[i] === 'sliding_attention' ? 'sliding' : 'full'
    const r = routes[i]
    title = <>L{i}<span className={`attn-kind ${kind}`}>{kind} attention</span></>
    facts = [
      ['Sees', kind === 'sliding' ? `the last ${config.sliding_window} tokens` : 'all tokens so far'],
      ['Experts used', flowStep === null ? 'press ▶ to send a token' : flowStep < i ? 'token not here yet' : (
        <>
          {r.experts.map((e, j) => `${e} (${Math.round(r.weights[j] * 100)}%)`).join(' · ')}
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
