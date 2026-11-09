import { type KeyboardEvent, type ReactNode } from 'react'
import type { GptOssConfig } from '../model/config'
import { formatBytes, formatCount, layerMemory } from '../model/params'
import type { LayerRoute } from '../model/routing'
import { layerTensors } from '../model/tensors'

const GROUP_LABEL: Record<string, string> = {
  norm: 'norms', attention: 'attention', router: 'router', experts: 'experts',
}

export type Focus = 'attention' | 'moe' | null

interface Props {
  config: GptOssConfig
  /** Whole checkpoint, to put one block's weights in proportion. */
  checkpointBytes: number
  layer: number
  route: LayerRoute | null
  focus: Focus
  onFocus: (f: Focus) => void
}

const W = 760
const RES_X = 40
const BOX_X = 90
const BOX_W = W - BOX_X - 20
const GRID_COLS = 16
const CELL = 16
const GAP = 3

function ClickableBox(props: {
  label: string
  pressed: boolean
  onActivate: () => void
  y: number
  h: number
  stroke: string
  children: ReactNode
}) {
  const onKeyDown = (e: KeyboardEvent<SVGGElement>) => {
    if (e.key !== 'Enter' && e.key !== ' ') return
    e.preventDefault()
    props.onActivate()
  }
  return (
    <g
      role="button"
      tabIndex={0}
      aria-label={props.label}
      aria-pressed={props.pressed}
      className={`svg-box${props.pressed ? ' pressed' : ''}`}
      onClick={props.onActivate}
      onKeyDown={onKeyDown}
    >
      <rect x={BOX_X} y={props.y} width={BOX_W} height={props.h} rx={2} className="box" style={{ stroke: props.stroke }} />
      {props.children}
    </g>
  )
}

function Junction({ y }: { y: number }) {
  return (
    <g className="junction">
      <circle cx={RES_X} cy={y} r={11} />
      <text x={RES_X} y={y + 5} textAnchor="middle">+</text>
    </g>
  )
}

function Norm({ y, eps }: { y: number; eps: number }) {
  return (
    <g>
      <line x1={RES_X} y1={y + 17} x2={BOX_X} y2={y + 17} className="wire" markerEnd="url(#arrow)" />
      <rect x={BOX_X} y={y} width={BOX_W} height={34} rx={2} className="box norm" />
      <text x={BOX_X + 14} y={y + 22} className="t">RMSNorm (ε={eps.toExponential()})</text>
    </g>
  )
}

export function BlockDetail({ config: c, checkpointBytes, layer, route, focus, onFocus }: Props) {
  const type = c.layer_types[layer]
  const H = c.hidden_size
  const I = c.intermediate_size
  const E = c.num_local_experts
  const k = c.num_experts_per_tok
  const nh = c.num_attention_heads
  const kv = c.num_key_value_heads
  const hd = c.head_dim
  const lim = c.swiglu_limit

  const attnY = 94
  const attnH = 178
  const add1 = 306
  const norm2 = 326
  const moeY = 380
  const gridX = BOX_X + 20
  const gridY = moeY + 70
  // The explanation text beside the grid needs ~120px even when there are few experts (20b: 2 rows).
  const gridH = Math.max(Math.ceil(E / GRID_COLS) * (CELL + GAP), 140)
  const moeH = 70 + gridH + 72
  const add2 = moeY + moeH + 26
  const H_SVG = add2 + 40
  const weightOf = new Map(route?.experts.map((e, i) => [e, route.weights[i]]) ?? [])
  const rx = gridX + GRID_COLS * (CELL + GAP) + 16

  const tensors = layerTensors(c, layer).filter((t) =>
    focus === null ? true : focus === 'attention' ? t.group === 'attention' : t.group === 'router' || t.group === 'experts',
  )
  const mem = layerMemory(c)

  return (
    <section className="panel detail" aria-label="Block detail">
      <div className="table-head">
        <h2 className="section-title">Layer {layer} <span className="section-count">{type === 'sliding_attention' ? `sliding · last ${c.sliding_window} tokens` : 'full · all tokens'}</span></h2>
      </div>
      <div className="diagram-scroll">
      <svg viewBox={`0 0 ${W} ${H_SVG}`} className="block-svg" role="group" aria-label={`Decoder block ${layer} diagram`}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--muted)" />
          </marker>
        </defs>
        <text x={RES_X - 30} y={18} className="t muted-t">x · residual stream [T, {H}]   (T = tokens)</text>
        <line x1={RES_X} y1={26} x2={RES_X} y2={add2 + 24} className="residual" />

        <Norm y={40} eps={c.rms_norm_eps} />
        <line x1={BOX_X + BOX_W / 2} y1={74} x2={BOX_X + BOX_W / 2} y2={attnY} className="wire" markerEnd="url(#arrow)" />
        <text x={BOX_X + BOX_W / 2 + 8} y={88} className="t shape-t">[T, {H}]</text>

        <ClickableBox label="Attention: show attention tensors" pressed={focus === 'attention'} onActivate={() => onFocus('attention')} y={attnY} h={attnH} stroke="var(--bf16)">
          <text x={BOX_X + 14} y={attnY + 24} className="t title">Attention ({type === 'sliding_attention' ? `sliding window ${c.sliding_window}` : 'full'})</text>
          <text x={BOX_X + 14} y={attnY + 50} className="t">Q [T, {nh}, {hd}]   K,V [T, {kv}, {hd}]   (bias: {String(c.attention_bias)})</text>
          <text x={BOX_X + 14} y={attnY + 72} className="t">GQA: {nh / kv} query heads per KV head</text>
          <text x={BOX_X + 14} y={attnY + 94} className="t">RoPE {c.rope_scaling.rope_type.toUpperCase()} ×{c.rope_scaling.factor} (θ={c.rope_theta}, base ctx {c.rope_scaling.original_max_position_embeddings})</text>
          <text x={BOX_X + 14} y={attnY + 116} className="t">softmax(QKᵀ/√{hd} ‖ sink) · V   — scale 1/√{hd}</text>
          <text x={BOX_X + 14} y={attnY + 138} className="t">sinks: {nh} learned logits (extra softmax denominator term)</text>
          <text x={BOX_X + 14} y={attnY + 160} className="t">o_proj [T, {nh * hd}] → [T, {H}]</text>
          <text x={BOX_X + BOX_W - 14} y={attnY + 24} className="t dtype bf16-t" textAnchor="end">bf16</text>
        </ClickableBox>
        <polyline points={`${BOX_X + BOX_W / 2},${attnY + attnH} ${BOX_X + BOX_W / 2},${add1} ${RES_X + 12},${add1}`} className="wire" markerEnd="url(#arrow)" fill="none" />
        <text x={BOX_X + BOX_W / 2 + 8} y={attnY + attnH + 20} className="t shape-t">[T, {H}]</text>
        <Junction y={add1} />

        <Norm y={norm2} eps={c.rms_norm_eps} />
        <line x1={BOX_X + BOX_W / 2} y1={norm2 + 34} x2={BOX_X + BOX_W / 2} y2={moeY} className="wire" markerEnd="url(#arrow)" />
        <text x={BOX_X + BOX_W / 2 + 8} y={norm2 + 48} className="t shape-t">[T, {H}]</text>

        <ClickableBox label="Mixture of experts: show MoE tensors" pressed={focus === 'moe'} onActivate={() => onFocus('moe')} y={moeY} h={moeH} stroke="var(--mxfp4)">
          <text x={BOX_X + 14} y={moeY + 24} className="t title">MoE: top-{k} of {E}, softmax over selected</text>
          <rect x={BOX_X + 14} y={moeY + 34} width={BOX_W - 28} height={24} rx={2} className="box sub" style={{ stroke: 'var(--bf16)' }} />
          <text x={BOX_X + 24} y={moeY + 51} className="t">router [T, {H}] → logits [T, {E}] → top-{k} → softmax</text>
          {Array.from({ length: E }, (_, e) => {
            const w = weightOf.get(e)
            return (
              <rect
                key={e}
                x={gridX + (e % GRID_COLS) * (CELL + GAP)}
                y={gridY + Math.floor(e / GRID_COLS) * (CELL + GAP)}
                width={CELL}
                height={CELL}
                rx={1}
                className={w === undefined ? 'cell' : 'cell on'}
              >
                <title>expert {e}{w === undefined ? '' : ` · weight ${w.toFixed(3)}`}</title>
              </rect>
            )
          })}
          <text x={rx} y={gridY + 10} className="t">each expert (MXFP4 weights):</text>
          <text x={rx} y={gridY + 32} className="t">gate_up = x·W₁+b₁ → [T, {2 * I}]</text>
          <text x={rx} y={gridY + 52} className="t">gate = clamp(g, max={lim})</text>
          <text x={rx} y={gridY + 72} className="t">up = clamp(u, ±{lim})</text>
          <text x={rx} y={gridY + 92} className="t">h = gate·σ(1.702·gate)·(up+1)</text>
          <text x={rx} y={gridY + 112} className="t">down = h·W₂+b₂ → [T, {H}]</text>
          <text x={rx} y={gridY + 132} className="t">out = Σ wᵢ · expertᵢ(x)</text>
          <text x={BOX_X + BOX_W - 14} y={moeY + 24} className="t dtype mxfp4-t" textAnchor="end">mxfp4</text>
          {route
            ? route.experts.map((e, i) => (
                <text key={e} x={gridX + (i % 2) * 260} y={gridY + gridH + 28 + Math.floor(i / 2) * 22} className="t active-t">
                  expert {e} · w={route.weights[i].toFixed(3)}
                </text>
              ))
            : <text x={gridX} y={gridY + gridH + 28} className="t muted-t">Play token flow to see this layer's routing</text>}
        </ClickableBox>
        <polyline points={`${BOX_X + BOX_W / 2},${moeY + moeH} ${BOX_X + BOX_W / 2},${add2} ${RES_X + 12},${add2}`} className="wire" markerEnd="url(#arrow)" fill="none" />
        <text x={BOX_X + BOX_W / 2 + 8} y={moeY + moeH + 18} className="t shape-t">[T, {H}]</text>
        <Junction y={add2} />
        <text x={RES_X + 20} y={add2 + 32} className="t muted-t">→ {layer + 1 < c.num_hidden_layers ? `layer ${layer + 1}` : 'final RMSNorm → lm_head'}</text>
      </svg>
      </div>

      <div className="table-head"><h3>Layer memory <span className="section-count">{formatBytes(mem.bytes)} × {c.num_hidden_layers} = {((c.num_hidden_layers * mem.bytes / checkpointBytes) * 100).toFixed(1)}% of checkpoint</span></h3></div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>component</th><th className="num">params</th><th className="num">active / token</th>
              <th>dtype</th><th className="num">bytes</th><th className="num">read / token</th>
            </tr>
          </thead>
          <tbody>
            {mem.rows.map((r) => (
              <tr key={r.id}>
                <td>{r.label}<div className="muted small">{r.note}</div></td>
                <td className="num">{formatCount(r.params)}</td>
                <td className="num">{formatCount(r.active)}</td>
                <td><span className={`dtype-tag ${r.dtype}`}>{r.dtype}</span></td>
                <td className="num">{formatBytes(r.bytes)}</td>
                <td className="num">{formatBytes(r.activeBytes)}</td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <th>This block</th>
              <td className="num">{formatCount(mem.params)}</td>
              <td className="num">{formatCount(mem.active)}</td>
              <td />
              <td className="num">{formatBytes(mem.bytes)}</td>
              <td className="num">{formatBytes(mem.activeBytes)}</td>
            </tr>
          </tfoot>
        </table>
      </div>

      <div className="table-head global-head">
        <h3>Tensors{focus ? ` · ${focus}` : ''}</h3>
        {focus && <button type="button" className="button-secondary" onClick={() => onFocus(null)}>Show all</button>}
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>name</th><th>shape</th><th>dtype</th><th className="num">params</th></tr>
          </thead>
          <tbody>
            {tensors.map((t, i) => [
              (i === 0 || t.group !== tensors[i - 1].group) && (
                <tr key={`g-${i}`} className="group-row"><th colSpan={4}>{GROUP_LABEL[t.group]}</th></tr>
              ),
              <tr key={t.name}>
                <td className="mono">{t.name}</td>
                <td className="mono shape">{t.shape.join(' × ')}</td>
                <td><span className={`dtype-tag ${t.dtype}`}>{t.dtype}</span></td>
                <td className="num">{formatCount(t.shape.reduce((a, b) => a * b, 1))}</td>
              </tr>,
            ])}
          </tbody>
        </table>
      </div>
    </section>
  )
}
