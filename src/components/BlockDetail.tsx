import { type ReactNode } from 'react'
import type { GptOssConfig } from '../model/config'
import { formatBytes, formatCount, layerMemory } from '../model/params'
import type { LayerRoute } from '../model/routing'
import { layerTensors } from '../model/tensors'

const GROUP_LABEL: Record<string, string> = {
  norm: 'norms', attention: 'attention', router: 'router', experts: 'experts',
}

interface Props {
  config: GptOssConfig
  /** Whole checkpoint, to put one block's weights in proportion. */
  checkpointBytes: number
  layer: number
  route: LayerRoute | null
}

const W = 760
const RES_X = 40
const BOX_X = 90
const BOX_W = W - BOX_X - 20
const GRID_COLS = 16
const CELL = 16
const GAP = 3

function Box(props: {
  y: number
  h: number
  stroke: string
  children: ReactNode
}) {
  return (
    <g className="svg-box">
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

export function BlockDetail({ config: c, checkpointBytes, layer, route }: Props) {
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
  const isDense = layer < (c.first_k_dense_replace ?? 0)
  const gridCols = E > 256 ? 32 : GRID_COLS
  const cell = E > 256 ? 7 : CELL
  const gap = E > 256 ? 2 : GAP
  const gridX = BOX_X + 20
  const gridY = moeY + 70
  // The explanation text beside the grid needs ~120px even when there are few experts (20b: 2 rows).
  const gridH = isDense ? 80 : Math.max(Math.ceil(E / gridCols) * (cell + gap), 140)
  const routeRows = route && !isDense ? Math.ceil(route.experts.length / 2) : 1
  const moeH = 70 + gridH + Math.max(72, 28 + routeRows * 22)
  const add2 = moeY + moeH + 26
  const H_SVG = add2 + 40
  const weightOf = new Map(route?.experts.map((e, i) => [e, route.weights[i]]) ?? [])
  const rx = gridX + gridCols * (cell + gap) + 16

  const tensors = layerTensors(c, layer)
  const mem = layerMemory(c, layer)

  return (
    <section className="panel detail" aria-label="Block detail">
      <div className="table-head">
        <h2 className="section-title">Layer {layer}</h2>
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

        <Box y={attnY} h={attnH} stroke="var(--bf16)">
          {type === 'linear_attention' ? (
            <>
              <text x={BOX_X + 14} y={attnY + 24} className="t title">Attention (Kimi Delta Attention · KDA)</text>
              <text x={BOX_X + 14} y={attnY + 50} className="t">Q,K,V [T, {nh}, {hd}]   short conv1d (kernel {c.short_conv_kernel_size ?? 4})</text>
              <text x={BOX_X + 14} y={attnY + 72} className="t">decay gate f_a → f_b ({hd}d low-rank) + A_log, dt_bias, β gate b_proj</text>
              <text x={BOX_X + 14} y={attnY + 94} className="t">recurrent state Sₜ = Diag(αₜ) Sₜ₋₁ + βₜ kₜ vₜᵀ   ([nh={nh}, {hd}×{hd}])</text>
              <text x={BOX_X + 14} y={attnY + 116} className="t">output gate g_proj [T, {nh * hd}] · RMSNorm ({hd}d)</text>
              <text x={BOX_X + 14} y={attnY + 138} className="t">o_proj [T, {nh * hd}] → [T, {H}]</text>
            </>
          ) : c.kv_lora_rank !== undefined ? (
            <>
              <text x={BOX_X + 14} y={attnY + 24} className="t title">Attention (Gated Multi-Latent Attention · MLA)</text>
              <text x={BOX_X + 14} y={attnY + 50} className="t">q_a [T, {c.q_lora_rank}] → q_b [T, {nh}, {(c.qk_nope_head_dim ?? 128) + (c.qk_rope_head_dim ?? 64)}]   (nope {c.qk_nope_head_dim ?? 128} + RoPE {c.qk_rope_head_dim ?? 64})</text>
              <text x={BOX_X + 14} y={attnY + 72} className="t">kv_a [T, {c.kv_lora_rank} + {c.qk_rope_head_dim ?? 64}] → kv_b [T, {nh}, {(c.qk_nope_head_dim ?? 128) + (c.v_head_dim ?? hd)}]</text>
              <text x={BOX_X + 14} y={attnY + 94} className="t">KV cache stores compressed latent ({c.kv_lora_rank} + {c.qk_rope_head_dim ?? 64} dims/token)</text>
              <text x={BOX_X + 14} y={attnY + 116} className="t">output gate g_proj [T, {nh * (c.v_head_dim ?? hd)}] · softmax(QKᵀ/√d)·V</text>
              <text x={BOX_X + 14} y={attnY + 138} className="t">o_proj [T, {nh * (c.v_head_dim ?? hd)}] → [T, {H}]</text>
            </>
          ) : (
            <>
              <text x={BOX_X + 14} y={attnY + 24} className="t title">Attention ({type === 'sliding_attention' ? `sliding window ${c.sliding_window}` : 'full'})</text>
              <text x={BOX_X + 14} y={attnY + 50} className="t">Q [T, {nh}, {hd}]   K,V [T, {kv}, {hd}]   (bias: {String(c.attention_bias)})</text>
              <text x={BOX_X + 14} y={attnY + 72} className="t">GQA: {nh / kv} query heads per KV head</text>
              <text x={BOX_X + 14} y={attnY + 94} className="t">RoPE {c.rope_scaling.rope_type.toUpperCase()} ×{c.rope_scaling.factor} (θ={c.rope_theta}, base ctx {c.rope_scaling.original_max_position_embeddings})</text>
              <text x={BOX_X + 14} y={attnY + 116} className="t">softmax(QKᵀ/√{hd} ‖ sink) · V   — scale 1/√{hd}</text>
              <text x={BOX_X + 14} y={attnY + 138} className="t">sinks: {nh} learned logits (extra softmax denominator term)</text>
              <text x={BOX_X + 14} y={attnY + 160} className="t">o_proj [T, {nh * hd}] → [T, {H}]</text>
            </>
          )}
          <text x={BOX_X + BOX_W - 14} y={attnY + 24} className="t dtype bf16-t" textAnchor="end">bf16</text>
        </Box>
        <polyline points={`${BOX_X + BOX_W / 2},${attnY + attnH} ${BOX_X + BOX_W / 2},${add1} ${RES_X + 12},${add1}`} className="wire" markerEnd="url(#arrow)" fill="none" />
        <text x={BOX_X + BOX_W / 2 + 8} y={attnY + attnH + 20} className="t shape-t">[T, {H}]</text>
        <Junction y={add1} />

        <Norm y={norm2} eps={c.rms_norm_eps} />
        <line x1={BOX_X + BOX_W / 2} y1={norm2 + 34} x2={BOX_X + BOX_W / 2} y2={moeY} className="wire" markerEnd="url(#arrow)" />
        <text x={BOX_X + BOX_W / 2 + 8} y={norm2 + 48} className="t shape-t">[T, {H}]</text>

        <Box y={moeY} h={moeH} stroke={isDense ? 'var(--bf16)' : 'var(--mxfp4)'}>
          {isDense ? (
            <>
              <text x={BOX_X + 14} y={moeY + 24} className="t title">Dense MLP ({c.hidden_act === 'situ' ? 'SiTU' : 'SwiGLU'}, layer {layer})</text>
              <text x={BOX_X + 14} y={moeY + 56} className="t">gate_proj, up_proj: [T, {H}] → [T, {c.dense_intermediate_size ?? I}]</text>
              <text x={BOX_X + 14} y={moeY + 80} className="t">h = {c.hidden_act === 'situ' ? `SiTU(gate) · up   (β=${c.activation_situ_beta ?? 4})` : 'SwiGLU(gate, up)'}</text>
              <text x={BOX_X + 14} y={moeY + 104} className="t">down_proj: [T, {c.dense_intermediate_size ?? I}] → [T, {H}]</text>
              <text x={BOX_X + BOX_W - 14} y={moeY + 24} className="t dtype bf16-t" textAnchor="end">bf16</text>
            </>
          ) : (
            <>
              <text x={BOX_X + 14} y={moeY + 24} className="t title">
                {c.routed_expert_hidden_size ? `LatentMoE: top-${k} of ${E} + ${c.num_shared_experts ?? 0} shared` : `MoE: top-${k} of ${E}, softmax over selected`}
              </text>
              <rect x={BOX_X + 14} y={moeY + 34} width={BOX_W - 28} height={24} rx={2} className="box sub" style={{ stroke: 'var(--bf16)' }} />
              <text x={BOX_X + 24} y={moeY + 51} className="t">
                router [T, {H}] → logits [T, {E}] → top-{k} → {c.routed_expert_hidden_size ? 'sigmoid + norm' : 'softmax'}
              </text>
              {Array.from({ length: E }, (_, e) => {
                const w = weightOf.get(e)
                return (
                  <rect
                    key={e}
                    x={gridX + (e % gridCols) * (cell + gap)}
                    y={gridY + Math.floor(e / gridCols) * (cell + gap)}
                    width={cell}
                    height={cell}
                    rx={1}
                    className={w === undefined ? 'cell' : 'cell on'}
                  >
                    <title>expert {e}{w === undefined ? '' : ` · weight ${w.toFixed(3)}`}</title>
                  </rect>
                )
              })}
              {c.routed_expert_hidden_size ? (
                <>
                  <text x={rx} y={gridY + 10} className="t">LatentMoE (MXFP4 routed):</text>
                  <text x={rx} y={gridY + 32} className="t">z = fc_latent_down(x) → [T, {c.routed_expert_hidden_size}]</text>
                  <text x={rx} y={gridY + 52} className="t">w1, w3: [T, {c.routed_expert_hidden_size}] → [T, {I}]</text>
                  <text x={rx} y={gridY + 72} className="t">h = SiTU(w1(z)) · w3(z)</text>
                  <text x={rx} y={gridY + 92} className="t">w2: [T, {I}] → [T, {c.routed_expert_hidden_size}]</text>
                  <text x={rx} y={gridY + 112} className="t">fc_latent_up(norm(Σ wᵢ·eᵢ(z)))</text>
                  <text x={rx} y={gridY + 132} className="t">+ shared_experts(x) → [T, {H}]</text>
                </>
              ) : (
                <>
                  <text x={rx} y={gridY + 10} className="t">each expert (MXFP4 weights):</text>
                  <text x={rx} y={gridY + 32} className="t">gate_up = x·W₁+b₁ → [T, {2 * I}]</text>
                  <text x={rx} y={gridY + 52} className="t">gate = clamp(g, max={lim})</text>
                  <text x={rx} y={gridY + 72} className="t">up = clamp(u, ±{lim})</text>
                  <text x={rx} y={gridY + 92} className="t">h = gate·σ(1.702·gate)·(up+1)</text>
                  <text x={rx} y={gridY + 112} className="t">down = h·W₂+b₂ → [T, {H}]</text>
                  <text x={rx} y={gridY + 132} className="t">out = Σ wᵢ · expertᵢ(x)</text>
                </>
              )}
              <text x={BOX_X + BOX_W - 14} y={moeY + 24} className="t dtype mxfp4-t" textAnchor="end">mxfp4</text>
              {route
                ? route.experts.map((e, i) => (
                    <text key={e} x={gridX + (i % 2) * 260} y={gridY + gridH + 28 + Math.floor(i / 2) * 22} className="t active-t">
                      expert {e} · w={route.weights[i].toFixed(3)}
                    </text>
                  ))
                : <text x={gridX} y={gridY + gridH + 28} className="t muted-t">Play token flow to see this layer's routing</text>}
            </>
          )}
        </Box>
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
        <h3>Tensors</h3>
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
