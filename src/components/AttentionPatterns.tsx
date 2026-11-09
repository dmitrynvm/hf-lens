import type { GptOssConfig, LayerType } from '../model/config'
import { kvCacheBytes } from '../model/kvcache'
import { formatBytes } from '../model/params'

interface Props {
  config: GptOssConfig
  layer: number
}

const S = 200 // mask side in SVG units
const PAD_L = 72
const PAD_T = 30

function MaskPanel({ config: c, type, current }: { config: GptOssConfig; type: LayerType; current: boolean }) {
  const w = c.sliding_window
  const n = 4 * w // tokens shown: long enough that the window is visibly narrower than the context
  const s = S / n
  const sliding = type === 'sliding_attention'
  const points = sliding
    ? `0,0 ${S},${S} ${(n - w) * s},${S} 0,${w * s}`
    : `0,0 ${S},${S} 0,${S}`
  const count = c.layer_types.filter((t) => t === type).length
  return (
    <figure className={`mask-panel ${sliding ? 'sliding' : 'full'}${current ? ' current' : ''}`}>
      <svg viewBox={`0 0 ${PAD_L + S + 16} ${PAD_T + S + 40}`} role="img"
        aria-label={sliding
          ? `Sliding attention mask: each query sees the previous ${w} tokens plus the sink`
          : 'Full causal attention mask: each query sees all previous tokens plus the sink'}>
        <g transform={`translate(${PAD_L},${PAD_T})`}>
          <rect x={0} y={0} width={S} height={S} className="mask-bg" />
          <polygon points={points} className="mask-on" />
          <rect x={-18} y={0} width={10} height={S} className="mask-sink" />
          <text x={-13} y={S + 14} className="t small-t" textAnchor="middle">sink</text>
          <text x={0} y={-8} className="t small-t">key 0 →</text>
          <text x={S} y={-8} className="t small-t" textAnchor="end">{n}</text>
          <text x={-24} y={8} className="t small-t" textAnchor="end">query 0</text>
          <text x={-24} y={S} className="t small-t" textAnchor="end">{n}</text>
          {sliding && (
            <g>
              <line x1={(n - w) * s} y1={S + 10} x2={S} y2={S + 10} className="dim" />
              <text x={(n - w / 2) * s} y={S + 26} className="t small-t" textAnchor="middle">{w} tokens</text>
            </g>
          )}
        </g>
      </svg>
      <figcaption>
        <strong>{sliding ? `sliding window ${w}` : 'full causal'}</strong> · {count} layers
        {current && <span className="badge current-badge">selected layer</span>}
      </figcaption>
    </figure>
  )
}

function HeadGroups({ config: c }: { config: GptOssConfig }) {
  const nh = c.num_attention_heads
  const kv = c.num_key_value_heads
  const group = nh / kv
  const cell = 8
  const gap = 2
  const groupGap = 12
  const groupW = group * (cell + gap) - gap
  const width = kv * groupW + (kv - 1) * groupGap
  const qY = 24
  const kvY = 88
  const groupX = (g: number) => g * (groupW + groupGap)
  return (
    <figure className="heads">
      <svg viewBox={`-4 0 ${width + 8} ${kvY + 40}`} role="img"
        aria-label={`${nh} query heads in ${kv} groups of ${group}; each group shares one key/value head`}>
        <text x={0} y={14} className="t small-t">{nh} query heads (each with its own sink logit)</text>
        {Array.from({ length: kv }, (_, g) => {
          const kvX = groupX(g) + groupW / 2
          return (
            <g key={g}>
              {Array.from({ length: group }, (_, h) => {
                const x = groupX(g) + h * (cell + gap)
                return (
                  <g key={h}>
                    <line x1={x + cell / 2} y1={qY + 16} x2={kvX} y2={kvY} className="head-link" />
                    <rect x={x} y={qY} width={cell} height={16} rx={1} className="q-head" />
                  </g>
                )
              })}
              <rect x={kvX - 18} y={kvY} width={36} height={16} rx={2} className="kv-head" />
              <text x={kvX} y={kvY + 12} className="t small-t kv-label" textAnchor="middle">KV{g}</text>
            </g>
          )
        })}
        <text x={0} y={kvY + 34} className="t small-t">
          {kv} key/value heads × {c.head_dim} dims: {group} query heads share each, so the KV cache is {group}× smaller than multi-head attention
        </text>
      </svg>
    </figure>
  )
}

export function AttentionPatterns({ config: c, layer }: Props) {
  const type = c.layer_types[layer]
  const contexts = [c.rope_scaling.original_max_position_embeddings, c.max_position_embeddings / 4, c.max_position_embeddings]
  const rows = contexts.map((ctx) => {
    const actual = kvCacheBytes(c, ctx)
    const mha = kvCacheBytes(c, ctx, { kvHeads: c.num_attention_heads, useSliding: false })
    return { ctx, actual, allFull: kvCacheBytes(c, ctx, { useSliding: false }), mha }
  })

  return (
    <section className="panel attn" aria-label="Attention patterns and KV cache">
      <h2 className="section-title">Attention patterns &amp; KV cache</h2>
      <div className="masks" title={`${4 * c.sliding_window} tokens shown`}>
        <MaskPanel config={c} type="sliding_attention" current={type === 'sliding_attention'} />
        <MaskPanel config={c} type="full_attention" current={type === 'full_attention'} />
      </div>

      <h3>Heads: grouped-query attention</h3>
      <HeadGroups config={c} />

      <h3 title={`per token per layer: ${c.num_key_value_heads} KV heads × ${c.head_dim} dims × K,V × 2 bytes; sliding layers keep the last ${c.sliding_window} tokens`}>KV cache per sequence (bf16)</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th className="num">context</th>
              <th className="num">gpt-oss (GQA + sliding)</th>
              <th className="num">GQA, all full</th>
              <th className="num">MHA ({c.num_attention_heads} KV heads), all full</th>
              <th title="bars, top to bottom: gpt-oss, GQA all full, MHA; one scale per row">size vs MHA</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.ctx}>
                <td className="num">{r.ctx.toLocaleString('en-US')}</td>
                <td className="num">{formatBytes(r.actual)}</td>
                <td className="num">{formatBytes(r.allFull)}</td>
                <td className="num">{formatBytes(r.mha)}</td>
                <td className="share">
                  <span className="kv-bars" aria-hidden="true">
                    <span className="bar kv" style={{ width: `${(r.actual / r.mha) * 100}%` }} />
                    <span className="bar gqa" style={{ width: `${(r.allFull / r.mha) * 100}%` }} />
                    <span className="bar mha" style={{ width: '100%' }} />
                  </span>
                  <span className="num">{(r.mha / r.actual).toFixed(1)}× smaller</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
