import type { ModelEntry } from '../model/config'
import { formatBytes, formatCount, mxfp4Bytes, type ParamBreakdown } from '../model/params'
import { quantSummary, quantText } from '../model/quant'

interface Props {
  model: ModelEntry
  breakdown: ParamBreakdown
}

export function ModelSummary({ model, breakdown: b }: Props) {
  const c = model.config
  const isKimi = c.model_type === 'kimi_k3' || c.model_type === 'kimi_linear' || c.routed_expert_hidden_size !== undefined
  const hasVision = 'vision_config' in model.json && model.json.vision_config !== null
  const kda = c.layer_types.filter((t) => t === 'linear_attention').length
  const sliding = c.layer_types.filter((t) => t === 'sliding_attention').length
  const full = c.num_hidden_layers - kda - sliding

  const summaryRows: [string, string][] = [
    ['Architecture', 'Mixture-of-Experts (MoE)'],
    ['Total Parameters', isKimi ? `2.8T (${formatCount(b.total)} text)` : `${formatCount(b.total)} (${b.total.toLocaleString('en-US')})`],
    ['Activated Parameters', isKimi ? '104B' : `${formatCount(b.activeExclEmbed)} (${b.activeExclEmbed.toLocaleString('en-US')})`],
    ['Number of Layers', String(c.num_hidden_layers)],
    ['Number of Dense Layers', String(c.first_k_dense_replace ?? 0)],
    [
      'Attention-Layer Composition',
      kda > 0
        ? `${kda} KDA + ${full} Gated MLA`
        : `${sliding} SWA (window ${c.sliding_window}) + ${full} Full`,
    ],
    ['Attention Hidden Dimension', String(c.hidden_size)],
    [
      'Number of Attention Heads',
      isKimi
        ? String(c.num_attention_heads)
        : `${c.num_attention_heads} Q / ${c.num_key_value_heads} KV (head_dim ${c.head_dim})`,
    ],
    ...(c.routed_expert_hidden_size !== undefined
      ? [['Latent MoE Dimension', String(c.routed_expert_hidden_size)] as [string, string]]
      : []),
    ['MoE Hidden Dimension (per Expert)', String(c.intermediate_size)],
    ['Number of Experts', String(c.num_local_experts)],
    ['Selected Experts per Token', String(c.num_experts_per_tok)],
    ['Number of Shared Experts', String(c.num_shared_experts ?? 0)],
    ['Vocabulary Size', c.vocab_size === 163840 ? '160K (163,840)' : c.vocab_size.toLocaleString('en-US')],
    ['Context Length', String(c.max_position_embeddings)],
    [
      'Attention Mechanism',
      isKimi
        ? 'KDA & Gated MLA'
        : `GQA + Sliding Window (${c.sliding_window}) + Learned Sinks + ${c.rope_scaling.rope_type.toUpperCase()} RoPE (×${c.rope_scaling.factor})`,
    ],
    [
      'Activation Function',
      isKimi ? 'SiTU-GLU' : `Clamped SwiGLU (limit ±${c.swiglu_limit})`,
    ],
    ...(hasVision
      ? ([
          ['Vision Encoder', 'MoonViT-V2'],
          ['Parameters of Vision Encoder', '401M'],
        ] as [string, string][])
      : []),
    ['Quantization', quantText(quantSummary(c))],
    ['Modality', hasVision ? 'Text, Image' : 'Text'],
  ]

  const expertsRow = b.rows.find((r) => r.id === 'experts')
  const expertTotal = expertsRow?.total ?? 0
  const expertActive = expertsRow?.active ?? 0
  const nonExpertTotal = b.total - expertTotal
  const nonExpertActive = b.activeExclEmbed - expertActive

  const quantRows = [
    {
      id: 'bf16',
      label: 'BF16',
      dtype: 'bf16',
      totalBytes: b.total * 2,
      activeBytes: b.activeExclEmbed * 2,
    },
    {
      id: 'fp16',
      label: 'FP16',
      dtype: 'fp16',
      totalBytes: b.total * 2,
      activeBytes: b.activeExclEmbed * 2,
    },
    {
      id: 'fp8-mixed',
      label: 'FP8 + BF16 (Mixed)',
      dtype: 'fp8',
      totalBytes: expertTotal + nonExpertTotal * 2,
      activeBytes: expertActive + nonExpertActive * 2,
    },
    {
      id: 'fp8',
      label: 'FP8',
      dtype: 'fp8',
      totalBytes: b.total,
      activeBytes: b.activeExclEmbed,
    },
    {
      id: 'int8',
      label: 'INT8',
      dtype: 'int8',
      totalBytes: b.total,
      activeBytes: b.activeExclEmbed,
    },
    {
      id: 'mxfp4',
      label: 'MXFP4 (Native Checkpoint)',
      dtype: 'mxfp4',
      totalBytes: b.bytes,
      activeBytes: mxfp4Bytes(expertActive) + nonExpertActive * 2,
    },
    {
      id: 'int4',
      label: 'INT4',
      dtype: 'int4',
      totalBytes: b.total * 0.5,
      activeBytes: b.activeExclEmbed * 0.5,
    },
  ]
  return (
    <section className="panel model-summary" aria-label="Model summary">
      <div className="table-head">
        <h2 className="section-title">Summary</h2>
      </div>
      <div className="table-wrap">
        <table>
          <tbody>
            {summaryRows.map(([k, v]) => (
              <tr key={k}>
                <td>{k}</td>
                <td className="num">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="table-head global-head">
        <h3>Model Size by Quantization <span className="section-count">{formatCount(b.total)} params</span></h3>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>dtype</th>
              <th className="num">bits / param</th>
              <th className="num">active / token</th>
              <th className="num">model size</th>
            </tr>
          </thead>
          <tbody>
            {quantRows.map((q) => (
              <tr key={q.id}>
                <td>{q.label}</td>
                <td className="num">{((q.totalBytes * 8) / b.total).toFixed(2)}</td>
                <td className="num">{formatBytes(q.activeBytes)}</td>
                <td className="num">{formatBytes(q.totalBytes)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="table-head global-head">
        <h3>Parameter &amp; Checkpoint Breakdown <span className="section-count">{formatBytes(b.bytes)} on disk</span></h3>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>component</th>
              <th className="num">total params</th>
              <th className="num">active / token</th>
              <th>dtype</th>
              <th className="num">checkpoint bytes</th>
            </tr>
          </thead>
          <tbody>
            {b.rows.map((r) => (
              <tr key={r.id}>
                <td>{r.label}<div className="muted small">{r.note}</div></td>
                <td className="num">{formatCount(r.total)}</td>
                <td className="num">{formatCount(r.active)}</td>
                <td><span className={`dtype-tag ${r.dtype}`}>{r.dtype}</span></td>
                <td className="num">{formatBytes(r.bytes)}</td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <th>Total</th>
              <td className="num">{formatCount(b.total)}</td>
              <td className="num">{formatCount(b.activeExclEmbed)}</td>
              <td />
              <td className="num">{formatBytes(b.bytes)}</td>
            </tr>
          </tfoot>
        </table>
      </div>
    </section>
  )
}
