import type { GptOssConfig } from './config'

export type Dtype = 'bf16' | 'mxfp4'

export interface ParamRow {
  id: 'embed' | 'attention' | 'router' | 'experts' | 'norms' | 'lm_head'
  label: string
  total: number
  active: number
  dtype: Dtype
  bytes: number
  note: string
}

export interface ParamBreakdown {
  rows: ParamRow[]
  total: number
  activeExclEmbed: number
  activeInclEmbed: number
  bytes: number
}

export interface LayerMemoryRow {
  id: 'attention' | 'router' | 'experts' | 'norms'
  label: string
  params: number
  active: number
  dtype: Dtype
  bytes: number
  /** Weight bytes read per token: everything except the experts the router skipped. */
  activeBytes: number
  note: string
}

export interface LayerMemory {
  rows: LayerMemoryRow[]
  params: number
  active: number
  bytes: number
  activeBytes: number
}

const BF16_BYTES = 2
/** MXFP4: 4-bit values + one 8-bit scale per 32-element block = 4.25 bits/param = 17/32 bytes. */
export const mxfp4Bytes = (params: number) => (params * 17) / 32

/** q/k/v/o projections with bias, plus one learned sink logit per query head. */
export function attentionParams(c: GptOssConfig): number {
  const H = c.hidden_size
  const q = c.num_attention_heads * c.head_dim
  const kv = c.num_key_value_heads * c.head_dim
  return H * q + q + 2 * (H * kv + kv) + q * H + H + c.num_attention_heads
}

export function routerParams(c: GptOssConfig): number {
  return c.hidden_size * c.num_local_experts + c.num_local_experts
}

/** One expert: fused gate_up (H→2I) and down (I→H), each with bias. */
export function expertParams(c: GptOssConfig): number {
  const H = c.hidden_size
  const I = c.intermediate_size
  return H * 2 * I + 2 * I + I * H + H
}

/** Weight matrices of one expert stored in MXFP4 (biases stay bf16). */
export function expertMxfp4Params(c: GptOssConfig): number {
  const H = c.hidden_size
  const I = c.intermediate_size
  return H * 2 * I + I * H
}

/** Weights of one decoder block. Every block is identical; sliding vs. full only changes the KV cache. */
export function layerMemory(c: GptOssConfig): LayerMemory {
  const E = c.num_local_experts
  const k = c.num_experts_per_tok
  const H = c.hidden_size
  const attention = attentionParams(c)
  const router = routerParams(c)
  const expert = expertParams(c)
  const expertMx = expertMxfp4Params(c)
  const expertBytes = mxfp4Bytes(expertMx) + (expert - expertMx) * BF16_BYTES
  const norms = 2 * H
  const dense = (id: LayerMemoryRow['id'], label: string, params: number, note: string): LayerMemoryRow => ({
    id, label, params, active: params, dtype: 'bf16', bytes: params * BF16_BYTES, activeBytes: params * BF16_BYTES, note,
  })
  const rows: LayerMemoryRow[] = [
    dense('attention', 'Attention (q/k/v/o + sinks)', attention, `${c.num_attention_heads} q / ${c.num_key_value_heads} kv heads`),
    dense('router', 'MoE router', router, `Linear ${H}→${E}`),
    {
      id: 'experts', label: 'Experts (SwiGLU MLPs)', params: E * expert, active: k * expert, dtype: 'mxfp4',
      bytes: E * expertBytes, activeBytes: k * expertBytes, note: `${k} of ${E} per token; biases bf16`,
    },
    dense('norms', 'RMSNorm weights', norms, 'input + post-attention'),
  ]
  const sum = (f: (r: LayerMemoryRow) => number) => rows.reduce((acc, r) => acc + f(r), 0)
  return {
    rows,
    params: sum((r) => r.params),
    active: sum((r) => r.active),
    bytes: sum((r) => r.bytes),
    activeBytes: sum((r) => r.activeBytes),
  }
}

export function paramBreakdown(c: GptOssConfig): ParamBreakdown {
  const L = c.num_hidden_layers
  const H = c.hidden_size
  const V = c.vocab_size
  const embed = V * H
  const layer = new Map(layerMemory(c).rows.map((r) => [r.id, r]))
  const stacked = (id: LayerMemoryRow['id'], note: string, extra = 0): ParamRow => {
    const r = layer.get(id)!
    return {
      id, label: r.label, total: L * r.params + extra, active: L * r.active + extra, dtype: r.dtype,
      bytes: L * r.bytes + extra * BF16_BYTES, note,
    }
  }

  const rows: ParamRow[] = [
    {
      id: 'embed', label: 'Token embedding', total: embed, active: 0, dtype: 'bf16',
      bytes: embed * BF16_BYTES, note: 'lookup, excluded from active',
    },
    stacked('attention', `${L} layers`),
    stacked('router', `${L} × Linear ${H}→${c.num_local_experts}`),
    stacked('experts', layer.get('experts')!.note),
    stacked('norms', '2 per layer + final', H),
    {
      id: 'lm_head', label: 'lm_head (untied)', total: embed, active: embed, dtype: 'bf16',
      bytes: embed * BF16_BYTES, note: `${H}→${V}`,
    },
  ]

  const sum = (f: (r: ParamRow) => number) => rows.reduce((acc, r) => acc + f(r), 0)
  const activeExclEmbed = sum((r) => r.active)
  return {
    rows,
    total: sum((r) => r.total),
    activeExclEmbed,
    activeInclEmbed: activeExclEmbed + embed,
    bytes: sum((r) => r.bytes),
  }
}

export function formatCount(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B'
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M'
  if (n >= 1e3) return (n / 1e3).toFixed(2) + 'K'
  return String(Math.round(n))
}

export function formatBytes(b: number): string {
  if (b < 2 ** 10) return `${Math.round(b)} B`
  if (b >= 2 ** 30) return (b / 2 ** 30).toFixed(2) + ' GiB'
  if (b >= 2 ** 20) return (b / 2 ** 20).toFixed(2) + ' MiB'
  return (b / 2 ** 10).toFixed(2) + ' KiB'
}
