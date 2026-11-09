import type { GptOssConfig } from './config'
import { weightDtype } from './quant'

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
/** Bytes for `params` weights stored as `dtype`. */
export const weightBytes = (dtype: Dtype, params: number) => (dtype === 'mxfp4' ? mxfp4Bytes(params) : params * BF16_BYTES)

/** q/k/v/o projections with bias, plus one learned sink logit per query head (or KDA / MLA for Kimi-K3). */
export function attentionParams(c: GptOssConfig, layerIdx = 0): number {
  const H = c.hidden_size
  if (c.layer_types[layerIdx] === 'linear_attention') {
    const P = c.num_attention_heads * c.head_dim
    const conv = c.short_conv_kernel_size ?? 4
    return (
      5 * H * P +
      3 * P * conv +
      H * c.head_dim +
      c.head_dim * P +
      H * c.num_attention_heads +
      c.num_attention_heads +
      P +
      c.head_dim
    )
  }
  if (c.q_lora_rank !== undefined && c.kv_lora_rank !== undefined) {
    const nh = c.num_attention_heads
    const nope = c.qk_nope_head_dim ?? 128
    const rope = c.qk_rope_head_dim ?? 64
    const qHead = nope + rope
    const vHead = c.v_head_dim ?? c.head_dim
    return (
      H * c.q_lora_rank +
      c.q_lora_rank +
      c.q_lora_rank * nh * qHead +
      H * (c.kv_lora_rank + rope) +
      c.kv_lora_rank +
      c.kv_lora_rank * nh * (nope + vHead) +
      (c.mla_use_output_gate ? H * nh * vHead : 0) +
      nh * vHead * H
    )
  }
  const q = c.num_attention_heads * c.head_dim
  const kv = c.num_key_value_heads * c.head_dim
  return H * q + q + 2 * (H * kv + kv) + q * H + H + c.num_attention_heads
}

export function routerParams(c: GptOssConfig, layerIdx = 1): number {
  if (layerIdx < (c.first_k_dense_replace ?? 0)) return 0
  return c.hidden_size * c.num_local_experts + c.num_local_experts
}

/** One expert: fused gate_up (H→2I) and down (I→H), each with bias (or LatentMoE w1/w2/w3). */
export function expertParams(c: GptOssConfig): number {
  const I = c.intermediate_size
  if (c.routed_expert_hidden_size !== undefined) {
    return 3 * c.routed_expert_hidden_size * I
  }
  const H = c.hidden_size
  return H * 2 * I + 2 * I + I * H + H
}

/** Weight matrices of one expert stored in MXFP4 (biases stay bf16). */
export function expertMxfp4Params(c: GptOssConfig): number {
  const I = c.intermediate_size
  if (c.routed_expert_hidden_size !== undefined) {
    return 3 * c.routed_expert_hidden_size * I
  }
  const H = c.hidden_size
  return H * 2 * I + I * H
}

/** Shared expert MLPs + LatentMoE down/up projections and norm (stored in bf16, active every token). */
export function sharedExpertParams(c: GptOssConfig): number {
  if (!c.num_shared_experts) return 0
  const H = c.hidden_size
  const sharedMlp = 3 * H * (c.num_shared_experts * c.intermediate_size)
  const latentProj = c.routed_expert_hidden_size
    ? 2 * H * c.routed_expert_hidden_size + (c.latent_moe_use_norm ? c.routed_expert_hidden_size : 0)
    : 0
  return sharedMlp + latentProj
}

/** Dense MLP parameter count for layers before `first_k_dense_replace`. */
export function denseMlpParams(c: GptOssConfig): number {
  return 3 * c.hidden_size * (c.dense_intermediate_size ?? c.intermediate_size)
}

/** Weights of one decoder block. */
export function layerMemory(c: GptOssConfig, layerIdx = c.first_k_dense_replace ?? 0): LayerMemory {
  const E = c.num_local_experts
  const k = c.num_experts_per_tok
  const H = c.hidden_size
  const isDense = layerIdx < (c.first_k_dense_replace ?? 0)
  const attention = attentionParams(c, layerIdx)
  const router = routerParams(c, layerIdx)
  const expert = expertParams(c)
  const expertMx = expertMxfp4Params(c)
  // Routed expert matrices take the config's dtype; their biases and LatentMoE projections stay bf16.
  const expertBytes = weightBytes(weightDtype(c, 'experts'), expertMx) + (expert - expertMx) * BF16_BYTES
  const shared = sharedExpertParams(c)
  const sharedBytes = weightBytes(weightDtype(c, 'shared_experts'), shared)
  const norms = 2 * H + (c.attn_res_block_size ? 4 * H : 0)
  const row = (id: LayerMemoryRow['id'], label: string, params: number, note: string, dtype: Dtype = 'bf16'): LayerMemoryRow => {
    const bytes = weightBytes(dtype, params)
    return { id, label, params, active: params, dtype, bytes, activeBytes: bytes, note }
  }
  const attnType = c.layer_types[layerIdx]
  const attnLabel = attnType === 'linear_attention'
    ? 'Attention (KDA linear)'
    : c.kv_lora_rank !== undefined
      ? 'Attention (Gated MLA)'
      : 'Attention (q/k/v/o + sinks)'
  const attnNote = attnType === 'linear_attention'
    ? `${c.num_attention_heads} heads × ${c.head_dim}d · conv${c.short_conv_kernel_size ?? 4}`
    : c.kv_lora_rank !== undefined
      ? `${c.num_attention_heads} heads · q/kv LoRA ${c.q_lora_rank}/${c.kv_lora_rank}`
      : `${c.num_attention_heads} q / ${c.num_key_value_heads} kv heads`
  const expertsRow: LayerMemoryRow = isDense
    ? row('experts', 'Dense MLP (SiTU)', denseMlpParams(c), `dense MLP (layer ${layerIdx})`, weightDtype(c, 'dense_mlp'))
    : {
        id: 'experts',
        label: c.routed_expert_hidden_size ? 'Experts (LatentMoE + shared)' : 'Experts (SwiGLU MLPs)',
        params: E * expert + shared,
        active: k * expert + shared,
        dtype: weightDtype(c, 'experts'),
        bytes: E * expertBytes + sharedBytes,
        activeBytes: k * expertBytes + sharedBytes,
        note: c.num_shared_experts ? `${k} of ${E} routed + ${c.num_shared_experts} shared` : `${k} of ${E} per token; biases bf16`,
      }
  const rows: LayerMemoryRow[] = [
    row('attention', attnLabel, attention, attnNote, weightDtype(c, 'attention')),
    row('router', 'MoE router', router, isDense ? 'none (dense layer)' : `Linear ${H}→${E}`, weightDtype(c, 'router')),
    expertsRow,
    row('norms', 'RMSNorm weights', norms, c.attn_res_block_size ? 'input + post-attention + AttnRes' : 'input + post-attention'),
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
  const rep = new Map(layerMemory(c).rows.map((r) => [r.id, r]))
  const totals: Record<LayerMemoryRow['id'], { total: number; active: number; bytes: number }> = {
    attention: { total: 0, active: 0, bytes: 0 },
    router: { total: 0, active: 0, bytes: 0 },
    experts: { total: 0, active: 0, bytes: 0 },
    norms: { total: 0, active: 0, bytes: 0 },
  }
  for (let i = 0; i < L; i++) {
    for (const r of layerMemory(c, i).rows) {
      totals[r.id].total += r.params
      totals[r.id].active += r.active
      totals[r.id].bytes += r.bytes
    }
  }
  const stacked = (id: LayerMemoryRow['id'], note: string, extra = 0, labelOverride?: string): ParamRow => {
    const r = rep.get(id)!
    const t = totals[id]
    return {
      id,
      label: labelOverride ?? r.label,
      total: t.total + extra,
      active: t.active + extra,
      dtype: r.dtype,
      bytes: t.bytes + extra * BF16_BYTES,
      note,
    }
  }
  const kdaCount = c.layer_types.filter((t) => t === 'linear_attention').length
  const attnNote = kdaCount > 0 ? `${kdaCount} KDA + ${L - kdaCount} MLA layers` : `${L} layers`
  const attnLabel = kdaCount > 0 ? 'Attention (KDA + MLA)' : undefined
  const moeLayers = L - (c.first_k_dense_replace ?? 0)

  const rows: ParamRow[] = [
    {
      id: 'embed', label: 'Token embedding', total: embed, active: 0, dtype: weightDtype(c, 'embed'),
      bytes: weightBytes(weightDtype(c, 'embed'), embed), note: 'lookup, excluded from active',
    },
    stacked('attention', attnNote, 0, attnLabel),
    stacked('router', `${moeLayers} × Linear ${H}→${c.num_local_experts}`),
    stacked('experts', rep.get('experts')!.note),
    stacked('norms', c.attn_res_block_size ? '2 per layer + AttnRes + final' : '2 per layer + final', H),
    {
      id: 'lm_head', label: 'lm_head (untied)', total: embed, active: embed, dtype: weightDtype(c, 'lm_head'),
      bytes: weightBytes(weightDtype(c, 'lm_head'), embed), note: `${H}→${V}`,
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
  if (n >= 1e12) return (n / 1e12).toFixed(2) + 'T'
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B'
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M'
  if (n >= 1e3) return (n / 1e3).toFixed(2) + 'K'
  return String(Math.round(n))
}

export function formatBytes(b: number): string {
  if (b < 2 ** 10) return `${Math.round(b)} B`
  if (b >= 2 ** 40) return (b / 2 ** 40).toFixed(2) + ' TiB'
  if (b >= 2 ** 30) return (b / 2 ** 30).toFixed(2) + ' GiB'
  if (b >= 2 ** 20) return (b / 2 ** 20).toFixed(2) + ' MiB'
  return (b / 2 ** 10).toFixed(2) + ' KiB'
}
