import type { GptOssConfig, ModelEntry } from './config'
import { expertMxfp4Params, expertParams, mxfp4Bytes, paramBreakdown, type Dtype } from './params'
import { quantSummary, ROLE_LABELS, weightDtype, type WeightRole } from './quant'

export interface TensorTypeRow {
  dtype: string
  params: number
  /** Bytes on disk: MXFP4 at 4.25 bits per param (U8 blocks + scales), otherwise the dtype's element size. */
  bytes: number
  /** The diagram's storage dtype these tensors are drawn as; null when config.json does not say which tensors they are. */
  storage: Dtype | null
  /** What these tensors are and where they sit, e.g. `routed experts · L0–L35`, for the picker's hover; empty when not attributable. */
  where: string[]
}

export const DEFAULT_TENSOR_TYPE = 'BF16'

/** Element size of safetensors dtypes; U8 holding MXFP4 is sized by `mxfp4Bytes` instead. */
const ELEMENT_BYTES: Record<string, number> = { F64: 8, F32: 4, F16: 2, BF16: 2, I64: 8, I32: 4, I16: 2, I8: 1, U8: 1, BOOL: 1, F8_E4M3: 1, F8_E5M2: 1 }

const span = (a: number, b: number) => (a === b ? `L${a}` : `L${a}–L${b}`)

/** Each weight stored as `d`, with the layers it sits in, in network order; the same parts the diagram lights for `d`. */
function placements(c: GptOssConfig, roles: WeightRole[], d: Dtype): string[] {
  const L = c.num_hidden_layers
  const k = c.first_k_dense_replace ?? 0
  const at: Record<WeightRole, string> = {
    embed: 'input', attention: span(0, L - 1), dense_mlp: span(0, k - 1), router: span(k, L - 1),
    experts: span(k, L - 1), shared_experts: span(k, L - 1), lm_head: 'output',
  }
  const out = roles.map((r) => `${ROLE_LABELS[r]} · ${at[r]}`)
  if (d === 'bf16') {
    // Always bf16, whatever quantization_config says: norms, and the biases of quantized expert matrices.
    if (weightDtype(c, 'experts') === 'mxfp4' && expertParams(c) > expertMxfp4Params(c)) out.push(`expert biases · ${at.experts}`)
    out.push(`RMSNorms · ${span(0, L - 1)} + final`)
  }
  return out
}

/** The checkpoint's stored tensor types as the hub's Safetensors box lists them: BF16 first (the default pick), then the hub's order. */
export function tensorTypes(m: ModelEntry): TensorTypeRow[] {
  const c = m.config
  const s = quantSummary(c)
  // MXFP4 is packed into U8 blocks and scales; everything else is BF16 (or, on the hub, a few F32 tensors).
  const row = ([dtype, params]: [string, number]): TensorTypeRow =>
    dtype === 'U8' && s.mxfp4.length ? { dtype, params, bytes: mxfp4Bytes(params), storage: 'mxfp4', where: placements(c, s.mxfp4, 'mxfp4') }
      : dtype === 'BF16' ? { dtype, params, bytes: params * 2, storage: 'bf16', where: placements(c, s.bf16, 'bf16') }
        : { dtype, params, bytes: params * (ELEMENT_BYTES[dtype] ?? 0), storage: null, where: [] }
  const ordered = (entries: [string, number][]) =>
    entries.filter(([, n]) => n > 0).map(row).sort((a, b) => Number(b.dtype === DEFAULT_TENSOR_TYPE) - Number(a.dtype === DEFAULT_TENSOR_TYPE))
  if (m.safetensors) return ordered(Object.entries(m.safetensors.parameters))
  // Not fetched from the hub: computed from config.json (matches the hub exactly for gpt-oss).
  const b = paramBreakdown(c)
  const moeLayers = c.num_hidden_layers - (c.first_k_dense_replace ?? 0)
  let u8 = weightDtype(c, 'experts') === 'mxfp4' ? moeLayers * c.num_local_experts * expertMxfp4Params(c) : 0
  for (const r of b.rows) if (r.dtype === 'mxfp4' && r.id !== 'experts') u8 += r.total
  return ordered([['BF16', b.total - u8], ['U8', u8]])
}
