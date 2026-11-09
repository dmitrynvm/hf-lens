import type { GptOssConfig } from './config'
import type { Dtype } from './params'

/**
 * Which weights a checkpoint stores in MXFP4, read from its `quantization_config`. Two shapes are understood:
 * - transformers MXFP4 (gpt-oss): `quant_method: "mxfp4"` converts the MoE expert modules only, minus `modules_to_not_convert`;
 * - compressed-tensors (Kimi): a config group with `format: "mxfp4-pack-quantized"` quantizes every module its `targets`
 *   name (here the `Linear` class), minus the `ignore` list (`re:` regexes or exact names).
 * Anything else stores every weight in bf16. Norms, biases, sinks and convolutions are never quantized.
 */
export type WeightRole = 'embed' | 'attention' | 'router' | 'experts' | 'shared_experts' | 'dense_mlp' | 'lm_head'

/** One checkpoint module per role: its state-dict path and whether it is an `nn.Linear` (compressed-tensors targets classes). */
const MODULES: Record<WeightRole, { name: string; linear: boolean }> = {
  embed: { name: 'model.embed_tokens', linear: false },
  attention: { name: 'model.layers.0.self_attn.q_proj', linear: true },
  // gpt-oss `mlp.router` and Kimi's `mlp.gate` are custom modules holding a weight, not `nn.Linear`.
  router: { name: 'model.layers.0.mlp.gate', linear: false },
  experts: { name: 'model.layers.0.mlp.experts.0.gate_proj', linear: true },
  shared_experts: { name: 'model.layers.0.mlp.shared_experts.gate_proj', linear: true },
  dense_mlp: { name: 'model.layers.0.mlp.gate_proj', linear: true },
  lm_head: { name: 'lm_head', linear: true },
}

export const ROLE_LABELS: Record<WeightRole, string> = {
  embed: 'embedding',
  attention: 'attention',
  router: 'router',
  experts: 'routed experts',
  shared_experts: 'shared experts',
  dense_mlp: 'dense MLP',
  lm_head: 'lm_head',
}

type Group = NonNullable<GptOssConfig['quantization_config']['config_groups']>[string]

const MXFP4_FORMAT = 'mxfp4-pack-quantized'

/** The compressed-tensors config group stored as MXFP4, if any. */
function mxfp4Group(q: GptOssConfig['quantization_config']): Group | undefined {
  if (q.quant_method !== 'compressed-tensors') return undefined
  return Object.values(q.config_groups ?? {}).find((g) => (g.format ?? q.format) === MXFP4_FORMAT)
}

/** transformers `modules_to_not_convert`: dotted paths, `*` for any segment run; a pattern covers its submodules. */
const notConverted = (patterns: readonly string[], name: string) =>
  patterns.some((p) => new RegExp(`^${p.split('*').map((s) => s.replace(/[.+?^${}()|[\]\\]/g, '\\$&')).join('.*')}(\\.|$)`).test(name))

/** compressed-tensors `ignore` / `targets`: `re:` entries are regexes matched from the start of the module path, others exact paths or class names. */
const listed = (entries: readonly string[], name: string, cls: string) =>
  entries.some((e) => (e.startsWith('re:') ? new RegExp(`^(?:${e.slice(3)})`).test(name) : e === name || e === cls))

/** Storage dtype of a role's weight matrices, as `quantization_config` prescribes. */
export function weightDtype(c: GptOssConfig, role: WeightRole): Dtype {
  const q = c.quantization_config
  if (q.quant_method === 'mxfp4') {
    return role === 'experts' && !notConverted(q.modules_to_not_convert ?? [], 'model.layers.0.mlp.experts') ? 'mxfp4' : 'bf16'
  }
  const g = mxfp4Group(q)
  if (!g) return 'bf16'
  const { name, linear } = MODULES[role]
  const cls = linear ? 'Linear' : ''
  return listed(g.targets ?? [], name, cls) && !listed(q.ignore ?? [], name, cls) ? 'mxfp4' : 'bf16'
}

export interface QuantSummary {
  /** `quant_method` as written, e.g. `mxfp4` or `compressed-tensors`. */
  method: string
  /** MXFP4 layout from the config, e.g. `4-bit float, 8-bit scale / 32`; null when nothing is quantized. */
  scheme: string | null
  mxfp4: WeightRole[]
  bf16: WeightRole[]
  activations: string
  kvCache: string
}

const fmt = (s: { num_bits?: number; type?: string } | null | undefined) => (s ? `${s.type ?? ''}${s.num_bits ?? ''}` : 'bf16')

/** The roles this model has, split by storage dtype, plus activation and KV-cache formats. */
export function quantSummary(c: GptOssConfig): QuantSummary {
  const q = c.quantization_config
  const roles: WeightRole[] = ['embed', 'attention']
  if ((c.first_k_dense_replace ?? 0) > 0) roles.push('dense_mlp')
  if ((c.first_k_dense_replace ?? 0) < c.num_hidden_layers) roles.push('router', 'experts')
  if (c.num_shared_experts) roles.push('shared_experts')
  roles.push('lm_head')
  const g = mxfp4Group(q)
  const w = g?.weights
  const mxfp4 = roles.filter((r) => weightDtype(c, r) === 'mxfp4')
  return {
    method: q.quant_method,
    scheme: mxfp4.length === 0
      ? null
      : w
        ? `${w.num_bits ?? 4}-bit ${w.type ?? 'float'}, ${w.scale_dtype === 'torch.uint8' ? '8-bit' : (w.scale_dtype ?? '8-bit')} scale / ${w.group_size ?? 32}`
        : '4-bit float, 8-bit scale / 32',
    mxfp4,
    bf16: roles.filter((r) => !mxfp4.includes(r)),
    activations: fmt(g?.input_activations),
    kvCache: fmt(q.kv_cache_scheme),
  }
}

/** One line per storage dtype, as `quantization_config` assigns the model's weights. */
export function quantText(s: QuantSummary): string {
  const roles = (rs: WeightRole[]) => rs.map((r) => ROLE_LABELS[r]).join(', ')
  const mx = s.mxfp4.length ? `MXFP4 ${roles(s.mxfp4)} (${s.scheme})` : null
  const bf = s.bf16.length ? `bf16 ${roles(s.bf16)}` : null
  return [mx, bf, `activations ${s.activations}, KV cache ${s.kvCache}`].filter(Boolean).join(' / ')
}
