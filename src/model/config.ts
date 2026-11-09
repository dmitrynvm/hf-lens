import config120b from './gpt-oss-120b.config.json'
import text120b from './gpt-oss-120b.config.json?raw'
import config20b from './gpt-oss-20b.config.json'
import text20b from './gpt-oss-20b.config.json?raw'
import configKimiK3 from './kimi-k3.config.json'
import textKimiK3 from './kimi-k3.config.json?raw'

export type LayerType = 'sliding_attention' | 'full_attention' | 'linear_attention'

/** The keys of config.json this app computes from; the files carry more (see `ModelEntry.json`). */
export interface GptOssConfig {
  model_type?: 'gpt_oss' | 'kimi_k3' | 'kimi_linear'
  hidden_size: number
  intermediate_size: number
  first_k_dense_replace?: number
  dense_intermediate_size?: number
  num_shared_experts?: number
  routed_expert_hidden_size?: number
  latent_moe_use_norm?: boolean
  hidden_act?: string
  activation_situ_beta?: number
  activation_situ_linear_beta?: number
  q_lora_rank?: number
  kv_lora_rank?: number
  qk_nope_head_dim?: number
  qk_rope_head_dim?: number
  v_head_dim?: number
  mla_use_output_gate?: boolean
  short_conv_kernel_size?: number
  attn_res_block_size?: number
  num_hidden_layers: number
  num_attention_heads: number
  num_key_value_heads: number
  head_dim: number
  num_local_experts: number
  num_experts_per_tok: number
  vocab_size: number
  sliding_window: number
  max_position_embeddings: number
  rope_theta: number
  rope_scaling: {
    rope_type: string
    factor: number
    original_max_position_embeddings: number
    beta_fast: number
    beta_slow: number
  }
  swiglu_limit: number
  rms_norm_eps: number
  attention_bias: boolean
  tie_word_embeddings: boolean
  layer_types: LayerType[]
  /** As written in config.json; `quant.ts` reads both shapes (transformers MXFP4 and compressed-tensors). */
  quantization_config: {
    quant_method: string
    /** transformers MXFP4: modules left in bf16. */
    modules_to_not_convert?: string[]
    /** compressed-tensors fields. */
    format?: string
    ignore?: string[]
    config_groups?: Record<string, {
      format?: string
      targets?: string[]
      weights?: { num_bits?: number; type?: string; group_size?: number; scale_dtype?: string } | null
      input_activations?: { num_bits?: number; type?: string } | null
    }>
    kv_cache_scheme?: { num_bits?: number; type?: string } | null
  }
}

export function normalizeKimiConfig(raw: Record<string, unknown>): GptOssConfig {
  const t = (raw.text_config !== null && typeof raw.text_config === 'object' ? raw.text_config : raw) as Record<string, unknown>
  const lac = (t.linear_attn_config !== null && typeof t.linear_attn_config === 'object' ? t.linear_attn_config : {}) as Record<string, unknown>
  const L = Number(t.num_hidden_layers)
  const kdaSet = new Set(Array.isArray(lac.kda_layers) ? (lac.kda_layers as number[]) : [])
  const layer_types: LayerType[] = Array.from({ length: L }, (_, i) =>
    kdaSet.has(i + 1) ? 'linear_attention' : 'full_attention',
  )
  const nh = Number(t.num_attention_heads)
  const headDim = typeof lac.head_dim === 'number'
    ? lac.head_dim
    : typeof t.v_head_dim === 'number'
      ? t.v_head_dim
      : Number(t.hidden_size) / nh
  const maxPos = Number(t.max_position_embeddings)
  return {
    model_type: raw.model_type === 'kimi_k3' ? 'kimi_k3' : 'kimi_linear',
    hidden_size: Number(t.hidden_size),
    intermediate_size: Number(t.moe_intermediate_size),
    dense_intermediate_size: Number(t.intermediate_size),
    first_k_dense_replace: typeof t.first_k_dense_replace === 'number' ? t.first_k_dense_replace : 0,
    num_hidden_layers: L,
    num_attention_heads: nh,
    num_key_value_heads: typeof t.num_key_value_heads === 'number' ? t.num_key_value_heads : nh,
    head_dim: headDim,
    num_local_experts: Number(t.num_experts),
    num_experts_per_tok: Number(t.num_experts_per_token),
    num_shared_experts: typeof t.num_shared_experts === 'number' ? t.num_shared_experts : undefined,
    routed_expert_hidden_size: typeof t.routed_expert_hidden_size === 'number' ? t.routed_expert_hidden_size : undefined,
    latent_moe_use_norm: typeof t.latent_moe_use_norm === 'boolean' ? t.latent_moe_use_norm : undefined,
    hidden_act: typeof t.hidden_act === 'string' ? t.hidden_act : undefined,
    activation_situ_beta: typeof t.activation_situ_beta === 'number' ? t.activation_situ_beta : undefined,
    activation_situ_linear_beta: typeof t.activation_situ_linear_beta === 'number' ? t.activation_situ_linear_beta : undefined,
    q_lora_rank: typeof t.q_lora_rank === 'number' ? t.q_lora_rank : undefined,
    kv_lora_rank: typeof t.kv_lora_rank === 'number' ? t.kv_lora_rank : undefined,
    qk_nope_head_dim: typeof t.qk_nope_head_dim === 'number' ? t.qk_nope_head_dim : undefined,
    qk_rope_head_dim: typeof t.qk_rope_head_dim === 'number' ? t.qk_rope_head_dim : undefined,
    v_head_dim: typeof t.v_head_dim === 'number' ? t.v_head_dim : undefined,
    mla_use_output_gate: typeof t.mla_use_output_gate === 'boolean' ? t.mla_use_output_gate : undefined,
    short_conv_kernel_size: typeof lac.short_conv_kernel_size === 'number' ? lac.short_conv_kernel_size : undefined,
    attn_res_block_size: typeof t.attn_res_block_size === 'number' ? t.attn_res_block_size : undefined,
    vocab_size: Number(t.vocab_size),
    sliding_window: 128,
    max_position_embeddings: maxPos,
    rope_theta: typeof t.rope_theta === 'number' ? t.rope_theta : 10000,
    rope_scaling: {
      rope_type: 'none',
      factor: 1,
      original_max_position_embeddings: maxPos,
      beta_fast: 32,
      beta_slow: 1,
    },
    swiglu_limit: typeof t.activation_situ_linear_beta === 'number' ? t.activation_situ_linear_beta : 7,
    rms_norm_eps: Number(t.rms_norm_eps),
    attention_bias: false,
    tie_word_embeddings: Boolean(t.tie_word_embeddings ?? raw.tie_word_embeddings),
    layer_types,
    quantization_config: (t.quantization_config ?? { quant_method: 'none' }) as GptOssConfig['quantization_config'],
  }
}

export const GPT_OSS_120B = config120b as GptOssConfig
export const GPT_OSS_20B = config20b as GptOssConfig
export const KIMI_K3 = normalizeKimiConfig(configKimiK3 as Record<string, unknown>)
/** The hub's safetensors summary for a repo (`/api/models/<repo>?expand[]=safetensors`): parameter count per stored dtype. */
export interface SafetensorsInfo { parameters: Record<string, number>; total: number }
/** A checkpoint the app can show: its config.json parsed, typed and verbatim. */
export interface ModelEntry {
  /** Hash and storage key: '120b' / '20b' for the bundled checkpoints, the Hugging Face repo id for added ones. */
  id: string
  /** Hugging Face repo id, e.g. `openai/gpt-oss-120b`. */
  repo: string
  /** Short name for the model picker and diffs. */
  name: string
  config: GptOssConfig
  json: Record<string, unknown>
  /** config.json as published, for the Config tab. */
  text: string
  /** As the hub reports it; null when it could not be fetched (the card then computes it from config.json). */
  safetensors: SafetensorsInfo | null
}

/** Verbatim https://huggingface.co/openai/gpt-oss-{120b,20b}/raw/main/config.json, parsed and as text. */
export const BUILTIN_MODELS: readonly ModelEntry[] = [
  {
    id: '120b', repo: 'openai/gpt-oss-120b', name: 'gpt-oss-120b', config: GPT_OSS_120B, json: config120b, text: text120b,
    safetensors: { parameters: { BF16: 2167371072, U8: 114661785600 }, total: 116829156672 },
  },
  {
    id: '20b', repo: 'openai/gpt-oss-20b', name: 'gpt-oss-20b', config: GPT_OSS_20B, json: config20b, text: text20b,
    safetensors: { parameters: { BF16: 1804459584, U8: 19110297600 }, total: 20914757184 },
  },
  {
    id: 'kimi-k3', repo: 'moonshotai/Kimi-K3', name: 'Kimi-K3', config: KIMI_K3, json: configKimiK3, text: textKimiK3,
    safetensors: { parameters: { F32: 11122432, BF16: 57179884544, U8: 2722740830208 }, total: 2779931837184 },
  },
]
export const DEFAULT_MODEL = '120b'

/** The model with this id, if the list has it. */
export const findModel = (models: readonly ModelEntry[], id: string | null) => models.find((m) => m.id === id)
