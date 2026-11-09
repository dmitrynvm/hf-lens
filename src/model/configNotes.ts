import type { GptOssConfig } from './config'
import { kvCacheBytes, layerKvCacheBytes } from './kvcache'
import { formatBytes, layerMemory } from './params'

export interface ConfigNote {
  /** Hover remark for the key, with numbers derived from this model's config. */
  note: (c: GptOssConfig) => string
  /** Read only while training; no effect on inference. */
  training?: true
}

const n = (x: number) => x.toLocaleString('en-US')

/** Remarks per config.json key path (`rope_scaling.factor`); keys without one show no remark. */
export const CONFIG_NOTES: Record<string, ConfigNote> = {
  architectures: { note: () => 'Transformers class that loads these weights' },
  attention_bias: {
    note: (c) => `q/k/v/o projections carry biases: ${n(c.num_attention_heads * c.head_dim + 2 * c.num_key_value_heads * c.head_dim + c.hidden_size)} params per block`,
  },
  attention_dropout: { note: () => 'Dropout on attention weights', training: true },
  eos_token_id: { note: () => 'Token id that ends generation' },
  experts_per_token: { note: (c) => `Same as num_experts_per_tok (${c.num_experts_per_tok})` },
  head_dim: {
    note: (c) => `Width of one attention head: q ${c.num_attention_heads} × ${c.head_dim} = ${n(c.num_attention_heads * c.head_dim)}, k and v ${c.num_key_value_heads} × ${c.head_dim} = ${n(c.num_key_value_heads * c.head_dim)} each`,
  },
  hidden_act: { note: (c) => `Gate activation in each expert's SwiGLU, clamped at ${c.swiglu_limit}` },
  hidden_size: { note: (c) => `Residual stream width d: every block reads and writes [T, ${c.hidden_size}]` },
  initial_context_length: {
    note: (c) => `Pretraining context; YaRN stretches it ×${c.rope_scaling.factor} to ${n(c.max_position_embeddings)}`,
  },
  initializer_range: { note: () => 'Std of the weight initialisation', training: true },
  intermediate_size: {
    note: (c) => `Expert hidden width I: gate_up ${c.hidden_size} → ${2 * c.intermediate_size}, down ${c.intermediate_size} → ${c.hidden_size}`,
  },
  layer_types: {
    note: (c) => {
      const sliding = c.layer_types.filter((t) => t === 'sliding_attention').length
      return `${sliding} sliding (last ${c.sliding_window} tokens), ${c.layer_types.length - sliding} full, alternating`
    },
  },
  max_position_embeddings: {
    note: (c) => `Longest context: ${n(c.max_position_embeddings)} tokens, with a ${formatBytes(kvCacheBytes(c, c.max_position_embeddings))} KV cache per sequence`,
  },
  model_type: { note: () => 'Transformers model family' },
  num_attention_heads: {
    note: (c) => `Query heads per block; ${c.num_attention_heads / c.num_key_value_heads} share each KV head (GQA)`,
  },
  num_experts_per_tok: {
    note: (c) => `Experts a token runs per block: ${c.num_experts_per_tok} of ${c.num_local_experts} (${(100 * c.num_experts_per_tok / c.num_local_experts).toFixed(1)}%)`,
  },
  num_hidden_layers: { note: (c) => `Decoder blocks, ${formatBytes(layerMemory(c).bytes)} of weights each` },
  num_key_value_heads: {
    note: (c) => `KV heads per block: the cache grows ${formatBytes(layerKvCacheBytes(c, 'full_attention', 1))} per token per full layer`,
  },
  num_local_experts: {
    note: (c) => {
      const m = layerMemory(c)
      const ex = m.rows.find((r) => r.id === 'experts')!
      return `Experts per MoE block: ${formatBytes(ex.bytes)} of the block's ${formatBytes(m.bytes)} (${(100 * ex.bytes / m.bytes).toFixed(1)}%)`
    },
  },
  output_router_logits: { note: () => 'Return router logits for the auxiliary loss', training: true },
  pad_token_id: { note: () => 'Token id used for padding' },
  quantization_config: { note: () => 'Experts in MXFP4: 4-bit values with one 8-bit scale per 32, 4.25 bits/param' },
  'quantization_config.modules_to_not_convert': { note: () => 'Stay bf16: attention, router, embedding, lm_head' },
  'quantization_config.quant_method': { note: () => 'Storage format of the expert weights' },
  rms_norm_eps: { note: () => 'Added under the square root in every RMSNorm' },
  rope_scaling: {
    note: (c) => `YaRN: RoPE trained on ${n(c.rope_scaling.original_max_position_embeddings)} positions, stretched to ${n(c.max_position_embeddings)}`,
  },
  'rope_scaling.beta_fast': { note: () => 'Frequencies turning more than this many times over the original context keep their rotation' },
  'rope_scaling.beta_slow': { note: () => 'Frequencies turning fewer than this many times are fully interpolated' },
  'rope_scaling.factor': { note: (c) => `Context stretch: ${n(c.rope_scaling.original_max_position_embeddings)} × ${c.rope_scaling.factor} = ${n(c.max_position_embeddings)}` },
  'rope_scaling.original_max_position_embeddings': { note: () => 'Positions RoPE was trained on' },
  'rope_scaling.rope_type': { note: () => 'Context-extension method' },
  'rope_scaling.truncate': { note: () => 'Whether YaRN rounds its frequency ramp to whole dimensions' },
  rope_theta: { note: () => 'RoPE base: sets the slowest rotation frequency' },
  router_aux_loss_coef: { note: () => 'Weight of the load-balancing loss', training: true },
  sliding_window: {
    note: (c) => `Tokens a sliding layer sees; its KV cache stops at ${formatBytes(layerKvCacheBytes(c, 'sliding_attention', c.max_position_embeddings))}`,
  },
  swiglu_limit: { note: (c) => `Clamp before SwiGLU: gate ≤ ${c.swiglu_limit}, linear in [−${c.swiglu_limit}, ${c.swiglu_limit}]` },
  tie_word_embeddings: {
    note: (c) => (c.tie_word_embeddings ? 'lm_head reuses the embedding table' : `lm_head is its own ${formatBytes(c.vocab_size * c.hidden_size * 2)} matrix`),
  },
  transformers_version: { note: () => 'Transformers version that wrote the file' },
  use_cache: { note: () => 'Generation keeps a KV cache' },
  vocab_size: { note: (c) => `Tokens: embedding rows and lm_head outputs, ${formatBytes(c.vocab_size * c.hidden_size * 2)} each in bf16` },
}
