import type { GptOssConfig, LayerType } from './config'

const BF16_BYTES = 2

export interface KvCacheOptions {
  /** KV heads per layer; pass num_attention_heads to model plain multi-head attention. */
  kvHeads?: number
  /** When false, sliding layers cache the full context like full-attention layers. */
  useSliding?: boolean
}

/** bf16 K+V (or KDA state / MLA latent) cache of one layer for one sequence of `ctx` tokens. */
export function layerKvCacheBytes(c: GptOssConfig, type: LayerType, ctx: number, opts: KvCacheOptions = {}): number {
  const { kvHeads = c.num_key_value_heads, useSliding = true } = opts
  if (type === 'linear_attention') {
    if (useSliding) {
      if (ctx <= 0) return 0
      const conv = c.short_conv_kernel_size ?? 4
      const stateBytes = c.num_attention_heads * c.head_dim * (c.head_dim + 3 * conv) * BF16_BYTES
      return Math.min(stateBytes, 2 * kvHeads * c.head_dim * BF16_BYTES * ctx)
    }
    return 2 * kvHeads * c.head_dim * BF16_BYTES * ctx
  }
  if (type === 'full_attention' && c.kv_lora_rank !== undefined && useSliding && opts.kvHeads === undefined) {
    const rope = c.qk_rope_head_dim ?? 64
    return (c.kv_lora_rank + rope) * BF16_BYTES * ctx
  }
  const cached = useSliding && type === 'sliding_attention' ? Math.min(ctx, c.sliding_window) : ctx
  return 2 * kvHeads * c.head_dim * BF16_BYTES * cached
}

/** bf16 K+V cache for one sequence of `ctx` tokens across all layers. */
export function kvCacheBytes(c: GptOssConfig, ctx: number, opts: KvCacheOptions = {}): number {
  return c.layer_types.reduce((sum, t) => sum + layerKvCacheBytes(c, t, ctx, opts), 0)
}
