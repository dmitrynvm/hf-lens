import type { GptOssConfig, LayerType } from './config'

const BF16_BYTES = 2

export interface KvCacheOptions {
  /** KV heads per layer; pass num_attention_heads to model plain multi-head attention. */
  kvHeads?: number
  /** When false, sliding layers cache the full context like full-attention layers. */
  useSliding?: boolean
}

/** bf16 K+V cache of one layer for one sequence of `ctx` tokens; sliding layers keep only the last `sliding_window`. */
export function layerKvCacheBytes(c: GptOssConfig, type: LayerType, ctx: number, opts: KvCacheOptions = {}): number {
  const { kvHeads = c.num_key_value_heads, useSliding = true } = opts
  const cached = useSliding && type === 'sliding_attention' ? Math.min(ctx, c.sliding_window) : ctx
  return 2 * kvHeads * c.head_dim * BF16_BYTES * cached
}

/** bf16 K+V cache for one sequence of `ctx` tokens across all layers. */
export function kvCacheBytes(c: GptOssConfig, ctx: number, opts: KvCacheOptions = {}): number {
  return c.layer_types.reduce((sum, t) => sum + layerKvCacheBytes(c, t, ctx, opts), 0)
}
