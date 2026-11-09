import { describe, expect, it } from 'vitest'
import { GPT_OSS_120B as c } from './config'
import { kvCacheBytes, layerKvCacheBytes } from './kvcache'

describe('kvCacheBytes', () => {
  it('caches only the window on sliding layers at full context', () => {
    expect(kvCacheBytes(c, 131072)).toBe(4_836_556_800)
    expect(kvCacheBytes(c, 131072, { useSliding: false })).toBe(9_663_676_416)
    expect(kvCacheBytes(c, 131072, { kvHeads: c.num_attention_heads, useSliding: false })).toBe(77_309_411_328)
  })

  it('makes sliding irrelevant when the context fits in the window', () => {
    expect(kvCacheBytes(c, c.sliding_window)).toBe(kvCacheBytes(c, c.sliding_window, { useSliding: false }))
    expect(kvCacheBytes(c, c.sliding_window + 1)).toBeLessThan(kvCacheBytes(c, c.sliding_window + 1, { useSliding: false }))
  })

  it('caps a sliding layer at the window while a full layer grows with context', () => {
    const perToken = 2 * c.num_key_value_heads * c.head_dim * 2
    expect(layerKvCacheBytes(c, 'sliding_attention', 100)).toBe(100 * perToken)
    expect(layerKvCacheBytes(c, 'sliding_attention', 131072)).toBe(c.sliding_window * perToken)
    expect(layerKvCacheBytes(c, 'full_attention', 131072)).toBe(131072 * perToken)
  })
})
