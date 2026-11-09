import { describe, expect, it } from 'vitest'
import { GPT_OSS_120B as c } from './config'
import { attentionParams, expertMxfp4Params, expertParams, routerParams } from './params'
import { globalTensors, layerTensors } from './tensors'

const product = (shape: number[]) => shape.reduce((a, b) => a * b, 1)
const total = (specs: { shape: number[] }[]) => specs.reduce((a, t) => a + product(t.shape), 0)

describe('layerTensors', () => {
  it.each([0, 35])('layer %i lists every tensor and sums to the per-layer param count', (i) => {
    const tensors = layerTensors(c, i)
    expect(tensors).toHaveLength(17)
    for (const t of tensors) expect(t.name.startsWith(`model.layers.${i}.`)).toBe(true)
    expect(total(tensors)).toBe(
      attentionParams(c) + routerParams(c) + c.num_local_experts * expertParams(c) + 2 * c.hidden_size,
    )
  })

  it('marks exactly the two expert projection weights as mxfp4', () => {
    const mx = layerTensors(c, 0).filter((t) => t.dtype === 'mxfp4')
    expect(mx.map((t) => t.name.split('.').at(-1))).toEqual(['gate_up_proj', 'down_proj'])
    expect(total(mx)).toBe(c.num_local_experts * expertMxfp4Params(c))
  })
})

describe('globalTensors', () => {
  it('covers embedding, final norm and lm_head', () => {
    const g = globalTensors(c)
    expect(g.map((t) => t.group)).toEqual(['embed', 'norm', 'lm_head'])
    expect(total(g)).toBe(2 * c.vocab_size * c.hidden_size + c.hidden_size)
  })
})
