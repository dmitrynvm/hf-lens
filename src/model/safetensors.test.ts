import { describe, expect, it } from 'vitest'
import { BUILTIN_MODELS } from './config'
import { tensorTypes } from './safetensors'

const builtin = (id: string) => BUILTIN_MODELS.find((m) => m.id === id)!

describe('tensorTypes', () => {
  it.each([
    ['120b', [['BF16', 2167371072], ['U8', 114661785600]]],
    ['20b', [['BF16', 1804459584], ['U8', 19110297600]]],
  ] as const)('computes gpt-oss-%s from config.json exactly as the hub reports it', (id, rows) => {
    expect(tensorTypes({ ...builtin(id), safetensors: null }).map((r) => [r.dtype, r.params])).toEqual(rows)
  })

  it('lists BF16 first, then the hub order, mapping only BF16 and U8 to diagram tensors', () => {
    expect(tensorTypes(builtin('kimi-k3')).map((r) => [r.dtype, r.params, r.storage])).toEqual([
      ['BF16', 57179884544, 'bf16'], ['F32', 11122432, null], ['U8', 2722740830208, 'mxfp4'],
    ])
  })

  it('sizes each type on disk: BF16 2 B, F32 4 B, U8 holding MXFP4 at 4.25 bits per param', () => {
    expect(tensorTypes(builtin('kimi-k3')).map((r) => r.bytes)).toEqual([57179884544 * 2, 11122432 * 4, (2722740830208 * 17) / 32])
  })

  it('says which weights each type holds and in which layers', () => {
    const [bf16, u8] = tensorTypes(builtin('120b'))
    expect(u8.where).toEqual(['routed experts · L0–L35'])
    expect(bf16.where).toEqual([
      'embedding · input', 'attention · L0–L35', 'router · L0–L35', 'lm_head · output',
      'expert biases · L0–L35', 'RMSNorms · L0–L35 + final',
    ])
  })
})
