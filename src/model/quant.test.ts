import { describe, expect, it } from 'vitest'
import { GPT_OSS_120B, KIMI_K3, type GptOssConfig } from './config'
import { quantSummary, weightDtype } from './quant'

const withQuant = (c: GptOssConfig, q: GptOssConfig['quantization_config']): GptOssConfig => ({ ...c, quantization_config: q })

describe('weightDtype', () => {
  it('stores only gpt-oss experts in MXFP4, unless modules_to_not_convert names them', () => {
    expect(weightDtype(GPT_OSS_120B, 'experts')).toBe('mxfp4')
    for (const r of ['embed', 'attention', 'router', 'lm_head'] as const) expect(weightDtype(GPT_OSS_120B, r)).toBe('bf16')
    const kept = withQuant(GPT_OSS_120B, { quant_method: 'mxfp4', modules_to_not_convert: ['model.layers.*.mlp.experts'] })
    expect(weightDtype(kept, 'experts')).toBe('bf16')
    // A sibling path is not a submodule.
    const sibling = withQuant(GPT_OSS_120B, { quant_method: 'mxfp4', modules_to_not_convert: ['model.layers.*.mlp.expert'] })
    expect(weightDtype(sibling, 'experts')).toBe('mxfp4')
  })

  it('reads Kimi compressed-tensors: Linear targets minus the ignore regexes', () => {
    expect(weightDtype(KIMI_K3, 'experts')).toBe('mxfp4')
    for (const r of ['embed', 'attention', 'router', 'shared_experts', 'dense_mlp', 'lm_head'] as const) {
      expect(weightDtype(KIMI_K3, r)).toBe('bf16')
    }
    const q = KIMI_K3.quantization_config
    const noAttnIgnore = withQuant(KIMI_K3, { ...q, ignore: q.ignore!.filter((e) => !e.includes('self_attn')) })
    expect(weightDtype(noAttnIgnore, 'attention')).toBe('mxfp4')
    // Non-Linear modules are never targeted by a Linear group.
    expect(weightDtype(noAttnIgnore, 'router')).toBe('bf16')
  })

  it('keeps everything bf16 without an MXFP4 scheme', () => {
    const none = withQuant(KIMI_K3, { quant_method: 'compressed-tensors', format: 'float-quantized', config_groups: { g: { format: 'float-quantized', targets: ['Linear'] } } })
    expect(weightDtype(none, 'experts')).toBe('bf16')
    expect(quantSummary(none).scheme).toBeNull()
  })
})

describe('quantSummary', () => {
  it('splits the roles a model has and reports unquantized activations and KV cache', () => {
    const k = quantSummary(KIMI_K3)
    expect(k.method).toBe('compressed-tensors')
    expect(k.mxfp4).toEqual(['experts'])
    expect(k.bf16).toEqual(['embed', 'attention', 'dense_mlp', 'router', 'shared_experts', 'lm_head'])
    expect(k.activations).toBe('bf16')
    expect(k.kvCache).toBe('bf16')
    expect(k.scheme).toBe('4-bit float, 8-bit scale / 32')
    const g = quantSummary(GPT_OSS_120B)
    expect(g.mxfp4).toEqual(['experts'])
    expect(g.bf16).toEqual(['embed', 'attention', 'router', 'lm_head'])
  })
})
