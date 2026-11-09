import type { GptOssConfig } from './config'
import { mxfp4Bytes, type Dtype } from './params'

export interface TensorSpec {
  name: string
  shape: number[]
  dtype: Dtype
  group: 'norm' | 'attention' | 'router' | 'experts' | 'embed' | 'lm_head'
}

export function layerTensors(c: GptOssConfig, i: number): TensorSpec[] {
  const H = c.hidden_size
  const I = c.intermediate_size
  const E = c.num_local_experts
  const q = c.num_attention_heads * c.head_dim
  const kv = c.num_key_value_heads * c.head_dim
  const p = `model.layers.${i}`
  const t = (name: string, shape: number[], group: TensorSpec['group'], dtype: Dtype = 'bf16'): TensorSpec => ({
    name: `${p}.${name}`, shape, dtype, group,
  })
  return [
    t('input_layernorm.weight', [H], 'norm'),
    t('self_attn.q_proj.weight', [q, H], 'attention'),
    t('self_attn.q_proj.bias', [q], 'attention'),
    t('self_attn.k_proj.weight', [kv, H], 'attention'),
    t('self_attn.k_proj.bias', [kv], 'attention'),
    t('self_attn.v_proj.weight', [kv, H], 'attention'),
    t('self_attn.v_proj.bias', [kv], 'attention'),
    t('self_attn.o_proj.weight', [H, q], 'attention'),
    t('self_attn.o_proj.bias', [H], 'attention'),
    t('self_attn.sinks', [c.num_attention_heads], 'attention'),
    t('post_attention_layernorm.weight', [H], 'norm'),
    t('mlp.router.weight', [E, H], 'router'),
    t('mlp.router.bias', [E], 'router'),
    t('mlp.experts.gate_up_proj', [E, H, 2 * I], 'experts', 'mxfp4'),
    t('mlp.experts.gate_up_proj_bias', [E, 2 * I], 'experts'),
    t('mlp.experts.down_proj', [E, I, H], 'experts', 'mxfp4'),
    t('mlp.experts.down_proj_bias', [E, H], 'experts'),
  ]
}

export function globalTensors(c: GptOssConfig): TensorSpec[] {
  const H = c.hidden_size
  const V = c.vocab_size
  return [
    { name: 'model.embed_tokens.weight', shape: [V, H], dtype: 'bf16', group: 'embed' },
    { name: 'model.norm.weight', shape: [H], dtype: 'bf16', group: 'norm' },
    { name: 'lm_head.weight', shape: [V, H], dtype: 'bf16', group: 'lm_head' },
  ]
}

/** Elements in the tensor. */
export const tensorParams = (t: TensorSpec) => t.shape.reduce((a, b) => a * b, 1)
/** Bytes on disk: bf16 2 B per element, MXFP4 4.25 bits (see `mxfp4Bytes`). */
export const tensorBytes = (t: TensorSpec) => (t.dtype === 'mxfp4' ? mxfp4Bytes(tensorParams(t)) : tensorParams(t) * 2)
