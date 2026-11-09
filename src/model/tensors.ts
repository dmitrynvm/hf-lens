import type { GptOssConfig } from './config'
import { mxfp4Bytes, type Dtype } from './params'
import { weightDtype } from './quant'

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
  const p = `model.layers.${i}`
  const t = (name: string, shape: number[], group: TensorSpec['group'], dtype: Dtype = 'bf16'): TensorSpec => ({
    name: `${p}.${name}`, shape, dtype, group,
  })
  // Weight matrices take their role's dtype from quantization_config; biases, norms, convolutions and sinks stay bf16.
  const attn = weightDtype(c, 'attention')
  const router = weightDtype(c, 'router')
  const experts = weightDtype(c, 'experts')
  if (c.routed_expert_hidden_size !== undefined) {
    const out: TensorSpec[] = [t('input_layernorm.weight', [H], 'norm')]
    if (c.attn_res_block_size) {
      out.push(t('attn_res_norm.weight', [H], 'norm'), t('attn_res_proj.weight', [H], 'norm'))
    }
    if (c.layer_types[i] === 'linear_attention') {
      const P = c.num_attention_heads * c.head_dim
      const conv = c.short_conv_kernel_size ?? 4
      out.push(
        t('self_attn.q_proj.weight', [P, H], 'attention', attn),
        t('self_attn.k_proj.weight', [P, H], 'attention', attn),
        t('self_attn.v_proj.weight', [P, H], 'attention', attn),
        t('self_attn.q_conv1d.weight', [P, 1, conv], 'attention'),
        t('self_attn.k_conv1d.weight', [P, 1, conv], 'attention'),
        t('self_attn.v_conv1d.weight', [P, 1, conv], 'attention'),
        t('self_attn.f_a_proj.weight', [c.head_dim, H], 'attention', attn),
        t('self_attn.f_b_proj.weight', [P, c.head_dim], 'attention', attn),
        t('self_attn.b_proj.weight', [c.num_attention_heads, H], 'attention', attn),
        t('self_attn.A_log', [c.num_attention_heads], 'attention'),
        t('self_attn.dt_bias', [P], 'attention'),
        t('self_attn.g_proj.weight', [P, H], 'attention', attn),
        t('self_attn.o_norm.weight', [c.head_dim], 'attention'),
        t('self_attn.o_proj.weight', [H, P], 'attention', attn),
      )
    } else {
      const nh = c.num_attention_heads
      const nope = c.qk_nope_head_dim ?? 128
      const rope = c.qk_rope_head_dim ?? 64
      const vHead = c.v_head_dim ?? c.head_dim
      const qRank = c.q_lora_rank ?? 1536
      const kvRank = c.kv_lora_rank ?? 512
      out.push(
        t('self_attn.q_a_proj.weight', [qRank, H], 'attention', attn),
        t('self_attn.q_a_layernorm.weight', [qRank], 'attention'),
        t('self_attn.q_b_proj.weight', [nh * (nope + rope), qRank], 'attention', attn),
        t('self_attn.kv_a_proj_with_mqa.weight', [kvRank + rope, H], 'attention', attn),
        t('self_attn.kv_a_layernorm.weight', [kvRank], 'attention'),
        t('self_attn.kv_b_proj.weight', [nh * (nope + vHead), kvRank], 'attention', attn),
      )
      if (c.mla_use_output_gate) out.push(t('self_attn.g_proj.weight', [nh * vHead, H], 'attention', attn))
      out.push(t('self_attn.o_proj.weight', [H, nh * vHead], 'attention', attn))
    }
    out.push(t('post_attention_layernorm.weight', [H], 'norm'))
    if (c.attn_res_block_size) {
      out.push(t('mlp_res_norm.weight', [H], 'norm'), t('mlp_res_proj.weight', [H], 'norm'))
    }
    if (i < (c.first_k_dense_replace ?? 0)) {
      const D = c.dense_intermediate_size ?? I
      const dense = weightDtype(c, 'dense_mlp')
      out.push(
        t('mlp.gate_proj.weight', [D, H], 'experts', dense),
        t('mlp.up_proj.weight', [D, H], 'experts', dense),
        t('mlp.down_proj.weight', [H, D], 'experts', dense),
      )
    } else {
      const R = c.routed_expert_hidden_size
      const S = (c.num_shared_experts ?? 1) * I
      out.push(
        t('mlp.gate.weight', [E, H], 'router', router),
        t('mlp.gate.e_score_correction_bias', [E], 'router'),
        t('mlp.fc_latent_down.weight', [R, H], 'experts'),
        t('mlp.experts.w1', [E, I, R], 'experts', experts),
        t('mlp.experts.w3', [E, I, R], 'experts', experts),
        t('mlp.experts.w2', [E, R, I], 'experts', experts),
      )
      if (c.latent_moe_use_norm) out.push(t('mlp.latent_out_norm.weight', [R], 'experts'))
      const shared = weightDtype(c, 'shared_experts')
      out.push(
        t('mlp.fc_latent_up.weight', [H, R], 'experts'),
        t('mlp.shared_experts.gate_proj.weight', [S, H], 'experts', shared),
        t('mlp.shared_experts.up_proj.weight', [S, H], 'experts', shared),
        t('mlp.shared_experts.down_proj.weight', [H, S], 'experts', shared),
      )
    }
    return out
  }
  const q = c.num_attention_heads * c.head_dim
  const kv = c.num_key_value_heads * c.head_dim
  return [
    t('input_layernorm.weight', [H], 'norm'),
    t('self_attn.q_proj.weight', [q, H], 'attention', attn),
    t('self_attn.q_proj.bias', [q], 'attention'),
    t('self_attn.k_proj.weight', [kv, H], 'attention', attn),
    t('self_attn.k_proj.bias', [kv], 'attention'),
    t('self_attn.v_proj.weight', [kv, H], 'attention', attn),
    t('self_attn.v_proj.bias', [kv], 'attention'),
    t('self_attn.o_proj.weight', [H, q], 'attention', attn),
    t('self_attn.o_proj.bias', [H], 'attention'),
    t('self_attn.sinks', [c.num_attention_heads], 'attention'),
    t('post_attention_layernorm.weight', [H], 'norm'),
    t('mlp.router.weight', [E, H], 'router', router),
    t('mlp.router.bias', [E], 'router'),
    t('mlp.experts.gate_up_proj', [E, H, 2 * I], 'experts', experts),
    t('mlp.experts.gate_up_proj_bias', [E, 2 * I], 'experts'),
    t('mlp.experts.down_proj', [E, I, H], 'experts', experts),
    t('mlp.experts.down_proj_bias', [E, H], 'experts'),
  ]
}

export function globalTensors(c: GptOssConfig): TensorSpec[] {
  const H = c.hidden_size
  const V = c.vocab_size
  return [
    { name: 'model.embed_tokens.weight', shape: [V, H], dtype: weightDtype(c, 'embed'), group: 'embed' },
    { name: 'model.norm.weight', shape: [H], dtype: 'bf16', group: 'norm' },
    { name: 'lm_head.weight', shape: [V, H], dtype: weightDtype(c, 'lm_head'), group: 'lm_head' },
  ]
}

/** Elements in the tensor. */
export const tensorParams = (t: TensorSpec) => t.shape.reduce((a, b) => a * b, 1)
/** Bytes on disk: bf16 2 B per element, MXFP4 4.25 bits (see `mxfp4Bytes`). */
export const tensorBytes = (t: TensorSpec) => (t.dtype === 'mxfp4' ? mxfp4Bytes(tensorParams(t)) : tensorParams(t) * 2)
