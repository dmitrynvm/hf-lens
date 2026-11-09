import type { GptOssConfig } from './config'
import { formatBytes, formatCount, type ParamBreakdown, type ParamRow } from './params'

export interface HeaderFact {
  label: string
  value: string
  /** Hover text: what the number means and the sum it comes from, with this model's numbers. */
  title: string
}

const SHORT: Record<ParamRow['id'], string> = {
  embed: 'embedding', attention: 'attention', router: 'router', experts: 'experts', norms: 'norms', lm_head: 'lm_head',
}

/** The top bar's facts, each with a hover explanation derived from the config. */
export function headerFacts(c: GptOssConfig, b: ParamBreakdown): HeaderFact[] {
  const L = c.num_hidden_layers
  const E = c.num_local_experts
  const k = c.num_experts_per_tok
  const sliding = c.layer_types.filter((t) => t === 'sliding_attention').length
  const kda = c.layer_types.filter((t) => t === 'linear_attention').length
  const row = (id: ParamRow['id']) => b.rows.find((r) => r.id === id)!
  const terms = (rows: ParamRow[], f: (r: ParamRow) => number) => rows.map((r) => `${SHORT[r.id]} ${formatCount(f(r))}`).join(' + ')
  const embed = row('embed')
  const experts = row('experts')
  const used = b.rows.filter((r) => r.active > 0)
  const rs = c.rope_scaling
  return [
    {
      label: 'Params',
      value: formatCount(b.total),
      title: `Every weight in the checkpoint:\n${terms(b.rows, (r) => r.total)}\n= ${formatCount(b.total)}`,
    },
    {
      label: 'Active',
      value: formatCount(b.activeExclEmbed),
      title:
        `Weights one token is multiplied by:\n${terms(used, (r) => r.active)}\n= ${formatCount(b.activeExclEmbed)}\n` +
        `Experts count ${L} layers × ${k} of ${E}. The embedding (${formatCount(embed.total)}) is a table lookup, not a multiply, so it is left out.`,
    },
    {
      label: 'Layers',
      value: String(L),
      title: kda > 0
        ? `Decoder blocks (num_hidden_layers): ${kda} with KDA linear attention (3:1 ratio) and ${L - kda} with Gated Multi-Latent Attention (MLA).`
        : `Decoder blocks (num_hidden_layers): ${sliding} with sliding attention, seeing the last ${c.sliding_window} tokens, alternating with ${L - sliding} with full attention, seeing all tokens.`,
    },
    {
      label: 'Experts',
      value: `${E} · top-${k}`,
      title: c.num_shared_experts
        ? `Each MoE block has ${E} routed expert MLPs; its router picks ${k} for each token, so ${((k / E) * 100).toFixed(1)}% of the routed expert weights run per token (+ ${c.num_shared_experts} shared experts).`
        : `Each block has ${E} expert MLPs (num_local_experts); its router picks ${k} for each token (num_experts_per_tok), so ${((k / E) * 100).toFixed(1)}% of the expert weights run per token.`,
    },
    {
      label: 'Context',
      value: c.max_position_embeddings.toLocaleString('en-US'),
      title: rs.rope_type === 'none'
        ? `Longest input in tokens (max_position_embeddings): ${c.max_position_embeddings.toLocaleString('en-US')} tokens.`
        : `Longest input in tokens (max_position_embeddings): trained at ${rs.original_max_position_embeddings.toLocaleString('en-US')} and stretched ×${rs.factor} with ${rs.rope_type.toUpperCase()} RoPE scaling.`,
    },
    {
      label: 'Size',
      value: formatBytes(b.bytes),
      title:
        `Checkpoint on disk:\nexpert weights in MXFP4, 4.25 bits each (4-bit values + one 8-bit scale per 32; biases bf16): ${formatBytes(experts.bytes)}\n` +
        `+ everything else in bf16, 2 bytes each: ${formatBytes(b.bytes - experts.bytes)}\n= ${formatBytes(b.bytes)}\n` +
        `All in bf16 it would be ${formatBytes(b.total * 2)}.`,
    },
  ]
}
