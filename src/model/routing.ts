import type { GptOssConfig } from './config'

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export interface LayerRoute {
  /** Distinct expert ids, sorted by weight descending. */
  experts: number[]
  /** Softmax over the selected logits; sums to 1. */
  weights: number[]
}

/** Seeded stand-in for the router: exponential logits (so top-k weights are peaked like a real router) → top-k → softmax over the k selected. */
export function simulateRouting(c: GptOssConfig, seed: number): LayerRoute[] {
  const rng = mulberry32(seed)
  const E = c.num_local_experts
  const k = c.num_experts_per_tok
  return Array.from({ length: c.num_hidden_layers }, () => {
    const logits = Array.from({ length: E }, () => -Math.log(1 - rng()))
    const top = logits
      .map((logit, id) => ({ logit, id }))
      .sort((a, b) => b.logit - a.logit)
      .slice(0, k)
    const max = top[0].logit
    const exps = top.map((e) => Math.exp(e.logit - max))
    const z = exps.reduce((a, b) => a + b, 0)
    return { experts: top.map((e) => e.id), weights: exps.map((x) => x / z) }
  })
}
