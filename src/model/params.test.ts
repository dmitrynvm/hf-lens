import { describe, expect, it } from 'vitest'
import { GPT_OSS_120B, GPT_OSS_20B, KIMI_K3 } from './config'
import { attentionParams, expertParams, layerMemory, paramBreakdown, routerParams } from './params'
import { simulateRouting } from './routing'

describe('paramBreakdown', () => {
  it('matches published totals for gpt-oss-120b', () => {
    const b = paramBreakdown(GPT_OSS_120B)
    expect(b.total).toBe(116_829_156_672)
    expect(b.activeExclEmbed).toBe(5_132_849_472)
    expect(b.activeInclEmbed).toBe(5_711_982_912)
    expect(b.bytes).toBe(65_248_815_744)
  })

  it('matches published totals for gpt-oss-20b (20.91B total, 3.61B active)', () => {
    const b = paramBreakdown(GPT_OSS_20B)
    expect(b.total).toBe(20_914_757_184)
    expect(b.activeExclEmbed).toBe(3_608_307_264)
    expect(b.bytes).toBe(13_761_264_768)
  })

  it('computes Kimi-K3 hybrid total and active parameters', () => {
    const b = paramBreakdown(KIMI_K3)
    expect(b.total).toBeGreaterThan(2.7e12)
    expect(b.total).toBeLessThan(2.8e12)
    expect(b.activeExclEmbed).toBeGreaterThan(100e9)
    expect(b.activeExclEmbed).toBeLessThan(110e9)
    expect(layerMemory(KIMI_K3, 0).rows.find((r) => r.id === 'router')!.params).toBe(0)
    expect(layerMemory(KIMI_K3, 1).rows.find((r) => r.id === 'router')!.params).toBe(7168 * 896 + 896)
  })

  it('computes per-module counts', () => {
    expect(attentionParams(GPT_OSS_120B)).toBe(26_550_144)
    expect(routerParams(GPT_OSS_120B)).toBe(368_768)
    expect(expertParams(GPT_OSS_120B)).toBe(24_891_840)
  })
})

describe('layerMemory', () => {
  it('reads attention, router, norms and only the top-k experts per token', () => {
    const m = layerMemory(GPT_OSS_120B)
    expect(m.bytes).toBe(1_748_118_784)
    expect(m.activeBytes).toBe(106_795_264)
    expect(m.active).toBe(126_492_032)
  })
})

describe('simulateRouting', () => {
  it('yields k distinct experts per layer with normalized weights, deterministically', () => {
    const routes = simulateRouting(GPT_OSS_120B, 42)
    expect(routes).toHaveLength(36)
    for (const r of routes) {
      expect(new Set(r.experts).size).toBe(4)
      for (const e of r.experts) expect(e >= 0 && e < 128).toBe(true)
      expect(Math.abs(r.weights.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-9)
      for (let i = 1; i < r.weights.length; i++) expect(r.weights[i]).toBeLessThanOrEqual(r.weights[i - 1])
    }
    const meanTop = routes.reduce((a, r) => a + r.weights[0], 0) / routes.length
    expect(meanTop).toBeGreaterThan(0.4)
    expect(simulateRouting(GPT_OSS_120B, 42)).toEqual(routes)
    expect(simulateRouting(GPT_OSS_120B, 43)).not.toEqual(routes)
  })

  it('stays within the 20b expert range', () => {
    const routes = simulateRouting(GPT_OSS_20B, 42)
    expect(routes).toHaveLength(24)
    for (const r of routes) {
      expect(new Set(r.experts).size).toBe(4)
      for (const e of r.experts) expect(e >= 0 && e < 32).toBe(true)
    }
  })
})
