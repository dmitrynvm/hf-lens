import { describe, expect, it } from 'vitest'
import { GPT_OSS_120B, GPT_OSS_20B } from './config'
import { headerFacts } from './facts'
import { formatBytes, formatCount, paramBreakdown } from './params'

describe('headerFacts', () => {
  it('labels and values match the top bar', () => {
    const c = GPT_OSS_120B
    expect(headerFacts(c, paramBreakdown(c)).map((f) => [f.label, f.value])).toEqual([
      ['Params', '116.83B'],
      ['Active', '5.13B'],
      ['Layers', '36'],
      ['Experts', '128 · top-4'],
      ['Context', '131,072'],
      ['Size', '60.77 GiB'],
    ])
  })

  it('each sum names its parts and its total', () => {
    for (const c of [GPT_OSS_120B, GPT_OSS_20B]) {
      const b = paramBreakdown(c)
      const facts = headerFacts(c, b)
      const t = (label: string) => facts.find((f) => f.label === label)!.title
      for (const r of b.rows) expect(t('Params')).toContain(formatCount(r.total))
      expect(t('Params').endsWith(`= ${formatCount(b.total)}`)).toBe(true)
      expect(t('Active')).toContain(`= ${formatCount(b.activeExclEmbed)}`)
      expect(t('Active')).not.toContain(`embedding ${formatCount(b.rows[0].total)} +`)
      expect(t('Size')).toContain(`= ${formatBytes(b.bytes)}`)
      expect(t('Size')).toContain(formatBytes(b.total * 2))
    }
  })

  it('context explains the RoPE stretch', () => {
    const c = GPT_OSS_120B
    const context = headerFacts(c, paramBreakdown(c)).find((f) => f.label === 'Context')!.title
    expect(context).toContain('trained at 4,096 and stretched ×32')
  })
})
