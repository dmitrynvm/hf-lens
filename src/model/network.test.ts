import { describe, expect, it } from 'vitest'
import { GPT_OSS_120B, GPT_OSS_20B } from './config'
import { CY, groupForStep, JUNCTION_R, networkLayout, slabHeight, slabThickness, textWidth, type Part } from './network'
import { layerMemory } from './params'

describe('networkLayout', () => {
  it('draws every block between the embedding and the head, each in its own group', () => {
    for (const c of [GPT_OSS_120B, GPT_OSS_20B]) {
      const L = c.num_hidden_layers
      const { groups } = networkLayout(c)
      expect(groups.map((g) => g.label)).toEqual(['embedding', ...Array.from({ length: L }, (_, i) => `L${i}`), 'lm_head'])
      expect(groups.flatMap((g) => g.layers)).toEqual(Array.from({ length: L }, (_, i) => i))
    }
  })

  it('gives every block two branches, each rejoining the stream at a junction past its slabs', () => {
    for (const g of networkLayout(GPT_OSS_120B).groups.filter((g) => g.kind === 'block')) {
      expect(g.branches).toHaveLength(2)
      for (const b of g.branches) {
        expect(b.tapX).toBeLessThan(b.parts[0].x0)
        expect(b.joinX).toBeGreaterThan(b.parts[b.parts.length - 1].x1)
      }
    }
  })

  it('stacks parallel tensors in depth and lays sequential ones along x', () => {
    const [attn, moe] = networkLayout(GPT_OSS_120B).groups.find((g) => g.id === 'L0')!.branches
    const qkv = attn.parts[1]
    expect(qkv.parallel).toBe(true)
    expect(qkv.slabs.map((s) => s.x - qkv.slabs[0].x)).toEqual([0, 5, 10])
    expect(qkv.slabs.map((s) => qkv.slabs[0].y - s.y)).toEqual([0, 5, 10])
    const experts = moe.parts[2]
    expect(experts.experts).toBe(true)
    expect(experts.slabs).toHaveLength(GPT_OSS_120B.num_experts_per_tok)
    attn.parts.slice(1).forEach((p, i) => expect(p.x0).toBeGreaterThan(attn.parts[i].x1))
  })

  it('lays groups out left to right without overlap', () => {
    const { groups, width } = networkLayout(GPT_OSS_120B)
    groups.slice(1).forEach((g, i) => expect(g.x0).toBeGreaterThan(groups[i].x1))
    expect(width).toBeGreaterThan(groups[groups.length - 1].x1)
  })

  it('keeps op labels and captions apart on each level', () => {
    for (const c of [GPT_OSS_120B, GPT_OSS_20B]) {
      const parts = networkLayout(c).groups.flatMap((g) => [...g.parts, ...g.branches.flatMap((b) => b.parts)])
      for (const level of [true, false]) {
        const onLevel = parts.filter((p) => (p.slabs[0].y === CY) === level)
        const apart = (spans: [number, number][]) => {
          const sorted = spans.filter((s) => s[1] > s[0]).sort((a, b) => a[0] - b[0])
          sorted.slice(1).forEach((s, i) => expect(s[0]).toBeGreaterThanOrEqual(sorted[i][1]))
        }
        apart(onLevel.map((p: Part) => [p.opX - p.opW / 2, p.opX + p.opW / 2]))
        apart(onLevel.map((p: Part) => [(p.x0 + p.x1) / 2 - p.capW / 2, (p.x0 + p.x1) / 2 + p.capW / 2]))
      }
    }
  })

  it('keeps branch captions clear of the tap and join wires', () => {
    for (const c of [GPT_OSS_120B, GPT_OSS_20B]) {
      for (const b of networkLayout(c).groups.flatMap((g) => g.branches)) {
        for (const p of b.parts.filter((p) => p.capW > 0)) {
          const capX = (p.x0 + p.x1) / 2
          expect(capX - p.capW / 2).toBeGreaterThan(b.tapX)
          expect(capX + p.capW / 2).toBeLessThan(b.joinX - JUNCTION_R)
        }
        for (const p of b.parts.filter((p) => p.opW > 0)) {
          expect(p.opX - p.opW / 2).toBeGreaterThan(b.tapX)
          expect(p.opX + p.opW / 2).toBeLessThan(b.joinX)
        }
      }
    }
  })

  it("fits each group's label inside its bracket", () => {
    for (const c of [GPT_OSS_120B, GPT_OSS_20B]) {
      for (const g of networkLayout(c).groups) expect(g.x1 - g.x0).toBeGreaterThanOrEqual(textWidth(g.label, 22, 0.55))
    }
  })

  it('maps flow steps to the embedding, each block or the head', () => {
    const { groups } = networkLayout(GPT_OSS_120B)
    expect(groupForStep(groups, -1, 36)?.kind).toBe('embed')
    expect(groupForStep(groups, 7, 36)?.label).toBe('L7')
    expect(groupForStep(groups, 36, 36)?.kind).toBe('head')
  })

  it('names each sublayer with its type and keeps the tag inside its branch', () => {
    for (const c of [GPT_OSS_120B, GPT_OSS_20B]) {
      const blocks = networkLayout(c).groups.filter((g) => g.kind === 'block')
      blocks.forEach((g, i) => {
        const kind = c.layer_types[i] === 'sliding_attention' ? 'sliding' : 'full'
        expect(g.branches.map((b) => [b.label, b.attention])).toEqual([
          [`${kind} attention`, kind],
          [`MoE · top-${c.num_experts_per_tok} of ${c.num_local_experts} experts`, null],
        ])
        for (const b of g.branches) expect(b.joinX - b.tapX).toBeGreaterThanOrEqual(textWidth(b.label, 12, 0.55) - 1e-9)
      })
    }
    const { groups } = networkLayout(GPT_OSS_120B)
    expect(groups.find((g) => g.id === 'L0')?.sub).toBe('last 128 tokens')
    expect(groups.find((g) => g.id === 'L1')?.sub).toBe('all tokens')
  })

  it('captions every activation with shape, dtype and size per token', () => {
    const { groups } = networkLayout(GPT_OSS_120B)
    const l0 = groups.find((g) => g.id === 'L0')!
    expect(l0.branches[0].parts[1].captions).toEqual(['q T×4096 bf16 · 8.00 KiB/tok', 'k T×512 bf16 · 1.00 KiB/tok', 'v T×512 bf16 · 1.00 KiB/tok'])
    expect(groups[0].parts[0].captions).toEqual(['T int64 · 8 B/tok'])
    for (const p of groups.flatMap((g) => [...g.parts, ...g.branches.flatMap((b) => b.parts)])) expect(p.captions.length).toBeGreaterThanOrEqual(1)
  })
})

describe('slabThickness', () => {
  it('grows with parameter count and is paper thin for parameter-free ops', () => {
    expect(slabThickness(0)).toBe(3)
    expect(slabThickness(2880)).toBe(3)
    const ps = [3.7e5, 1.5e6, 1.2e7, 2.5e7, 5.8e8]
    const ts = ps.map(slabThickness)
    ts.slice(1).forEach((t, i) => expect(t).toBeGreaterThan(ts[i]))
  })
})

describe('slabHeight', () => {
  it('grows with the feature width', () => {
    const dims = [128, 512, 2880, 4096, 5760, 201088]
    const hs = dims.map(slabHeight)
    hs.slice(1).forEach((h, i) => expect(h).toBeGreaterThan(hs[i]))
  })
})

describe('weight tape', () => {
  const configs = [GPT_OSS_120B, GPT_OSS_20B]

  it('colours each segment like its sidebar tile', () => {
    for (const c of configs) {
      expect(networkLayout(c).tape.map((b) => b.kind)).toEqual([
        'end',
        ...c.layer_types.map((t) => (t === 'sliding_attention' ? 'sliding' : 'full')),
        'end',
      ])
    }
  })

  it('tiles one gap-free tape from the embedding to lm_head', () => {
    for (const c of configs) {
      const { groups, tape } = networkLayout(c)
      expect(tape.map((b) => b.groupId)).toEqual(groups.map((g) => g.id))
      expect(tape[0].bf16[0]).toBe(groups[0].x0)
      expect(tape.at(-1)!.mxfp4[1]).toBe(groups.at(-1)!.x1)
      tape.forEach((b, i) => {
        if (i > 0) expect(b.bf16[0]).toBeCloseTo(tape[i - 1].mxfp4[1], 9)
        expect(b.bf16[1]).toBe(b.mxfp4[0])
        expect(b.labelX).toBeGreaterThanOrEqual(b.bf16[0])
        expect(b.labelX).toBeLessThanOrEqual(b.mxfp4[1])
      })
    }
  })

  it('splits each segment by its bytes', () => {
    for (const c of configs) {
      const { tape } = networkLayout(c)
      const mem = layerMemory(c)
      const ex = mem.rows.find((r) => r.id === 'experts')!
      const seg = (id: string) => tape.find((b) => b.groupId === id)!
      const share = (b: (typeof tape)[number]) => (b.bf16[1] - b.bf16[0]) / (b.mxfp4[1] - b.bf16[0])
      expect(share(seg('L0'))).toBeCloseTo((mem.bytes - ex.bytes) / mem.bytes, 6)
      for (let i = 1; i < c.num_hidden_layers; i++) expect(share(seg(`L${i}`))).toBeCloseTo(share(seg('L0')), 6)
      for (const id of ['embed', 'head']) expect(seg(id).mxfp4[1] - seg(id).mxfp4[0]).toBe(0)
    }
  })

  it('labels every block with its size, inside its group', () => {
    expect(networkLayout(GPT_OSS_120B).tape.find((b) => b.groupId === 'L7')!.label).toBe('1.63 GiB')
    for (const c of configs) {
      const { groups, tape } = networkLayout(c)
      tape.forEach((b, i) => {
        const g = groups[i]
        expect(textWidth(b.label, 13, 0.62)).toBeLessThanOrEqual(g.x1 - g.x0)
      })
    }
  })

  it('explains how each segment size is computed', () => {
    for (const c of configs) {
      for (const b of networkLayout(c).tape) expect(b.title.endsWith(`= ${b.label}`)).toBe(true)
    }
    const { tape } = networkLayout(GPT_OSS_120B)
    expect(tape.find((b) => b.groupId === 'L3')!.title).toBe([
      'L3 weights: bf16 at 2 bytes per param, MXFP4 at 4.25 bits',
      'attention 26.55M × 2 B = 50.64 MiB',
      '+ router 368.77K × 2 B = 720.25 KiB',
      '+ experts 128 × 24.89M: weights 3.19B × 4.25 bits = 1.58 GiB, biases 1.11M × 2 B = 2.11 MiB',
      '+ norms 5.76K × 2 B = 11.25 KiB',
      '= 1.63 GiB',
    ].join('\n'))
    expect(tape.find((b) => b.groupId === 'embed')!.title).toBe(
      'embedding weights: 201,088 tokens × 2,880 = 579.13M params × 2 B (bf16) = 1.08 GiB',
    )
  })
})
