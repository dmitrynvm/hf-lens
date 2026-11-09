import { describe, expect, it } from 'vitest'
import { GPT_OSS_120B, GPT_OSS_20B, KIMI_K3 } from './config'
import { CAP_PITCH, CY, DY, GROUP_BG_BOTTOM, groupForStep, JUNCTION_R, networkLayout, slabHeight, slabThickness, TAPE_TOP, textWidth, type Part } from './network'
import { layerMemory } from './params'

describe('networkLayout', () => {
  it('draws every block between the embedding and the head, each in its own group', () => {
    for (const c of [GPT_OSS_120B, GPT_OSS_20B]) {
      const L = c.num_hidden_layers
      const { groups } = networkLayout(c)
      expect(groups.map((g) => g.label)).toEqual(['Input', ...Array.from({ length: L }, (_, i) => `L${i}`), 'Output'])
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

  it('keeps every slab vertically clear of its op label above and caption below', () => {
    for (const c of [GPT_OSS_120B, GPT_OSS_20B, KIMI_K3]) {
      const parts = networkLayout(c).groups.flatMap((g) => [...g.parts, ...g.branches.flatMap((b) => b.parts)])
      for (const p of parts) {
        const topY = Math.min(...p.slabs.map((s) => s.y - s.h / 2 - DY / 2))
        const botY = Math.max(...p.slabs.map((s) => s.y + s.h / 2 + DY / 2))
        if (p.op) expect(topY).toBeGreaterThanOrEqual(p.opY + 6)
        if (p.captions.length) expect(botY).toBeLessThanOrEqual(p.captionY - 7)
      }
    }
  })

  it("fits each group's label inside its bracket", () => {
    for (const c of [GPT_OSS_120B, GPT_OSS_20B]) {
      for (const g of networkLayout(c).groups) expect(g.x1 - g.x0).toBeGreaterThanOrEqual(textWidth(g.label, 18, 0.55))
    }
  })

  it('maps flow steps to the embedding, each block or the head', () => {
    const { groups } = networkLayout(GPT_OSS_120B)
    expect(groupForStep(groups, -1, 36)?.kind).toBe('embed')
    expect(groupForStep(groups, 7, 36)?.label).toBe('L7')
    expect(groupForStep(groups, 36, 36)?.kind).toBe('head')
  })

  it('unpacks L0 in full and marks RMSNorm, SWA/FA and MoE on every later block', () => {
    for (const c of [GPT_OSS_120B, GPT_OSS_20B]) {
      const { groups } = networkLayout(c)
      const blocks = groups.filter((g) => g.kind === 'block')
      expect(blocks[0].branches.map((b) => b.parts.length)).toEqual([4, 6])
      for (const g of blocks.slice(1)) {
        expect(g.branches.map((b) => b.parts.length)).toEqual([2, 2])
        const attnOp = g.attention === 'sliding' ? 'SWA' : 'FA'
        expect(g.branches[0].parts.map((p) => p.op)).toEqual(['RMSNorm', attnOp])
        expect(g.branches[1].parts.map((p) => p.op)).toEqual(['RMSNorm', 'MoE'])
        expect(g.branches[1].parts[1].experts).toBe(true)
        expect(g.x1 - g.x0).toBeLessThan(blocks[0].x1 - blocks[0].x0)
      }
      expect(groups[0].parts.map((p) => p.op)).toEqual(['tokens', 'Embed'])
      expect(groups.at(-1)!.parts.map((p) => p.op)).toEqual(['RMSNorm', 'Unembed', 'Softmax'])
    }
  })

  it('captions every activation with compact dimensions and dtype, keeping full details in hover titles', () => {
    const layout = networkLayout(GPT_OSS_120B)
    const { groups } = layout
    const l0 = groups.find((g) => g.id === 'L0')!
    expect(l0.branches[0].parts[1].captions).toEqual(['64q·8kv×64', 'bf16'])
    expect(l0.branches[0].parts[2].captions).toEqual(['64h×64', 'bf16'])
    expect(l0.branches[0].parts[1].slabs[0].title).toContain('T×4096 bf16 · 8.00 KiB/tok')
    expect(l0.branches[1].parts[1].captions).toEqual(['4/128', 'bf16'])
    expect(l0.branches[1].parts[2].captions).toEqual(['5760', 'mxfp4'])
    expect(groups[0].parts[0].captions).toEqual(['1', 'int64'])
    expect(groups[0].parts[1].captions).toEqual(['2880', 'bf16'])
    expect(groups.at(-1)!.parts[1].captions).toEqual(['201088', 'bf16'])
    expect(groups.at(-1)!.parts[2].captions).toEqual(['201088', 'bf16'])
    expect(layout.streamLabel).toBe('residual stream (2880)')
    for (const p of groups.flatMap((g) => [...g.parts, ...g.branches.flatMap((b) => b.parts)])) {
      expect(p.captions.at(-1)).toMatch(/^(int64|bf16|mxfp4)$/)
    }
  })

  it('keeps branch captions between the tallest slab and the residual stream', () => {
    for (const c of [GPT_OSS_120B, GPT_OSS_20B, KIMI_K3]) {
      for (const p of networkLayout(c).groups.flatMap((g) => g.branches.flatMap((b) => b.parts))) {
        const slabBottom = Math.max(...p.slabs.map((s) => s.y + s.h / 2 + DY / 2))
        expect(p.captionY - 9).toBeGreaterThan(slabBottom) // ~9 units of glyph above the baseline
        expect(p.captionY + (p.captions.length - 1) * CAP_PITCH).toBeLessThan(CY - 9)
      }
    }
  })

  it('lays out Kimi-K3 with dense L0, KDA/MLA blocks, and gap-free weight tape', () => {
    const { groups, tape } = networkLayout(KIMI_K3)
    expect(groups).toHaveLength(95)
    const l0 = groups.find((g) => g.id === 'L0')!
    expect(l0.branches[0].parts.map((p) => p.op)).toEqual(['RMSNorm', 'q·k·v', 'KDA', 'o_proj'])
    expect(l0.branches[1].parts.map((p) => p.op)).toEqual(['RMSNorm', 'gate_up', 'SiTU', 'down'])
    const l1 = groups.find((g) => g.id === 'L1')!
    expect(l1.branches[0].parts.map((p) => p.op)).toEqual(['RMSNorm', 'KDA'])
    expect(l1.branches[1].parts.map((p) => p.op)).toEqual(['RMSNorm', 'LatentMoE'])
    const l3 = groups.find((g) => g.id === 'L3')!
    expect(l3.branches[0].parts.map((p) => p.op)).toEqual(['RMSNorm', 'MLA'])
    expect(tape[0].bf16[0]).toBe(groups[0].x0)
    expect(tape.at(-1)!.mxfp4[1]).toBe(groups.at(-1)!.x1)
    for (const g of groups.filter((g) => g.kind === 'block')) {
      const parts = g.branches.flatMap((b) => b.parts).filter((p) => p.weights)
      const sum = parts.reduce((acc, p) => acc + p.weights!.bf16 + p.weights!.mxfp4, 0)
      expect(sum).toBe(layerMemory(KIMI_K3, g.layers[0]).bytes)
    }
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

  it('creates one segment per weighted tensor step across all groups', () => {
    for (const c of configs) {
      const { groups, tape } = networkLayout(c)
      const weightedCount = groups
        .flatMap((g) => [...g.parts, ...g.branches.flatMap((b) => b.parts)])
        .filter((p) => p.weights).length
      expect(tape).toHaveLength(weightedCount)
      expect(tape.filter((b) => b.groupId === 'L0')).toHaveLength(8)
      expect(tape.filter((b) => b.groupId === 'L3')).toHaveLength(4)
    }
  })

  it('tiles one gap-free tape from the embedding to lm_head', () => {
    for (const c of configs) {
      const { groups, tape } = networkLayout(c)
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

  it('splits each block into its tensor weight sizes summing to the block memory', () => {
    for (const c of configs) {
      const { groups } = networkLayout(c)
      const mem = layerMemory(c)
      for (const g of groups.filter((g) => g.kind === 'block')) {
        const parts = g.branches.flatMap((b) => b.parts).filter((p) => p.weights)
        const sum = parts.reduce((acc, p) => acc + p.weights!.bf16 + p.weights!.mxfp4, 0)
        expect(sum).toBe(mem.bytes)
      }
    }
    const l3 = networkLayout(GPT_OSS_120B).tape.filter((b) => b.groupId === 'L3')
    expect(l3.map((b) => b.label)).toEqual(['5.63 KiB', '50.64 MiB', '5.63 KiB', '1.58 GiB'])
  })

  it('fits every tensor size label inside its tape segment', () => {
    for (const c of configs) {
      for (const b of networkLayout(c).tape) {
        expect(textWidth(b.label, 10.5, 0.62)).toBeLessThanOrEqual(b.mxfp4[1] - b.bf16[0] + 1e-9)
      }
    }
  })

  it('explains how each segment size is computed', () => {
    for (const c of configs) {
      for (const b of networkLayout(c).tape) expect(b.title.endsWith(`= ${b.label}`)).toBe(true)
    }
  })

  it('reads, per token, exactly the block bytes the sidebar reports as active', () => {
    for (const c of [GPT_OSS_120B, GPT_OSS_20B, KIMI_K3]) {
      const { groups, tape } = networkLayout(c)
      for (const g of groups.filter((g) => g.kind === 'block')) {
        const sum = tape.filter((b) => b.groupId === g.id).reduce((acc, b) => acc + b.active, 0)
        expect(sum).toBeCloseTo(layerMemory(c, g.layers[0]).activeBytes, 0)
      }
    }
    expect(networkLayout(GPT_OSS_120B).tape.find((b) => b.groupId === 'embed')!.active).toBe(2880 * 2)
  })

  it('draws the block tape tiling every group with no gap', () => {
    for (const c of configs) {
      const { groups, blockTape } = networkLayout(c)
      const segs = blockTape.segments
      expect(segs).toHaveLength(groups.length)
      expect(segs[0].x0).toBe(groups[0].x0)
      expect(segs.at(-1)!.x1).toBe(groups.at(-1)!.x1)
      segs.forEach((s, i) => i > 0 && expect(s.x0).toBe(segs[i - 1].x1))
    }
  })

  it('labels a block with its total and per-token bytes', () => {
    const l3 = networkLayout(GPT_OSS_120B).blockTape.segments.find((s) => s.groupId === 'L3')!
    expect(['1.63 GiB · 101.85 MiB/tok', '1.63 GiB']).toContain(l3.label)
    expect(l3.title).toContain('101.85 MiB read per token')
  })

  it('colours each block segment by its attention type, like the minimap', () => {
    const segs = networkLayout(KIMI_K3).blockTape.segments
    const kind = (id: string) => segs.find((s) => s.groupId === id)!.kind
    expect(kind('embed')).toBe('end')
    expect(kind('L1')).toBe('sliding') // KDA linear attention
    expect(kind('L3')).toBe('full') // MLA
    expect(kind('head')).toBe('end')
    const oss = networkLayout(GPT_OSS_120B).blockTape.segments
    expect(oss.find((s) => s.groupId === 'L0')!.kind).toBe('sliding')
    expect(oss.find((s) => s.groupId === 'L1')!.kind).toBe('full')
  })
})

describe('block squares', () => {
  const inside = (a: number, b: number, lo: number, hi: number) => {
    expect(a).toBeGreaterThanOrEqual(lo)
    expect(b).toBeLessThanOrEqual(hi)
  }

  it('puts a mask under attention and an expert grid under MoE, inside each block and above the tapes', () => {
    for (const c of [GPT_OSS_120B, GPT_OSS_20B, KIMI_K3]) {
      for (const g of networkLayout(c).groups) {
        if (g.kind !== 'block') {
          expect(g.squares).toBeUndefined()
          continue
        }
        const s = g.squares!
        const [attn, moe] = g.branches
        expect(s.mask.kind).toBe(c.layer_types[g.layers[0]] === 'linear_attention' ? 'linear' : g.attention)
        inside(s.mask.x - 6, s.mask.x + s.mask.side, attn.tapX, attn.joinX)
        if (s.moe.grid) inside(s.moe.grid.x, s.moe.grid.x + s.moe.grid.width, moe.tapX, moe.joinX)
        expect(s.captionY + 4).toBeLessThanOrEqual(GROUP_BG_BOTTOM)
      }
    }
    expect(GROUP_BG_BOTTOM).toBeLessThan(TAPE_TOP)
  })

  it('draws the sliding band, the causal triangle and the routed share', () => {
    const { groups } = networkLayout(GPT_OSS_120B)
    const l2 = groups.find((g) => g.id === 'L2')!.squares!
    const l3 = groups.find((g) => g.id === 'L3')!.squares!
    expect(l2.mask.points).toBe('0,0 120,120 90,120 0,30')
    expect(l2.mask.keys).toBe(128)
    expect(l2.mask.shown).toBe(512)
    expect(l3.mask.keys).toBe(512)
    expect(l3.mask.shown).toBe(512)
    expect(l3.moe.caption).toBe('4/128')
  })

  it('draws KDA as a fading causal triangle, not a window', () => {
    const groups = networkLayout(KIMI_K3).groups
    const kda = groups.find((g) => g.id === 'L1')!.squares!.mask
    expect(kda.kind).toBe('linear')
    expect(kda.points).toBe('0,0 120,120 0,120')
    expect(kda.keys).toBe(512)
    expect(kda.shown).toBe(512)
    expect(kda.title).toContain('∏α')
    expect(groups.find((g) => g.id === 'L3')!.squares!.mask.kind).toBe('full')
  })

  it('fits the expert grid in a square the size of the mask and skips it for dense blocks', () => {
    const shape = (c: typeof GPT_OSS_120B, id: string) => {
      const g = networkLayout(c).groups.find((x) => x.id === id)!.squares!
      const gr = g.moe.grid!
      expect(gr.width).toBe(g.mask.side)
      expect(gr.cols * gr.pitch).toBeCloseTo(gr.width)
      expect(gr.rows * gr.pitch).toBeCloseTo(gr.height)
      expect(gr.height).toBeLessThanOrEqual(gr.width)
      expect(gr.cols * gr.rows).toBeGreaterThanOrEqual(gr.experts)
      return [gr.cols, gr.rows]
    }
    expect(shape(GPT_OSS_120B, 'L3')).toEqual([12, 11])
    expect(shape(GPT_OSS_20B, 'L3')).toEqual([6, 6])
    expect(shape(KIMI_K3, 'L1')).toEqual([30, 30])
    const l0 = networkLayout(KIMI_K3).groups.find((g) => g.id === 'L0')!.squares!.moe
    expect(l0.grid).toBeNull()
    expect(l0.caption).toBe('1/1')
  })
})
