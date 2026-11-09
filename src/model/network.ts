import type { GptOssConfig } from './config'
import { expertMxfp4Params, expertParams, formatBytes, formatCount, layerMemory } from './params'

/**
 * Whole-network diagram geometry. Visual grammar:
 * - The residual stream is a thick horizontal line; every sublayer is a branch that leaves it, runs above it and
 *   rejoins at ⊕. That is the transformer's signature and matches the Block tab's vertical residual.
 * - Every slab is an activation [T, d]. Height = d (log scale), thickness = parameters of the op that produced it
 *   (log scale), depth = T (constant). Operations are labels above the gaps; dimensions are captions below.
 * - Tensors computed in parallel (q/k/v, the top-k experts) stack in depth; sequential ones follow along x.
 * Pure data so it can be tested without a DOM; `NetworkView` only draws it.
 */

export type SlabKind = 'bf16' | 'mxfp4' | 'sliding' | 'full' | 'output' | 'input'

export interface Slab {
  x: number
  /** Centre of the front face. */
  y: number
  /** Thickness along the flow axis. */
  t: number
  h: number
  kind: SlabKind
  title: string
}

export interface Part {
  /** Front-most first; parallel parts are drawn back to front. */
  slabs: Slab[]
  parallel: boolean
  x0: number
  /** Right edge of the front slab's face, where the next connector starts. */
  frontX1: number
  /** Right edge including the last side face. */
  x1: number
  op: string
  opX: number
  opY: number
  /** Shape, dtype and size of the activations this op produces, one line each. */
  captions: string[]
  captionY: number
  /** Expert slabs: the view appends the routed expert ids under the captions once the token has passed. */
  experts?: boolean
  /** Reserved label widths (px) used for spacing: op slot (label or weight column) and captions (0 when there is none). */
  opW: number
  capW: number
}
export interface Branch {
  /** Sublayer name drawn above the branch, with its type: `sliding attention`, `MoE · top-4 of 128 experts`. */
  label: string
  /** Attention type of an attention branch, which colours its tag like the block subtitle; null for the MoE branch. */
  attention: 'sliding' | 'full' | null
  tapX: number
  joinX: number
  parts: Part[]
}

export type GroupKind = 'embed' | 'block' | 'head'

export interface Group {
  id: string
  kind: GroupKind
  label: string
  sub: string
  /** Attention type of a drawn block; null for other groups. */
  attention: 'sliding' | 'full' | null
  /** The decoder layer a block draws; empty for the embedding and head. */
  layers: number[]
  x0: number
  x1: number
  /** Parts sitting on the stream itself (embedding, head). */
  parts: Part[]
  branches: Branch[]
}

export interface NetworkLayout {
  groups: Group[]
  /** The residual stream's x extent. */
  stream: [number, number]
  streamLabel: string
  width: number
  height: number
  /** One weight-tape segment per group, left to right, touching with no gap: together they span the embedding's x0 to lm_head's x1. */
  tape: WeightBar[]
}

/** One group's stretch of the weight tape: from halfway to the previous group to halfway to the next (the outer ends stop at the group edge), split by the group's bf16 / mxfp4 bytes and coloured like its sidebar tile. */
export interface WeightBar {
  groupId: string
  /** Colour family, matching the sidebar tile underline: the block's attention type, or 'end' for the embedding and lm_head. */
  kind: 'sliding' | 'full' | 'end'
  /** x-extents of the bf16 part, then the mxfp4 part (empty for the embedding and lm_head). */
  bf16: [number, number]
  mxfp4: [number, number]
  /** Group centre, where the size label sits (not the segment centre). */
  labelX: number
  label: string
  /** Hover text: how the size label is computed, term by term, ending "= <label>". */
  title: string
}

export const LABEL_Y = 26 // group label (22px), top of the diagram
export const SUB_Y = 46 // group subtitle (16px)
export const BRACKET_Y = 58 // bracket under the header, ticks pointing down into the group
export const SUBLAYER_Y = 82 // sublayer tag row
export const OP_Y_BRANCH = 104
export const BY = 170 // branch centre line
export const CAP_Y_BRANCH = 230 // first caption line under branch slabs
export const CY = 300 // residual stream
export const OP_Y_STREAM = 210
export const CAP_Y_STREAM = 380
export const CAP_PITCH = 14 // caption line spacing
export const GROUP_BG_BOTTOM = 406 // selection background ends here
export const BAR_Y = 422
export const BAR_H = 16
export const SIZE_LABEL_Y = 456
export const TAPE_KEY_Y = 486
export const HEIGHT = 502
export const DEPTH = 28
export const DX = DEPTH * 0.5
export const DY = DEPTH * 0.5
export const JUNCTION_R = 11
export const STEP = 5 // depth offset between parallel slabs
const GAP = 16
const GROUP_GAP = 20
const TAP = 6
const PAD = 16
const OP_FONT = [14, 0.55] as const // .net-op: Barlow Semi Condensed 600 14px
const CAP_FONT = [13, 0.62] as const // .net-cap: mono 13px
const LABEL_FONT = [22, 0.55] as const // .net-label
const SUB_FONT = [16, 0.5] as const // .net-sub
const SUBLAYER_FONT = [12, 0.55] as const // .net-sublayer
const LABEL_GAP = 8 // min clear space between two labels, or a label and a wire

/** Rough advance width of a label; the diagram reserves this much so text never overlaps. */
export function textWidth(text: string, px: number, em: number): number {
  return text.length * px * em
}

/** Slab height for a feature width: log scale so 128 and 201,088 both stay readable. */
export function slabHeight(dim: number): number {
  return Math.max(14, 10 * (Math.log2(dim) - 5))
}

/** Slab thickness for the producing op's parameter count: log scale, parameter-free ops are paper thin. */
export function slabThickness(params: number): number {
  return params <= 0 ? 3 : Math.min(22, Math.max(3, 1.8 * (Math.log2(params) - 15)))
}

interface SlabSpec { dim: number; params: number; kind: SlabKind; title: string }
interface PartSpec { op: string; slabs: SlabSpec[]; parallel?: boolean; captions: string[]; experts?: boolean }

export function networkLayout(c: GptOssConfig): NetworkLayout {
  const L = c.num_hidden_layers
  const H = c.hidden_size
  const I = c.intermediate_size
  const I2 = 2 * I
  const V = c.vocab_size
  const nh = c.num_attention_heads
  const q = nh * c.head_dim
  const kv = c.num_key_value_heads * c.head_dim
  const E = c.num_local_experts
  const k = c.num_experts_per_tok

  const groups: Group[] = []
  let x = PAD
  // Right edge of the last op slot / caption drawn per level (0 = branch line, 1 = stream).
  const opLane = [-Infinity, -Infinity]
  const capLane = [-Infinity, -Infinity]
  // Widest routed caption the view can show (expert ids joined by '·').
  const routedReserve = Array(k).fill(String(E - 1)).join('·')
  const act = (dim: number, name = '', times = 1) =>
    `${name ? `${name} ` : ''}T×${dim} bf16 · ${formatBytes(dim * 2)}/tok${times > 1 ? `, ×${times}` : ''}`

  /**
   * Lay parts out left to right from `from`, each preceded by a gap that carries its op label and weight column. A part
   * moves right when its op slot or captions would run into the previous one on the same level, or left of `capBound` (a wire).
   */
  const placeParts = (specs: PartSpec[], y: number, from: number, firstGap: number, capBound: number): Part[] => {
    const level = y === CY ? 1 : 0
    opLane[level] = Math.max(opLane[level], capBound)
    capLane[level] = Math.max(capLane[level], capBound)
    let prevRight = from
    x = from
    return specs.map((s, n) => {
      x += n === 0 ? firstGap : GAP
      let x0 = x
      const slabs = s.slabs.map((sl, i) => {
        const off = s.parallel ? i * STEP : 0
        const slab: Slab = { x: x + off, y: y - off, t: slabThickness(sl.params), h: slabHeight(sl.dim), kind: sl.kind, title: sl.title }
        if (!s.parallel) x += slab.t + 4
        return slab
      })
      let frontX1 = s.parallel ? x0 + slabs[0].t : x - 4
      let x1 = (s.parallel ? x0 + slabs[0].t + (slabs.length - 1) * STEP : x - 4) + DX
      const opW = s.op ? textWidth(s.op, ...OP_FONT) : 0
      const capLines = s.experts ? [...s.captions, `experts ${routedReserve}`] : s.captions
      const capW = Math.max(0, ...capLines.map((l) => textWidth(l, ...CAP_FONT)))
      // Moving x0 by d moves the op label (midway between prevRight and x0) by d / 2.
      const dOp = opW ? 2 * (opLane[level] + LABEL_GAP + opW / 2 - (prevRight + x0) / 2) : 0
      const dCap = capW ? capLane[level] + LABEL_GAP + capW / 2 - (x0 + x1) / 2 : 0
      const d = Math.max(0, dOp, dCap)
      for (const slab of slabs) slab.x += d
      x0 += d
      frontX1 += d
      x1 += d
      x = x1
      const opX = (prevRight + x0) / 2
      const capX = (x0 + x1) / 2
      if (opW) opLane[level] = opX + opW / 2
      if (capW) capLane[level] = capX + capW / 2
      const part: Part = {
        slabs, parallel: !!s.parallel, x0, frontX1, x1,
        op: s.op, opX, opY: level ? OP_Y_STREAM : OP_Y_BRANCH,
        captions: s.captions, captionY: level ? CAP_Y_STREAM : CAP_Y_BRANCH, experts: s.experts, opW, capW,
      }
      prevRight = frontX1
      return part
    })
  }

  /** A sublayer: tap the stream, run the parts along the branch line, rejoin at ⊕ clear of the captions and tag. */
  const branch = (label: string, attention: Branch['attention'], specs: PartSpec[], from: number): Branch => {
    const tapX = from + TAP
    const parts = placeParts(specs, BY, tapX, GAP, tapX + LABEL_GAP)
    const tagW = textWidth(label, ...SUBLAYER_FONT)
    const joinX = Math.max(x + 4 + JUNCTION_R, capLane[0] + LABEL_GAP + JUNCTION_R, opLane[0] + LABEL_GAP, tapX + tagW)
    x = joinX + JUNCTION_R
    return { label, attention, tapX, joinX, parts }
  }

  type Header = { id: string; kind: GroupKind; label: string; sub: string; layers: number[]; attention: 'sliding' | 'full' | null }
  const header = (id: string, kind: GroupKind, label: string, sub: string, layers: number[], attention: 'sliding' | 'full' | null): Header =>
    ({ id, kind, label, sub, layers, attention })

  const layer = layerMemory(c)

  /** Widen [x0, x1] so the group's label and subtitle fit; grow left into free space first, then right. */
  const fit = (x0: number, x1: number, g: Header): [number, number] => {
    const leftLimit = groups.length ? groups[groups.length - 1].x1 + 12 : 8
    const need = Math.max(textWidth(g.label, ...LABEL_FONT), textWidth(g.sub, ...SUB_FONT)) + LABEL_GAP
    const extra = need - (x1 - x0)
    if (extra <= 0) return [x0, x1]
    const left = Math.max(0, Math.min(extra / 2, x0 - leftLimit))
    return [x0 - left, x1 + extra - left]
  }

  const streamGroup = (g: Header, specs: PartSpec[]) => {
    const parts = placeParts(specs, CY, x, groups.length ? GROUP_GAP : 0, x)
    // Captions and weight columns may overhang the slabs; the group spans them too.
    const left = Math.min(parts[0].x0, ...parts.flatMap((p) => [p.opX - p.opW / 2, (p.x0 + p.x1) / 2 - p.capW / 2]))
    const right = Math.max(x, ...parts.flatMap((p) => [p.opX + p.opW / 2, (p.x0 + p.x1) / 2 + p.capW / 2]))
    const [x0, x1] = fit(left, right, g)
    x = x1
    groups.push({ ...g, x0, x1, parts, branches: [] })
  }

  const xn = (title: string): SlabSpec => ({ dim: H, params: H, kind: 'bf16', title })
  streamGroup(header('embed', 'embed', 'embedding', `token id → ${H}-d vector`, [], null), [
    { op: '', slabs: [{ dim: 1, params: 0, kind: 'input', title: 'token ids [T]' }], captions: ['T int64 · 8 B/tok'] },
    { op: 'token lookup', slabs: [{ dim: H, params: V * H, kind: 'bf16', title: `embed_tokens [${V}, ${H}] → x [T, ${H}]` }], captions: [act(H)] },
  ])
  const stream0 = groups[0].parts[1].frontX1

  const block = (i: number) => {
    const type = c.layer_types[i]
    const sliding = type === 'sliding_attention'
    const attnKind: SlabKind = sliding ? 'sliding' : 'full'
    const attnType = sliding ? `sliding ${c.sliding_window}` : 'full'
    const sub = sliding ? `last ${c.sliding_window} tokens` : 'all tokens'
    const attn: SlabSpec = { dim: q, params: nh, kind: attnKind, title: `attention (${attnType}): softmax(QKᵀ ‖ sink)·V → [T, ${q}]` }
    const perExpert = (f: (e: number) => SlabSpec) => Array.from({ length: k }, (_, e) => f(e))
    const attnSpecs: PartSpec[] = [
      { op: 'normalize', slabs: [xn(`input RMSNorm → [T, ${H}]`)], captions: [act(H)] },
      {
        op: 'query key value', parallel: true,
        slabs: [
          { dim: q, params: q * H, kind: 'bf16', title: `q_proj → [T, ${q}]` },
          { dim: kv, params: kv * H, kind: 'bf16', title: `k_proj → [T, ${kv}]` },
          { dim: kv, params: kv * H, kind: 'bf16', title: `v_proj → [T, ${kv}]` },
        ],
        captions: [act(q, 'q'), act(kv, 'k'), act(kv, 'v')],
      },
      { op: 'attend', slabs: [attn], captions: [act(q), `${nh} heads × ${c.head_dim}`] },
      { op: 'project back', slabs: [{ dim: H, params: H * q, kind: 'bf16', title: `o_proj → [T, ${H}]` }], captions: [act(H)] },
    ]
    const moeSpecs: PartSpec[] = [
      { op: 'normalize', slabs: [xn(`post-attention RMSNorm → [T, ${H}]`)], captions: [act(H)] },
      { op: 'pick experts', slabs: [{ dim: E, params: E * H, kind: 'bf16', title: `router → logits [T, ${E}], top-${k}, softmax` }], captions: [act(E)] },
      {
        op: `expand ×${k}`, parallel: true, experts: true,
        slabs: perExpert((e) => ({ dim: I2, params: H * I2 + I2, kind: 'mxfp4', title: `expert ${e + 1} of top-${k}: gate_up_proj → [T, ${I2}]` })),
        captions: [act(I2, '', k)],
      },
      {
        op: `gate ×${k}`, parallel: true,
        slabs: perExpert((e) => ({ dim: I, params: 0, kind: 'mxfp4', title: `expert ${e + 1}: clamped SwiGLU(gate, up) → [T, ${I}]` })),
        captions: [act(I, '', k)],
      },
      {
        op: `project down ×${k}`, parallel: true,
        slabs: perExpert((e) => ({ dim: H, params: I * H + H, kind: 'mxfp4', title: `expert ${e + 1}: down_proj → [T, ${H}]` })),
        captions: [act(H, '', k)],
      },
      { op: 'weighted sum', slabs: [{ dim: H, params: 0, kind: 'bf16', title: `Σ wᵢ · expertᵢ(x) → [T, ${H}]` }], captions: [act(H)] },
    ]
    x += GROUP_GAP
    const start = x
    const a = branch(`${sliding ? 'sliding' : 'full'} attention`, sliding ? 'sliding' : 'full', attnSpecs, x)
    const m = branch(`MoE · top-${k} of ${E} experts`, null, moeSpecs, x + 8)
    const g = header(`L${i}`, 'block', `L${i}`, sub, [i], sliding ? 'sliding' : 'full')
    const [x0, x1] = fit(start, x, g)
    x = x1
    groups.push({ ...g, x0, x1, parts: [], branches: [a, m] })
  }

  for (let i = 0; i < L; i++) block(i)

  streamGroup(header('head', 'head', 'lm_head', `${V.toLocaleString('en-US')} token scores`, [], null), [
    { op: 'normalize', slabs: [xn(`final RMSNorm → [T, ${H}]`)], captions: [act(H)] },
    { op: 'score vocab', slabs: [{ dim: V, params: V * H, kind: 'bf16', title: `lm_head → logits [T, ${V}]` }], captions: [act(V)] },
    { op: 'probabilities', slabs: [{ dim: V, params: 0, kind: 'output', title: `softmax → next-token probabilities [T, ${V}]` }], captions: [act(V)] },
  ])
  const head = groups[groups.length - 1]

  const ex = layer.rows.find((r) => r.id === 'experts')!
  const n = (x: number) => x.toLocaleString('en-US')
  const derivation = (g: Group): string => {
    if (g.kind === 'embed') {
      return `${g.label} weights: ${n(V)} tokens × ${n(H)} = ${formatCount(V * H)} params × 2 B (bf16) = ${formatBytes(V * H * 2)}`
    }
    if (g.kind === 'head') {
      return [
        `${g.label} weights, bf16 at 2 bytes per param:`,
        `${g.label} ${n(H)} × ${n(V)} = ${formatCount(V * H)} × 2 B = ${formatBytes(V * H * 2)}`,
        `+ final norm ${n(H)} × 2 B = ${formatBytes(H * 2)}`,
        `= ${formatBytes((V * H + H) * 2)}`,
      ].join('\n')
    }
    const bias = E * (expertParams(c) - expertMxfp4Params(c))
    const rows = layer.rows.map((r, i) => {
      const term = r.dtype === 'bf16'
        ? `${r.id} ${formatCount(r.params)} × 2 B = ${formatBytes(r.bytes)}`
        : `experts ${E} × ${formatCount(expertParams(c))}: weights ${formatCount(E * expertMxfp4Params(c))} × 4.25 bits = ${formatBytes(ex.bytes - bias * 2)}, biases ${formatCount(bias)} × 2 B = ${formatBytes(bias * 2)}`
      return (i === 0 ? '' : '+ ') + term
    })
    return [`${g.label} weights: bf16 at 2 bytes per param, MXFP4 at 4.25 bits`, ...rows, `= ${formatBytes(layer.bytes)}`].join('\n')
  }
  const units = groups.map((g) => {
    const bf16 = g.kind === 'embed' ? V * H * 2 : g.kind === 'head' ? (V * H + H) * 2 : layer.bytes - ex.bytes
    return { g, bf16, mxfp4: g.kind === 'block' ? ex.bytes : 0 }
  })
  const bars: WeightBar[] = units.map(({ g, bf16, mxfp4 }, i) => {
    const left = i === 0 ? g.x0 : (groups[i - 1].x1 + g.x0) / 2
    const right = i === groups.length - 1 ? g.x1 : (g.x1 + groups[i + 1].x0) / 2
    const xm = left + ((right - left) * bf16) / (bf16 + mxfp4)
    return {
      groupId: g.id,
      kind: g.attention ?? 'end',
      bf16: [left, xm],
      mxfp4: [xm, right],
      labelX: (g.x0 + g.x1) / 2,
      label: formatBytes(bf16 + mxfp4),
      title: derivation(g),
    }
  })

  return {
    groups,
    stream: [stream0, head.parts[0].x0 - 2],
    streamLabel: `residual stream x [T, ${H}]`,
    width: x + PAD,
    height: HEIGHT,
    tape: bars,
  }
}

/** Group drawn for a layer index, or the embedding (-1) / head (≥ L) flow steps. */
export function groupForStep(groups: Group[], step: number, L: number): Group | undefined {
  if (step < 0) return groups.find((g) => g.kind === 'embed')
  if (step >= L) return groups.find((g) => g.kind === 'head')
  return groups.find((g) => g.layers.includes(step))
}

/** The two groups outside the decoder stack. */
export type End = 'embed' | 'head'
/** What the diagram and sidebar can select: a decoder layer, or one of the ends. */
export type Target = number | End

export const isEnd = (s: string | null): s is End => s === 'embed' || s === 'head'
