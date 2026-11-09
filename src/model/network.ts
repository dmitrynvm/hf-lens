import type { GptOssConfig } from './config'
import { attentionParams, denseMlpParams, expertMxfp4Params, expertParams, formatBytes, formatCount, layerMemory, routerParams, sharedExpertParams, weightBytes, type Dtype } from './params'
import { weightDtype, type WeightRole } from './quant'

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

export interface PartWeights {
  bf16: number
  mxfp4: number
  /** Bytes read per token: the whole tensor, except unrouted experts and unused embedding rows. */
  active: number
  label: string
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
  /** Shape and dtype of the activations this op produces, one line each. */
  captions: string[]
  captionY: number
  /** Expert slabs: the view appends the routed expert ids under the captions once the token has passed. */
  experts?: boolean
  /** Weight bytes owned by this tensor step on the weight tape. */
  weights?: PartWeights
  /** Reserved label widths (px) used for spacing: op label above the slab and captions/weight size below. */
  opW: number
  capW: number
}
export interface Branch {
  tapX: number
  joinX: number
  parts: Part[]
}

export type GroupKind = 'embed' | 'block' | 'head'

export interface Group {
  id: string
  kind: GroupKind
  label: string
  sublabel?: string
  /** Attention type of a drawn block; null for other groups. */
  attention: 'sliding' | 'full' | null
  layers: number[]
  x0: number
  x1: number
  /** Parts sitting on the stream itself (embedding, head). */
  parts: Part[]
  branches: Branch[]
  /** Attention mask + expert grid drawn under a decoder block; absent for embed/head. */
  squares?: BlockSquares
}

/** Attention mask square; x/y is the top-left corner, `points` relative to it. */
export interface MaskSquare { x: number; y: number; side: number; kind: 'sliding' | 'full' | 'linear'; points: string; keys: number; shown: number; title: string }
/** One square cell per expert, row-major; the frame (x/y top-left, width × height) hugs the cells with no inner margin. `pitch` = cell + gap. Cells past `experts` fill the last row and are drawn dark. */
export interface ExpertGrid { x: number; y: number; width: number; height: number; cols: number; rows: number; pitch: number; gap: number; experts: number }
export interface BlockSquares { mask: MaskSquare; moe: { cx: number; caption: string; title: string; grid: ExpertGrid | null }; captionY: number }

export interface NetworkLayout {
  groups: Group[]
  /** The residual stream's x extent. */
  stream: [number, number]
  streamLabel: string
  width: number
  height: number
  /** One weight-tape segment per weighted tensor step, left to right, touching with no gap from the embedding's x0 to lm_head's x1. */
  tape: WeightBar[]
  /** Weight tape drawn under the diagram: one bar per group, coloured by attention type. */
  blockTape: TapeRow
}

export type TapeKind = 'sliding' | 'full' | 'end'
export interface TapeSegment { id: string; groupId: string; x0: number; x1: number; kind: TapeKind; labelX: number; label: string; title: string }
export interface TapeRow { barY0: number; barY1: number; labelY: number; segments: TapeSegment[] }

/** One tensor step's stretch of the weight tape, split by its bf16 / mxfp4 bytes and labelled with its byte size directly below the slab. */
export interface WeightBar {
  id: string
  groupId: string
  kind: 'sliding' | 'full' | 'end'
  bf16: [number, number]
  mxfp4: [number, number]
  bf16Bytes: number
  mxfp4Bytes: number
  /** Bytes read per token. */
  active: number
  labelX: number
  label: string
  /** Hover text: how the size label is computed, term by term, ending "= <label>". */
  title: string
}

export const LABEL_Y = 17 // group label (18px), top of the diagram
export const SUBLABEL_Y = 30 // block attention kind under the group label
export const BRACKET_Y = 36 // bracket under the header, ticks pointing down into the group
export const OP_Y_BRANCH = 52
export const BY = 108 // branch centre line
export const CAP_PITCH = 12 // caption line spacing; branch parts caption at most two lines
// The tallest slab (slabHeight caps at 68) spans ±(34 + DY/2) around its line; captions start 17 below that.
export const CAP_Y_BRANCH = BY + 56 // first caption line under branch slabs
export const CY = CAP_Y_BRANCH + CAP_PITCH + 12 // residual stream, just under the branch captions' second line
export const OP_Y_STREAM = CY - 56
export const CAP_Y_STREAM = CY + 56
export const SQ_TOP = CY + 28 // top edge of the per-block squares, just clear of the residual-stream label (baseline CY + 20)
export const SQ_SIDE = 120 // mask and expert-grid side: fills the narrowest attention branch (gpt-oss, 137.7 wide) with the 6px sink on its left, and the narrowest MoE branch (gpt-oss-120b, 131 wide)
export const GROUP_BG_BOTTOM = SQ_TOP + SQ_SIDE + 20 // selection background covers the squares and their captions
export const BAR_H = 18 // tall enough to hold its size label inside
export const TAPE_TOP = GROUP_BG_BOTTOM + 6
export const TAPE_H = BAR_H + 6
export const HEIGHT = TAPE_TOP + TAPE_H
export const DEPTH = 20
export const DX = DEPTH * 0.5
export const DY = DEPTH * 0.5
export const JUNCTION_R = 9
export const STEP = 5 // depth offset between parallel slabs
const GAP = 6
const GROUP_GAP = 8
const TAP = 4
const PAD = 8
const OP_FONT = [11, 0.56] as const // .net-op: Barlow Semi Condensed 600 11px
const CAP_FONT = [10.5, 0.62] as const // .net-cap: mono 10.5px
const LABEL_FONT = [18, 0.55] as const // .net-label
const LABEL_GAP = 6 // min clear space between two labels, or a label and a wire
/** Rough advance width of a label; the diagram reserves this much so text never overlaps. */
export function textWidth(text: string, px: number, em: number): number {
  return text.length * px * em
}

/** Slab height for a feature width: log scale so 128 and 201,088 both stay readable. */
export function slabHeight(dim: number): number {
  return Math.min(68, Math.max(12, 7.5 * (Math.log2(dim) - 5)))
}

/** Slab thickness for the producing op's parameter count: log scale, parameter-free ops are paper thin. */
export function slabThickness(params: number): number {
  return params <= 0 ? 3 : Math.min(22, Math.max(3, 1.8 * (Math.log2(params) - 15)))
}

interface SlabSpec { dim: number; params: number; kind: SlabKind; title: string }
interface WeightSpec { bf16: number; mxfp4?: number; /** Bytes read per token; defaults to the whole tensor. */ active?: number; title: string }
interface PartSpec { op: string; slabs: SlabSpec[]; parallel?: boolean; captions: string[]; experts?: boolean; weights?: WeightSpec }
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
      const weights: PartWeights | undefined = s.weights && {
        bf16: s.weights.bf16,
        mxfp4: s.weights.mxfp4 ?? 0,
        active: s.weights.active ?? s.weights.bf16 + (s.weights.mxfp4 ?? 0),
        label: formatBytes(s.weights.bf16 + (s.weights.mxfp4 ?? 0)),
        title: s.weights.title,
      }
      const opW = s.op ? textWidth(s.op, ...OP_FONT) : 0
      const capLines = s.captions
      const weightW = weights ? textWidth(weights.label, ...CAP_FONT) : 0
      const capW = Math.max(weightW, ...capLines.map((l) => textWidth(l, ...CAP_FONT)))
      const dOp = opW ? opLane[level] + LABEL_GAP + opW / 2 - (x0 + x1) / 2 : 0
      const dCap = capW ? capLane[level] + LABEL_GAP + capW / 2 - (x0 + x1) / 2 : 0
      const d = Math.max(0, dOp, dCap)
      for (const slab of slabs) slab.x += d
      x0 += d
      frontX1 += d
      x1 += d
      x = x1
      const midX = (x0 + x1) / 2
      if (opW) opLane[level] = midX + opW / 2
      if (capW) capLane[level] = midX + capW / 2
      return {
        slabs, parallel: !!s.parallel, x0, frontX1, x1,
        op: s.op, opX: midX, opY: level ? OP_Y_STREAM : OP_Y_BRANCH,
        captions: s.captions, captionY: level ? CAP_Y_STREAM : CAP_Y_BRANCH, experts: s.experts, weights, opW, capW,
      }
    })
  }

  /** A sublayer: tap the stream, run the parts along the branch line, rejoin at ⊕ clear of the captions and op labels. */
  const branch = (specs: PartSpec[], from: number): Branch => {
    const tapX = from + TAP
    const parts = placeParts(specs, BY, tapX, GAP, tapX)
    const joinX = Math.max(x + 3 + JUNCTION_R, capLane[0] + LABEL_GAP + JUNCTION_R, opLane[0] + LABEL_GAP)
    x = joinX + JUNCTION_R
    return { tapX, joinX, parts }
  }

  type Header = { id: string; kind: GroupKind; label: string; sublabel?: string; layers: number[]; attention: 'sliding' | 'full' | null }
  const header = (id: string, kind: GroupKind, label: string, layers: number[], attention: 'sliding' | 'full' | null, sublabel?: string): Header =>
    ({ id, kind, label, sublabel, layers, attention })

  const layer = layerMemory(c)

  /** Widen [x0, x1] so the group's label and sublabel fit; grow left into free space first, then right. */
  const fit = (x0: number, x1: number, g: Header): [number, number] => {
    const leftLimit = groups.length ? groups[groups.length - 1].x1 + 8 : 6
    const subW = g.sublabel ? textWidth(g.sublabel, ...OP_FONT) : 0
    const need = Math.max(textWidth(g.label, ...LABEL_FONT), subW) + LABEL_GAP
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

  const n = (v: number) => v.toLocaleString('en-US')
  const ex = layer.rows.find((r) => r.id === 'experts')!
  const xn = (title: string): SlabSpec => ({ dim: H, params: H, kind: 'bf16', title: `${title} (${act(H)})` })
  // Storage dtypes come from quantization_config (see quant.ts): slab colour, caption and weight-tape split follow them.
  const dt = (role: WeightRole) => weightDtype(c, role)
  const PER: Record<Dtype, string> = { bf16: '2 B', mxfp4: '4.25 bits' }
  /** Tape bytes and hover text for `params` weights of one role. */
  const wt = (role: WeightRole, params: number, what: string): WeightSpec => {
    const d = dt(role)
    const b = weightBytes(d, params)
    return { bf16: d === 'bf16' ? b : 0, mxfp4: d === 'mxfp4' ? b : 0, title: `${what}: ${formatCount(params)} params × ${PER[d]} (${d}) = ${formatBytes(b)}` }
  }
  const xd = dt('experts')
  /** Routed expert matrices in the experts' dtype plus their bf16 biases; a token reads k of E. */
  const expertWeights = (mx: number, bias: number, what: string): WeightSpec => {
    const mb = weightBytes(xd, mx)
    const bb = bias * 2
    return {
      bf16: bb + (xd === 'bf16' ? mb : 0), mxfp4: xd === 'mxfp4' ? mb : 0, active: ((mb + bb) * k) / E,
      title: `${what}: ${formatCount(mx)} × ${PER[xd]} (${xd}) = ${formatBytes(mb)}, biases ${formatCount(bias)} × 2 B = ${formatBytes(bb)} = ${formatBytes(mb + bb)}`,
    }
  }
  streamGroup(header('embed', 'embed', 'Input', [], null), [
    { op: 'tokens', slabs: [{ dim: 1, params: 0, kind: 'input', title: 'token ids [T] · int64 · 8 B/tok' }], captions: ['1', 'int64'] },
    {
      op: 'Embed',
      slabs: [{ dim: H, params: V * H, kind: dt('embed'), title: `embed_tokens [${V}, ${H}] → x [T, ${H}] (${act(H)})` }],
      captions: [`${H}`, dt('embed')],
      weights: { ...wt('embed', V * H, `embedding weights (${n(V)} tokens × ${n(H)})`), active: weightBytes(dt('embed'), H) },
    },
  ])
  const stream0 = groups[0].parts[1].frontX1

  const qkvParams = H * q + q + 2 * (H * kv + kv)
  const oParams = q * H + H
  const rParams = routerParams(c, c.first_k_dense_replace ?? 0)
  const upMx = E * H * I2
  const upBias = E * I2
  const downMx = E * I * H
  const downBias = E * H
  const totalBias = E * (expertParams(c) - expertMxfp4Params(c))
  const sharedParams = sharedExpertParams(c)
  /** One combined MoE column: router, routed experts (+ bf16 biases) and shared experts, each in its config dtype. */
  const latentWeights = (i: number): WeightSpec => {
    const rd = dt('router')
    const sd = dt('shared_experts')
    const rB = weightBytes(rd, rParams)
    const sB = weightBytes(sd, sharedParams)
    const xB = weightBytes(xd, E * expertMxfp4Params(c))
    const bB = totalBias * 2
    const of = (d: Dtype) => (rd === d ? rB : 0) + (sd === d ? sB : 0) + (xd === d ? xB : 0)
    const exDtypes = [...new Set([xd, ...(sharedParams ? [sd] : []), ...(totalBias ? ['bf16'] : [])])].join(' + ')
    return {
      bf16: of('bf16') + bB, mxfp4: of('mxfp4'), active: rB + sB + ((xB + bB) * k) / E,
      title: `L${i} MoE (router + ${E} experts): router ${formatBytes(rB)} (${rd}) + experts ${formatBytes(ex.bytes)} (${exDtypes}) = ${formatBytes(rB + ex.bytes)}`,
    }
  }
  const ad = dt('attention')
  const normHalf = H + (c.attn_res_block_size ? 2 * H : 0)
  const slabCount = Math.min(k, 4)

  const block = (i: number) => {
    const type = c.layer_types[i]
    const sliding = type === 'sliding_attention' || type === 'linear_attention'
    const attnKind: SlabKind = sliding ? 'sliding' : 'full'
    const attnType = type === 'linear_attention' ? 'KDA' : sliding ? 'SWA' : c.kv_lora_rank ? 'MLA' : 'FA'
    const aParams = attentionParams(c, i)
    const isDense = i < (c.first_k_dense_replace ?? 0)
    const attn: SlabSpec = { dim: q, params: nh, kind: attnKind, title: `attention (${attnType}): → [T, ${q}] (${act(q)})` }
    const perExpert = (f: (e: number) => SlabSpec) => Array.from({ length: slabCount }, (_, e) => f(e))
    const full = i === 0
    const attnNorm: PartSpec = {
      op: 'RMSNorm',
      slabs: [xn(`input RMSNorm → [T, ${H}]`)],
      captions: [`${H}`, 'bf16'],
      weights: { bf16: normHalf * 2, title: `L${i} input RMSNorm: ${n(normHalf)} params × 2 B (bf16) = ${formatBytes(normHalf * 2)}` },
    }
    const moeNorm: PartSpec = {
      op: 'RMSNorm',
      slabs: [xn(`post-attention RMSNorm → [T, ${H}]`)],
      captions: [`${H}`, 'bf16'],
      weights: { bf16: normHalf * 2, title: `L${i} post-attention RMSNorm: ${n(normHalf)} params × 2 B (bf16) = ${formatBytes(normHalf * 2)}` },
    }
    const qkvFullParams = type === 'linear_attention' ? 3 * H * q + 3 * q * (c.short_conv_kernel_size ?? 4) : qkvParams
    const oFullParams = type === 'linear_attention' ? H * q + c.head_dim : oParams
    const midAttnParams = type === 'linear_attention' ? aParams - qkvFullParams - oFullParams : nh
    const attnSpecs: PartSpec[] = full
      ? [
          attnNorm,
          {
            op: 'q·k·v', parallel: true,
            slabs: [
              { dim: q, params: q * H, kind: ad, title: `q_proj → [T, ${q}] (${act(q, 'q')})` },
              { dim: kv, params: kv * H, kind: ad, title: `k_proj → [T, ${kv}] (${act(kv, 'k')})` },
              { dim: kv, params: kv * H, kind: ad, title: `v_proj → [T, ${kv}] (${act(kv, 'v')})` },
            ],
            captions: [`${nh}q·${c.num_key_value_heads}kv×${c.head_dim}`, ad],
            weights: wt('attention', qkvFullParams, `L${i} q/k/v projections`),
          },
          {
            op: attnType, slabs: [attn], captions: [`${nh}h×${c.head_dim}`, ad],
            weights: wt('attention', midAttnParams, `L${i} ${attnType} state/sinks`),
          },
          {
            op: 'o_proj', slabs: [{ dim: H, params: H * q, kind: ad, title: `o_proj → [T, ${H}] (${act(H)})` }], captions: [`${H}`, ad],
            weights: wt('attention', oFullParams, `L${i} o_proj`),
          },
        ]
      : [
          attnNorm,
          {
            op: attnType, slabs: [attn], captions: [`${nh}h×${c.head_dim}`, ad],
            weights: wt('attention', aParams, `L${i} attention (${attnType})`),
          },
        ]
    const D = c.dense_intermediate_size ?? I
    const dd = dt('dense_mlp')
    const moeSpecs: PartSpec[] = isDense
      ? full
        ? [
            moeNorm,
            {
              op: 'gate_up',
              slabs: [{ dim: 2 * D, params: 2 * H * D, kind: dd, title: `dense gate_proj + up_proj → [T, ${2 * D}] (${act(2 * D)})` }],
              captions: [`${2 * D}`, dd],
              weights: wt('dense_mlp', 2 * H * D, `L${i} gate_up_proj (dense)`),
            },
            {
              op: c.hidden_act === 'situ' ? 'SiTU' : 'SwiGLU',
              slabs: [{ dim: D, params: 0, kind: 'bf16', title: `dense ${c.hidden_act === 'situ' ? 'SiTU' : 'SwiGLU'} → [T, ${D}] (${act(D)})` }],
              captions: [`${D}`, 'bf16'],
            },
            {
              op: 'down',
              slabs: [{ dim: H, params: H * D, kind: dd, title: `dense down_proj → [T, ${H}] (${act(H)})` }],
              captions: [`${H}`, dd],
              weights: wt('dense_mlp', H * D, `L${i} down_proj (dense)`),
            },
          ]
        : [
            moeNorm,
            {
              op: 'MLP',
              slabs: [{ dim: 2 * D, params: denseMlpParams(c), kind: dd, title: `dense MLP [T, ${H}] → [T, ${D}] → [T, ${H}]` }],
              captions: [`${D}`, dd],
              weights: wt('dense_mlp', denseMlpParams(c), `L${i} dense MLP`),
            },
          ]
      : full
        ? [
            moeNorm,
            {
              op: 'router', slabs: [{ dim: E, params: E * H, kind: dt('router'), title: `router → logits [T, ${E}], top-${k}, softmax (${act(E)})` }], captions: [`${k}/${E}`, dt('router')],
              weights: wt('router', rParams, `L${i} router`),
            },
            {
              op: 'gate_up', parallel: true, experts: true,
              slabs: perExpert((e) => ({ dim: I2, params: H * I2 + I2, kind: xd, title: `expert ${e + 1} of top-${k}: gate_up_proj → [T, ${I2}] (${act(I2, '', k)})` })),
              captions: [`${I2}`, xd],
              weights: expertWeights(upMx, upBias, `L${i} gate_up_proj (${E} experts)`),
            },
            {
              op: 'SwiGLU', parallel: true,
              slabs: perExpert((e) => ({ dim: I, params: 0, kind: xd, title: `expert ${e + 1}: clamped SwiGLU(gate, up) → [T, ${I}] (${act(I, '', k)})` })),
              captions: [`${I}`, 'bf16'],
            },
            {
              op: 'down', parallel: true,
              slabs: perExpert((e) => ({ dim: H, params: I * H + H, kind: xd, title: `expert ${e + 1}: down_proj → [T, ${H}] (${act(H, '', k)})` })),
              captions: [`${H}`, xd],
              weights: expertWeights(downMx, downBias, `L${i} down_proj (${E} experts)`),
            },
            { op: 'sum', slabs: [{ dim: H, params: 0, kind: 'bf16', title: `Σ wᵢ · expertᵢ(x) → [T, ${H}] (${act(H)})` }], captions: [`${H}`, 'bf16'] },
          ]
        : [
            moeNorm,
            {
              op: c.routed_expert_hidden_size ? 'LatentMoE' : 'MoE', parallel: true, experts: true,
              slabs: perExpert((e) => ({ dim: I2, params: H * I2 + I2, kind: xd, title: `expert ${e + 1} of top-${k}: MLP [T, ${H}] → [T, ${I2}] → [T, ${H}]` })),
              captions: [`${I2}`, xd],
              weights: latentWeights(i),
            },
          ]
    x += GROUP_GAP
    const start = x
    const a = branch(attnSpecs, x)
    const m = branch(moeSpecs, x + 4)
    const sublabel = type === 'linear_attention' ? 'Linear Attention' : sliding ? 'Sliding Attention' : 'Full Attention'
    const g = header(`L${i}`, 'block', `L${i}`, [i], sliding ? 'sliding' : 'full', sublabel)
    const [x0, x1] = fit(start, x, g)
    x = x1
    const w = c.sliding_window
    const mask: MaskSquare = {
      x: (a.tapX + a.joinX) / 2 - SQ_SIDE / 2,
      y: SQ_TOP,
      side: SQ_SIDE,
      kind: type === 'linear_attention' ? 'linear' : sliding ? 'sliding' : 'full',
      points: type === 'sliding_attention'
        ? `0,0 ${SQ_SIDE},${SQ_SIDE} ${0.75 * SQ_SIDE},${SQ_SIDE} 0,${0.25 * SQ_SIDE}`
        : `0,0 ${SQ_SIDE},${SQ_SIDE} 0,${SQ_SIDE}`,
      keys: type === 'sliding_attention' ? w : 4 * w, shown: 4 * w, // keys a query sees / tokens shown
      title: type === 'linear_attention'
        ? `L${i} Kimi Delta Attention: every earlier key feeds a ${c.head_dim}×${c.head_dim} recurrent state per head, fading by the cumulative per-channel decay ∏α; no window, no sink (${4 * w} tokens shown)`
        : sliding
          ? `L${i} sliding window: each query sees the previous ${w} tokens plus the sink (${4 * w} tokens shown)`
          : `L${i} full causal attention: each query sees every earlier token plus the sink (${4 * w} tokens shown)`,
    }
    const moeCx = (m.tapX + m.joinX) / 2
    const cols = Math.ceil(Math.sqrt(E))
    const rows = Math.ceil(E / cols)
    const pitch = SQ_SIDE / cols
    const gap = pitch * 0.2
    const moe = isDense
      ? { cx: moeCx, caption: '1/1', title: `L${i} dense MLP: every token uses the same weights, no routing`, grid: null }
      : {
          cx: moeCx,
          caption: `${k}/${E}`,
          title: `L${i} experts (simulated router): top-${k} of ${E}`,
          grid: {
            x: moeCx - SQ_SIDE / 2, y: SQ_TOP, width: SQ_SIDE, height: rows * pitch,
            cols, rows, pitch, gap, experts: E,
          },
        }
    groups.push({ ...g, x0, x1, parts: [], branches: [a, m], squares: { mask, moe, captionY: SQ_TOP + SQ_SIDE + 14 } })
  }

  for (let i = 0; i < L; i++) block(i)

  streamGroup(header('head', 'head', 'Output', [], null), [
    {
      op: 'RMSNorm', slabs: [xn(`final RMSNorm → [T, ${H}]`)], captions: [`${H}`, 'bf16'],
      weights: { bf16: H * 2, title: `final RMSNorm: ${n(H)} params × 2 B (bf16) = ${formatBytes(H * 2)}` },
    },
    {
      op: 'Unembed', slabs: [{ dim: V, params: V * H, kind: dt('lm_head'), title: `lm_head → logits [T, ${V}] (${act(V)})` }], captions: [`${V}`, dt('lm_head')],
      weights: wt('lm_head', V * H, `lm_head (${n(H)} × ${n(V)})`),
    },
    { op: 'Softmax', slabs: [{ dim: V, params: 0, kind: 'output', title: `softmax → next-token probabilities [T, ${V}] (${act(V)})` }], captions: [`${V}`, 'bf16'] },
  ])
  const head = groups[groups.length - 1]

  const weighted = groups.flatMap((g, gi) => {
    const parts = [...g.parts, ...g.branches.flatMap((b) => b.parts)].filter((p): p is Part & { weights: PartWeights } => !!p.weights)
    return parts.map((p, pi) => {
      const cx = (p.x0 + p.x1) / 2
      const half = textWidth(p.weights.label, ...CAP_FONT) / 2
      return { g, gi, p, pi, cx, lEdge: cx - half, rEdge: cx + half }
    })
  })
  const bars: WeightBar[] = weighted.map((w, idx) => {
    const prev = idx > 0 ? weighted[idx - 1] : null
    const next = idx < weighted.length - 1 ? weighted[idx + 1] : null
    const left = !prev
      ? groups[0].x0
      : prev.gi !== w.gi
        ? (groups[prev.gi].x1 + w.g.x0) / 2
        : (prev.rEdge + w.lEdge) / 2
    const right = !next
      ? groups.at(-1)!.x1
      : next.gi !== w.gi
        ? (w.g.x1 + groups[next.gi].x0) / 2
        : (w.rEdge + next.lEdge) / 2
    const { bf16, mxfp4, active, label, title } = w.p.weights
    const xm = left + ((right - left) * bf16) / (bf16 + mxfp4)
    return {
      id: `${w.g.id}-${w.pi}`,
      groupId: w.g.id,
      kind: w.g.attention ?? 'end',
      bf16: [left, xm],
      mxfp4: [xm, right],
      bf16Bytes: bf16,
      mxfp4Bytes: mxfp4,
      active,
      labelX: w.cx,
      label,
      title,
    }
  })

  const groupSpan = (id: string): [number, number] => {
    const own = bars.filter((b) => b.groupId === id)
    return [Math.min(...own.map((b) => b.bf16[0])), Math.max(...own.map((b) => b.mxfp4[1]))]
  }
  const fits = (label: string, w: number) => textWidth(label, ...CAP_FONT) <= w - 4
  const row = (build: () => TapeSegment[]): TapeRow => {
    const barY0 = TAPE_TOP
    const barY1 = barY0 + BAR_H
    return { barY0, barY1, labelY: (barY0 + barY1) / 2, segments: build() }
  }

  const blockTape = row(() =>
    groups.map((g, i) => {
      const [x0, x1] = groupSpan(g.id)
      const own = bars.filter((b) => b.groupId === g.id)
      const bf = own.reduce((s, b) => s + b.bf16Bytes, 0)
      const mx = own.reduce((s, b) => s + b.mxfp4Bytes, 0)
      const perTok = own.reduce((s, b) => s + b.active, 0)
      const full = `${formatBytes(bf + mx)} · ${formatBytes(perTok)}/tok`
      return {
        id: `block-${i}`, groupId: g.id, x0, x1, kind: g.attention ?? 'end', labelX: (x0 + x1) / 2,
        label: fits(full, x1 - x0) ? full : formatBytes(bf + mx),
        title: `${g.label}: ${formatBytes(bf + mx)} weights (${formatBytes(bf)} bf16 + ${formatBytes(mx)} mxfp4), ${formatBytes(perTok)} read per token`,
      }
    }))
  return {
    groups,
    stream: [stream0, head.parts[0].x0 - 2],
    streamLabel: `residual stream (${H})`,
    width: x + PAD,
    height: HEIGHT,
    tape: bars,
    blockTape,
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
