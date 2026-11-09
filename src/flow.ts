export interface FlowState {
  playing: boolean
  /** null = idle, -1 = embedding, 0..L-1 = layers, L = final norm + lm_head. */
  step: number | null
}

/** Advance one step. Playing wraps from lm_head (= numLayers) back to the embedding and keeps going; a single step stops at lm_head. */
export function advanceFlow(f: FlowState, numLayers: number): FlowState {
  if (f.step === null) return { ...f, step: -1 }
  if (f.step >= numLayers) return f.playing ? { ...f, step: -1 } : f
  return { ...f, step: f.step + 1 }
}

/** Step back one position and pause; the embedding step (-1) is the earliest, idle is only reachable via reset. */
export function retreatFlow(f: FlowState): FlowState {
  return { ...f, playing: false, step: f.step === null ? null : Math.max(f.step - 1, -1) }
}

/** Play/pause; playing again after the token reached lm_head replays from the embedding, with no idle beat in between. */
export function togglePlay(f: FlowState, numLayers: number): FlowState {
  if (f.playing) return { ...f, playing: false }
  return { ...f, playing: true, step: f.step !== null && f.step >= numLayers ? -1 : f.step }
}

/** Where the token is, for the status line; layers use the same 0-based `L<i>` labels as the stack. */
export function flowStatus(step: number, numLayers: number): string {
  if (step === -1) return 'embedding'
  if (step >= numLayers) return 'lm_head'
  return `L${step} of ${numLayers}`
}
