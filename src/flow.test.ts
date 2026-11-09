import { describe, expect, it } from 'vitest'
import { advanceFlow, retreatFlow, togglePlay, type FlowState } from './flow'

const L = 36
const base: FlowState = { playing: true, step: null }

describe('advanceFlow', () => {
  it('starts at the embedding step from idle', () => {
    expect(advanceFlow(base, L)).toMatchObject({ step: -1, playing: true })
  })

  it('keeps playing through the last layer', () => {
    expect(advanceFlow({ ...base, step: 34 }, L)).toMatchObject({ step: 35, playing: true })
  })

  it('reaches lm_head and keeps playing', () => {
    expect(advanceFlow({ ...base, step: 35 }, L)).toMatchObject({ step: L, playing: true })
  })

  it('wraps to the embedding when playing past lm_head', () => {
    expect(advanceFlow({ ...base, step: L }, L)).toMatchObject({ step: -1, playing: true })
  })

  it('a single step holds at lm_head', () => {
    expect(advanceFlow({ ...base, playing: false, step: L }, L)).toMatchObject({ step: L, playing: false })
  })

  it('single-steps without starting playback', () => {
    expect(advanceFlow({ ...base, playing: false, step: 3 }, L)).toMatchObject({ step: 4, playing: false })
  })
})

describe('retreatFlow', () => {
  it('steps back one layer and pauses', () => {
    expect(retreatFlow({ ...base, step: 5 })).toMatchObject({ step: 4, playing: false })
  })

  it('steps back from lm_head to the last layer', () => {
    expect(retreatFlow({ ...base, step: L })).toMatchObject({ step: L - 1, playing: false })
  })

  it('stops at the embedding step and leaves idle untouched', () => {
    expect(retreatFlow({ ...base, step: -1 })).toMatchObject({ step: -1 })
    expect(retreatFlow({ ...base, step: null })).toMatchObject({ step: null, playing: false })
  })
})

describe('togglePlay', () => {
  it('pauses in place and resumes from the same step', () => {
    const paused = togglePlay({ ...base, step: 7 }, L)
    expect(paused).toMatchObject({ step: 7, playing: false })
    expect(togglePlay(paused, L)).toMatchObject({ step: 7, playing: true })
  })

  it('replays from the embedding once the token reached lm_head', () => {
    expect(togglePlay({ ...base, playing: false, step: L }, L)).toMatchObject({ step: -1, playing: true })
  })
})
