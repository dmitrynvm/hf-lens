import { useEffect, useRef, type Dispatch, type SetStateAction } from 'react'
import { advanceFlow, restartFlow, retreatFlow, togglePlay, type FlowState } from '../flow'
import type { GptOssConfig } from '../model/config'
import { Icon } from './Icon'

interface Props {
  flow: FlowState
  setFlow: Dispatch<SetStateAction<FlowState>>
  config: GptOssConfig
}

export function TokenFlowControls({ flow, setFlow, config: c }: Props) {
  const numLayers = c.num_hidden_layers
  const step = flow.step
  const done = step !== null && step >= numLayers
  // Space anywhere plays/pauses, as in capalo. Ignored in form fields, and on a button or link reached by
  // keyboard (Space activates it); after a mouse click the focused button does not count.
  const pointerFocus = useRef(false)
  useEffect(() => {
    const onPointer = () => void (pointerFocus.current = true)
    const onTab = (e: KeyboardEvent) => void (e.key === 'Tab' && (pointerFocus.current = false))
    window.addEventListener('pointerdown', onPointer, true)
    window.addEventListener('keydown', onTab, true)
    return () => {
      window.removeEventListener('pointerdown', onPointer, true)
      window.removeEventListener('keydown', onTab, true)
    }
  }, [])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.code !== 'Space' && e.key !== 'Enter') || e.repeat || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return
      const t = e.target as HTMLElement | null
      if (t?.closest('input, textarea, [contenteditable]')) return
      if (t?.closest('button, a[href], summary, label, [role="button"], [role="tab"]') && !pointerFocus.current) return
      e.preventDefault() // no page scroll, no click on the mouse-focused button or select
      if (e.key === 'Enter') {
        setFlow(restartFlow)
      } else {
        setFlow((f) => togglePlay(f, numLayers))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [setFlow, numLayers])

  return (
    <div className="form-actions-group" role="group" aria-label="Token flow">
      <button
        type="button"
        className="button-secondary button-icon-only"
        aria-label="Step back"
        title="Step back"
        onClick={() => setFlow(retreatFlow)}
        disabled={step === null || step <= -1}
      >
        <Icon name="back" />
      </button>
      <button
        type="button"
        className="button-primary button-icon-only"
        aria-keyshortcuts="Space"
        aria-label={flow.playing ? 'Pause' : done ? 'Replay' : 'Play'}
        title={`${flow.playing ? 'Pause' : done ? 'Replay' : 'Play'} (Space)`}
        onClick={() => setFlow((f) => togglePlay(f, numLayers))}
      >
        <Icon name={flow.playing ? 'pause' : 'play'} />
      </button>
      <button
        type="button"
        className="button-secondary button-icon-only"
        aria-label="Step forward"
        title="Step forward"
        onClick={() => setFlow((f) => ({ ...advanceFlow(f, numLayers), playing: false }))}
        disabled={done}
      >
        <Icon name="forward" />
      </button>
      <button
        type="button"
        className="button-secondary button-icon-only"
        aria-keyshortcuts="Enter"
        aria-label="Reset"
        title="Reset (Enter)"
        onClick={() => setFlow(restartFlow)}
      >
        <Icon name="reset" />
      </button>
    </div>
  )
}
