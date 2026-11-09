import { useCallback, useEffect, useMemo, useState } from 'react'
import { AddModel } from './components/AddModel'
import { AttentionPatterns } from './components/AttentionPatterns'
import { BlockDetail } from './components/BlockDetail'
import { ConfigView } from './components/ConfigView'
import { LayerDetail, LayerRod } from './components/LayerStack'
import { MainTabs } from './components/MainTabs'
import { ModelSummary } from './components/ModelSummary'
import { NetworkView } from './components/NetworkView'
import { TokenFlowControls } from './components/TokenFlowControls'
import { TensorTypePicker } from './components/TensorTypePicker'
import { DEFAULT_TENSOR_TYPE, tensorTypes } from './model/safetensors'
import { advanceFlow, flowStatus, STEP_MS, STEP_MS_MAX, STEP_MS_MIN, type FlowState } from './flow'
import { BUILTIN_MODELS, DEFAULT_MODEL, findModel, type ModelEntry } from './model/config'
import { loadCustomModels, saveCustomModels } from './model/hub'
import { headerFacts } from './model/facts'
import { isEnd, type End, type Target } from './model/network'
import { paramBreakdown } from './model/params'
import { simulateRouting } from './model/routing'
import { isMainView, MAIN_VIEWS, type MainView } from './views'

interface HashState {
  /** `ModelEntry.id` of a listed model. */
  model: string
  view: MainView
  layer: number
  /** Selected end of the network; `layer` stays the decoder layer the other tabs show. */
  end: End | null
}

/**
 * `model` is a bundled id (`120b`) or the repo id of a model added from Hugging Face in this browser.
 * Invalid or missing values fall back silently.
 */
function parseHash(models: readonly ModelEntry[]): HashState {
  const p = new URLSearchParams(location.hash.slice(1))
  const entry = findModel(models, p.get('model')) ?? findModel(models, DEFAULT_MODEL)!
  const model = entry.id
  const v = p.get('view')
  const layerParam = p.get('layer')
  const layer = layerParam !== null && /^\d+$/.test(layerParam) ? Number(layerParam) : NaN
  return {
    model,
    view: isMainView(v) ? v : MAIN_VIEWS[0].id,
    layer: layer < entry.config.num_hidden_layers ? layer : 0,
    end: isEnd(layerParam) ? layerParam : null,
  }
}

const hashOf = (s: HashState) => `#model=${s.model}&view=${s.view}&layer=${s.end ?? s.layer}`

/** Simulated routing is seeded so every load, link and screenshot shows the same experts. */
const ROUTING_SEED = 42

export default function App() {
  const [models, setModels] = useState<readonly ModelEntry[]>(() => [...BUILTIN_MODELS, ...loadCustomModels()])
  const [initial] = useState(() => parseHash(models))
  const [model, setModel] = useState(initial.model)
  const [view, setView] = useState(initial.view)
  const [selectedLayer, setSelectedLayer] = useState(initial.layer)
  const [selectedEnd, setSelectedEnd] = useState(initial.end)
  const selected: Target = selectedEnd ?? selectedLayer
  const select = useCallback((t: Target) => {
    if (typeof t === 'number') {
      setSelectedLayer(t)
      setSelectedEnd(null)
    } else {
      setSelectedEnd(t)
    }
  }, [])
  const [flow, setFlow] = useState<FlowState>({ playing: false, step: null })
  const [stepMs, setStepMs] = useState(STEP_MS)
  const entry = findModel(models, model)!
  const config = entry.config
  const L = config.num_hidden_layers
  const name = entry.name
  /** Picking another model in the sidebar starts over at its first block; a hash link keeps the layer it names. */
  const switchModel = (id: string) => {
    setModel(id)
    select(0)
  }
  const addModel = (m: ModelEntry) => {
    if (!findModel(models, m.id)) {
      const next = [...models, m]
      setModels(next)
      saveCustomModels(next.filter((x) => !BUILTIN_MODELS.includes(x)))
    }
    switchModel(m.id)
  }
  const breakdown = useMemo(() => paramBreakdown(config), [config])
  const types = useMemo(() => tensorTypes(entry), [entry])
  const [tensorType, setTensorType] = useState(DEFAULT_TENSOR_TYPE)
  // BF16 unless the user picked another type of this model (a model without BF16 falls back to its first type).
  const tensorRow = types.find((r) => r.dtype === tensorType) ?? types[0]
  const routes = useMemo(() => simulateRouting(config, ROUTING_SEED), [config])

  useEffect(() => {
    if (!flow.playing) return
    const id = setInterval(() => setFlow((f) => advanceFlow(f, L)), stepMs)
    return () => clearInterval(id)
  }, [flow.playing, L, stepMs])

  useEffect(() => {
    history.replaceState(null, '', hashOf({ model, view, layer: selectedLayer, end: selectedEnd }))
  }, [model, view, selectedLayer, selectedEnd])

  useEffect(() => {
    document.title = `${name} · hf-lens`
  }, [name])

  // Hand-edited hash (address bar, pasted link on an open tab): adopt it, then normalize invalid values in place.
  useEffect(() => {
    const onHashChange = () => {
      const next = parseHash(models)
      setModel(next.model)
      setView(next.view)
      setSelectedLayer(next.layer)
      setSelectedEnd(next.end)
      history.replaceState(null, '', hashOf(next))
    }
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [models])

  // Number keys switch tabs, except while typing in a field.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.repeat || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return
      const v = MAIN_VIEWS.find((m) => m.key === e.key)
      if (!v) return
      if ((e.target as HTMLElement | null)?.closest('input, textarea, [contenteditable]')) return
      e.preventDefault()
      setView(v.id)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // A different model has a different layer count: keep the layer if it still exists, and stop any token flow.
  const [prevModel, setPrevModel] = useState(model)
  if (model !== prevModel) {
    setPrevModel(model)
    setSelectedLayer((l) => Math.min(l, L - 1))
    setFlow((f) => ({ ...f, playing: false, step: null }))
    setTensorType(DEFAULT_TENSOR_TYPE)
  }

  // Selection follows the token everywhere it goes: the embedding (step -1), each layer, lm_head (step L).
  const [prevStep, setPrevStep] = useState(flow.step)
  if (flow.step !== prevStep) {
    setPrevStep(flow.step)
    if (flow.step !== null) select(flow.step < 0 ? 'embed' : flow.step >= L ? 'head' : flow.step)
  }

  const facts = headerFacts(config, breakdown)

  return (
    <div className="app">
      <header className="topbar">
        <h1 className="brand">{entry.repo}</h1>
        <dl className="topbar-facts">
          {facts.map((f) => (
            <div key={f.label} title={f.title}><dt>{f.label}</dt><dd>{f.value}</dd></div>
          ))}
        </dl>
        <p className="topbar-status" aria-live="polite">
          {flow.step === null
            ? <span className="idle">Idle</span>
            : <span className="badge"><span className="badge-dot" aria-hidden="true" />{flowStatus(flow.step, L)}</span>}
        </p>
      </header>

      <aside className="sidebar">
        <section className="form" aria-label="Model and token flow">
          <div className="field">
            <label className="field-label" htmlFor="model">Model</label>
            <div className="field-input">
              <select
                id="model"
                value={model}
                onChange={(e) => {
                  switchModel(e.target.value)
                  e.currentTarget.blur()
                }}
                onKeyDown={(e) => {
                  if (e.code === 'Space' || e.key === 'Enter') e.currentTarget.blur()
                }}
              >
                {models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
              </select>
            </div>
          </div>
          <TensorTypePicker rows={types} value={tensorRow.dtype} onChange={setTensorType} />
          <div className="field">
            <label className="field-label speed-label" htmlFor="speed">
              Speed <output htmlFor="speed">{(stepMs / 1000).toFixed(1)} s / step</output>
            </label>
            <input
              id="speed"
              className="speed"
              type="range"
              min={STEP_MS_MIN}
              max={STEP_MS_MAX}
              step={100}
              // Right = faster: the slider runs over the step duration reversed.
              value={STEP_MS_MIN + STEP_MS_MAX - stepMs}
              aria-valuetext={`${(stepMs / 1000).toFixed(1)} seconds per step`}
              onChange={(e) => setStepMs(STEP_MS_MIN + STEP_MS_MAX - Number(e.target.value))}
            />
          </div>
          <div className="form-actions">
            <TokenFlowControls flow={flow} setFlow={setFlow} config={config} />
            <AddModel models={models} onAdd={addModel} />
          </div>
        </section>
        <LayerDetail config={config} selected={selected} flowStep={flow.step} routes={routes} />
      </aside>

      <main className="main">
        <div className="main-bar">
          <MainTabs view={view} onChange={setView} />
          <LayerRod config={config} selected={selected} onSelect={select} flowStep={flow.step} />
        </div>
        <div className="view" role="tabpanel" id={`panel-${view}`} aria-labelledby={`tab-${view}`}>
          {view === 'model' && <ModelSummary model={entry} breakdown={breakdown} />}
          {view === 'network' && (
            <NetworkView
              config={config} modelName={name} routes={routes} selected={selected} flowStep={flow.step} stepMs={stepMs} onSelect={select}
              highlight={tensorRow.storage}
            />
          )}
          {view === 'block' && (
            <BlockDetail
              config={config}
              checkpointBytes={breakdown.bytes}
              layer={selectedLayer}
              route={flow.step !== null && flow.step >= selectedLayer ? routes[selectedLayer] : null}
            />
          )}
          {view === 'attention' && <AttentionPatterns config={config} layer={selectedLayer} />}
          {view === 'config' && <ConfigView model={entry} />}
        </div>
      </main>
    </div>
  )
}
