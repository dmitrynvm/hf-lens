import { MAIN_VIEWS, type MainView } from '../views'

/** Tab bar for the main column; the parent renders the panel with id `panel-<view>`. */
export function MainTabs({ view, onChange }: { view: MainView; onChange: (v: MainView) => void }) {
  return (
    <div className="tabs" role="tablist" aria-label="View">
      {MAIN_VIEWS.map((v, i) => (
        <button
          key={v.id}
          type="button"
          role="tab"
          id={`tab-${v.id}`}
          className="tab"
          aria-selected={view === v.id}
          aria-controls={`panel-${v.id}`}
          aria-keyshortcuts={v.key}
          tabIndex={view === v.id ? 0 : -1}
          title={`${v.label} (${v.key})`}
          onClick={() => onChange(v.id)}
          onKeyDown={(e) => {
            if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
            e.preventDefault()
            const next = MAIN_VIEWS[(i + (e.key === 'ArrowRight' ? 1 : MAIN_VIEWS.length - 1)) % MAIN_VIEWS.length]
            onChange(next.id)
            document.getElementById(`tab-${next.id}`)?.focus()
          }}
        >
          {v.label}
          <kbd className="tab-key">{v.key}</kbd>
        </button>
      ))}
    </div>
  )
}
