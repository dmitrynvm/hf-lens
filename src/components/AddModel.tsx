import { useState, type SubmitEvent } from 'react'
import type { ModelEntry } from '../model/config'
import { fetchModel, repoFromLink } from '../model/hub'
import { Icon } from './Icon'

interface Props {
  models: readonly ModelEntry[]
  /** Called with the new model, or with the listed one when the link names a model already there. */
  onAdd: (model: ModelEntry) => void
}

/** "+" button in the media-control row; opens a form, spanning the row, that loads a model from its Hugging Face link. */
export function AddModel({ models, onAdd }: Props) {
  const [open, setOpen] = useState(false)
  const [link, setLink] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const close = () => {
    setOpen(false)
    setLink('')
    setError(null)
  }

  const submit = async (e: SubmitEvent<HTMLFormElement>) => {
    e.preventDefault()
    const repo = repoFromLink(link)
    if (!repo) {
      setError('Paste a model link like https://huggingface.co/openai/gpt-oss-20b')
      return
    }
    const listed = models.find((m) => m.repo.toLowerCase() === repo.toLowerCase())
    if (listed) {
      onAdd(listed)
      close()
      return
    }
    setBusy(true)
    setError(null)
    try {
      onAdd(await fetchModel(repo))
      close()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <button
        type="button"
        className="button-secondary button-icon-only"
        aria-label="Add model from Hugging Face"
        title="Add model from Hugging Face"
        aria-expanded={open}
        aria-controls="add-model"
        onClick={() => (open ? close() : setOpen(true))}
      >
        <Icon name="add" />
      </button>
      {open && (
        <form id="add-model" className="add-model" onSubmit={submit} onKeyDown={(e) => e.key === 'Escape' && close()}>
          <label className="field-label" htmlFor="add-model-link">Hugging Face link</label>
          <div className="field-input">
            <input
              id="add-model-link"
              type="text"
              inputMode="url"
              spellCheck={false}
              autoComplete="off"
              autoFocus
              placeholder="huggingface.co/openai/gpt-oss-20b"
              value={link}
              onChange={(e) => setLink(e.target.value)}
              disabled={busy}
              aria-invalid={error !== null}
              aria-describedby={error ? 'add-model-error' : undefined}
            />
          </div>
          <div className="add-model-actions">
            <button type="submit" className="button-primary" disabled={busy || !link.trim()}>{busy ? 'Loading…' : 'Add'}</button>
            <button type="button" className="button-secondary" onClick={close}>Cancel</button>
          </div>
          {error && <p id="add-model-error" className="field-error" role="alert">{error}</p>}
        </form>
      )}
    </>
  )
}
