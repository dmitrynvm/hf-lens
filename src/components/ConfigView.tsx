import type { ReactNode } from 'react'
import { BUILTIN_MODELS, type ModelEntry } from '../model/config'
import { CONFIG_NOTES } from '../model/configNotes'

const KEY_LINE = /^(\s*)"([^"]+)": (.*)$/
const TOKEN = /("(?:[^"\\]|\\.)*")(\s*:)?|(-?\d[\d.eE+-]*|true|false|null)/g

/** Value at a dotted path, or undefined. */
function at(obj: unknown, path: string): unknown {
  return path.split('.').reduce<unknown>((o, k) => (o !== null && typeof o === 'object' ? (o as Record<string, unknown>)[k] : undefined), obj)
}

/** Colour one line of JSON text: keys, strings, literals; everything else (indent, punctuation) stays as written. */
function highlight(text: string): ReactNode[] {
  const out: ReactNode[] = []
  let last = 0
  for (const m of text.matchAll(TOKEN)) {
    out.push(text.slice(last, m.index))
    if (m[1] && m[2]) out.push(<span key={m.index} className="cfg-key">{m[1]}</span>, m[2])
    else if (m[1]) out.push(<span key={m.index} className="cfg-str">{m[1]}</span>)
    else out.push(<span key={m.index} className="cfg-lit">{m[3]}</span>)
    last = m.index + m[0].length
  }
  out.push(text.slice(last))
  return out
}

/** One sliding/full cell per layer instead of a line per layer. */
function LayerStrip({ types }: { types: string[] }) {
  return (
    <span className="cfg-strip">
      {types.map((t, i) => (
        <span key={i} className={`cfg-cell ${t === 'sliding_attention' || t === 'linear_attention' ? 'sliding' : 'full'}`}>
          <title>{`L${i} · ${t}`}</title>
        </span>
      ))}
    </span>
  )
}

/**
 * config.json exactly as published. Hover a line for what its key controls, with numbers derived from this model;
 * values that differ from a bundled model (the other one, or for an added model the one with its depth) carry a bar
 * and their hover gives that value. Training-only keys are muted.
 */
export function ConfigView({ model }: { model: ModelEntry }) {
  const { config: c, json } = model
  const others = BUILTIN_MODELS.filter((m) => m.id !== model.id)
  const otherModel = others.find((m) => m.config.num_hidden_layers === c.num_hidden_layers) ?? others[0]
  const other = otherModel.json
  const text = model.text.trimEnd().split('\n')

  const lines: ReactNode[] = []
  const objects: string[] = [] // keys of the objects enclosing the current line
  let array: string | null = null // path of the array whose items the current line belongs to
  for (let i = 0; i < text.length; i++) {
    const m = KEY_LINE.exec(text[i])
    let path: string | null = array
    let body: ReactNode = highlight(text[i])
    if (m) {
      objects.length = m[1].length / 2 - 1
      path = [...objects, m[2]].join('.')
      const rest = m[3]
      if (rest === '{') objects.push(m[2])
      if (rest === '[') array = path
      if (path === 'layer_types' && rest === '[') {
        // Collapse the per-layer lines into one strip, keeping the closing bracket's comma.
        let j = i + 1
        while (!text[j].trim().startsWith(']')) j++
        body = <>{highlight(`${m[1]}"${m[2]}": [`)}<LayerStrip types={c.layer_types} />{text[j].trim()}</>
        i = j
        array = null
      } else if ((path.endsWith('.kda_layers') || path.endsWith('.full_attn_layers')) && rest === '[') {
        let j = i + 1
        const items: string[] = []
        while (!text[j].trim().startsWith(']')) {
          items.push(text[j].trim().replace(/,$/, ''))
          j++
        }
        body = highlight(`${m[1]}"${m[2]}": [${items.join(', ')}]${text[j].trim().slice(1)}`)
        i = j
        array = null
      }
    } else if (array !== null && text[i].trim().startsWith(']')) {
      array = null
    }
    if (path === 'quantization_config.modules_to_not_convert' && !m && text[i].trim().startsWith('"')) {
      body = <>{body} <span className="dtype-tag bf16">bf16</span></>
    }
    if (path === 'quantization_config.quant_method') body = <>{body} <span className="dtype-tag mxfp4">expert weights</span></>
    const note = path ? CONFIG_NOTES[path] : undefined
    const mine = path ? at(json, path) : undefined
    const theirs = path ? at(other, path) : undefined
    const leaf = m !== null && (mine === null || typeof mine !== 'object' || Array.isArray(mine))
    const diff = leaf && JSON.stringify(mine) !== JSON.stringify(theirs)
      ? `${otherModel.name}: ${Array.isArray(theirs) ? `${theirs.length} entries` : JSON.stringify(theirs)}`
      : ''
    const title = [note?.note(c), diff].filter(Boolean).join('\n')
    lines.push(
      <div key={i} className={`cfg-line${note?.training ? ' training' : ''}${diff ? ' diff' : ''}${title ? ' noted' : ''}`} title={title || undefined}>
        {body}
      </div>,
    )
  }

  return (
    <section className="panel config" aria-label="config.json">
      <div className="table-head">
        <h2 className="section-title">
          config.json{' '}
          <a className="section-count" href={`https://huggingface.co/${model.repo}/blob/main/config.json`} target="_blank" rel="noreferrer">
            {model.name}
          </a>
        </h2>
      </div>
      <div className="cfg">{lines}</div>
    </section>
  )
}
