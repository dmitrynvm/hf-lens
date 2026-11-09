import { formatBytes, formatCount } from '../model/params'
import type { TensorTypeRow } from '../model/safetensors'

/**
 * The checkpoint's stored tensor types (the hub's Safetensors box) with parameter counts and sizes on disk. Picking one marks the
 * tensors stored in it on the Network diagram; the hover says which weights those are and in which layers.
 */
export function TensorTypePicker({ rows, value, onChange }: { rows: TensorTypeRow[]; value: string; onChange: (dtype: string) => void }) {
  const row = rows.find((r) => r.dtype === value)
  return (
    <div className="field">
      <label className="field-label" htmlFor="tensor-type">DType</label>
      <div className="field-input">
        <select
          id="tensor-type"
          value={value}
          title={row && (row.where.length ? row.where.join('\n') : 'not mapped to config.json tensors')}
          onChange={(e) => {
            onChange(e.target.value)
            e.currentTarget.blur()
          }}
          onKeyDown={(e) => {
            if (e.code === 'Space' || e.key === 'Enter') e.currentTarget.blur()
          }}
        >
          {rows.map((r) => (
            <option key={r.dtype} value={r.dtype}>{r.dtype} · {formatCount(r.params)}{r.bytes > 0 && ` (${formatBytes(r.bytes)})`}</option>
          ))}
        </select>
      </div>
    </div>
  )
}
