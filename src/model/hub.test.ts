import { describe, expect, it } from 'vitest'
import text20b from './gpt-oss-20b.config.json?raw'
import { customModel, loadCustomModels, repoFromLink, saveCustomModels } from './hub'

describe('repoFromLink', () => {
  it('takes the repo id from model links, file links and bare ids', () => {
    for (const s of [
      'https://huggingface.co/openai/gpt-oss-20b',
      '  huggingface.co/openai/gpt-oss-20b/  ',
      'https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json',
      'https://hf.co/openai/gpt-oss-20b?library=transformers',
      'openai/gpt-oss-20b',
    ]) expect(repoFromLink(s)).toBe('openai/gpt-oss-20b')
  })

  it('rejects other sites, non-model pages and incomplete ids', () => {
    for (const s of ['https://github.com/openai/gpt-oss', 'https://huggingface.co/datasets/a/b', 'https://huggingface.co/openai', 'gpt-oss-20b', ''])
      expect(repoFromLink(s)).toBeNull()
  })
})

describe('customModel', () => {
  const edit = (patch: Record<string, unknown>) => JSON.stringify({ ...JSON.parse(text20b), ...patch })

  it('accepts a gpt_oss MXFP4 config and keys it by repo', () => {
    const m = customModel('acme/oss-ft', text20b)
    expect(m).toMatchObject({ id: 'acme/oss-ft', repo: 'acme/oss-ft', name: 'acme/oss-ft', text: text20b })
    expect(m.config.num_hidden_layers).toBe(24)
  })

  it('names what makes a config unsupported', () => {
    expect(() => customModel('a/b', '{')).toThrow('not valid JSON')
    expect(() => customModel('a/b', edit({ model_type: 'llama' }))).toThrow('model_type is "llama"')
    expect(() => customModel('a/b', edit({ quantization_config: undefined }))).toThrow('not MXFP4')
    expect(() => customModel('a/b', edit({ hidden_size: undefined, rope_scaling: {} }))).toThrow(
      'lacks hidden_size, rope_scaling.factor, rope_scaling.original_max_position_embeddings, rope_scaling.beta_fast, rope_scaling.beta_slow',
    )
    expect(() => customModel('a/b', edit({ layer_types: ['full_attention'] }))).toThrow('layer_types')
  })
})

describe('custom model storage', () => {
  it('round-trips added models and drops entries that no longer parse', () => {
    const store = new Map<string, string>()
    const storage = { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => void store.set(k, v) }
    saveCustomModels([customModel('acme/oss-ft', text20b)], storage)
    store.set('gpt-oss.models', JSON.stringify([...JSON.parse(store.get('gpt-oss.models')!), { repo: 'x/y', text: '{}' }]))
    expect(loadCustomModels(storage).map((m) => m.repo)).toEqual(['acme/oss-ft'])
    store.set('gpt-oss.models', 'not json')
    expect(loadCustomModels(storage)).toEqual([])
  })
})
