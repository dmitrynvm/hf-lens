import type { GptOssConfig, ModelEntry } from './config'

/**
 * Models added from a Hugging Face link. Only gpt_oss configs with MXFP4 experts are accepted: every size, KV and
 * routing number in the app is computed from that architecture and storage format.
 */

const HUB = 'https://huggingface.co'
const HUB_HOST = /^(?:https?:\/\/)?(?:www\.)?(?:huggingface\.co|hf\.co)(?:\/|$)/i
const NAME = /^[\w.-]+$/
const NOT_MODELS = new Set(['datasets', 'spaces', 'collections', 'docs', 'blog', 'papers', 'organizations'])

/** Repo id (`owner/name`) from a model page link, a file link inside it, or a bare id; null when it is none of these. */
export function repoFromLink(input: string): string | null {
  let path = input.trim()
  if (HUB_HOST.test(path)) path = path.replace(HUB_HOST, '')
  else if (/^[a-z][a-z+.-]*:\/\//i.test(path)) return null
  const [owner, name] = path.split(/[?#]/)[0].split('/')
  if (!owner || !name || !NAME.test(owner) || !NAME.test(name) || NOT_MODELS.has(owner)) return null
  return `${owner}/${name}`
}

const NUMBERS = [
  'hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'head_dim',
  'num_local_experts', 'num_experts_per_tok', 'vocab_size', 'sliding_window', 'max_position_embeddings', 'rope_theta',
  'swiglu_limit', 'rms_norm_eps',
] as const
const ROPE_NUMBERS = ['factor', 'original_max_position_embeddings', 'beta_fast', 'beta_slow'] as const

const isObject = (x: unknown): x is Record<string, unknown> => x !== null && typeof x === 'object' && !Array.isArray(x)

/** The config if this app can draw it; otherwise throws an Error that says what is missing or unsupported. */
export function parseGptOssConfig(json: unknown): GptOssConfig {
  if (!isObject(json)) throw new Error('config.json is not a JSON object')
  if (json.model_type !== 'gpt_oss') {
    throw new Error(`model_type is ${JSON.stringify(json.model_type ?? null)}; only gpt_oss models are supported`)
  }
  const q = json.quantization_config
  if (!isObject(q) || q.quant_method !== 'mxfp4') {
    throw new Error('experts are not MXFP4 (quantization_config.quant_method); only MXFP4 checkpoints are supported')
  }
  const rope = json.rope_scaling
  const missing = [
    ...NUMBERS.filter((k) => typeof json[k] !== 'number'),
    ...(isObject(rope) ? ROPE_NUMBERS.filter((k) => typeof rope[k] !== 'number').map((k) => `rope_scaling.${k}`) : ['rope_scaling']),
    ...(['attention_bias', 'tie_word_embeddings'] as const).filter((k) => typeof json[k] !== 'boolean'),
  ]
  if (missing.length) throw new Error(`config.json lacks ${missing.join(', ')}`)
  const types = json.layer_types
  if (!Array.isArray(types) || types.length !== json.num_hidden_layers
    || !types.every((t) => t === 'sliding_attention' || t === 'full_attention')) {
    throw new Error('layer_types must list sliding_attention or full_attention for each of num_hidden_layers')
  }
  return json as unknown as GptOssConfig
}

/** Entry for a repo from its config.json text; throws like `parseGptOssConfig`. */
export function customModel(repo: string, text: string): ModelEntry {
  let json: unknown
  try {
    json = JSON.parse(text)
  } catch {
    throw new Error('config.json is not valid JSON')
  }
  const config = parseGptOssConfig(json)
  return { id: repo, repo, name: repo, config, json: json as Record<string, unknown>, text }
}

/** Downloads `config.json` from the repo's main branch. Rejects with a message fit for the user. */
export async function fetchModel(repo: string): Promise<ModelEntry> {
  let res: Response
  try {
    res = await fetch(`${HUB}/${repo}/resolve/main/config.json`)
  } catch {
    throw new Error('Could not reach huggingface.co')
  }
  // The hub answers 401 for missing repos as well as gated or private ones, so it cannot tell them apart.
  if (res.status === 401 || res.status === 403) throw new Error(`${repo} does not exist, or is gated or private`)
  if (res.status === 404) throw new Error(`${repo} has no config.json`)
  if (!res.ok) throw new Error(`huggingface.co answered ${res.status}`)
  return customModel(repo, await res.text())
}

const STORAGE_KEY = 'gpt-oss.models'

/** Models added earlier in this browser; entries that no longer parse are dropped. */
export function loadCustomModels(storage: Pick<Storage, 'getItem'> = localStorage): ModelEntry[] {
  let saved: unknown
  try {
    saved = JSON.parse(storage.getItem(STORAGE_KEY) ?? '[]')
  } catch {
    return []
  }
  if (!Array.isArray(saved)) return []
  return saved.flatMap((s: unknown) => {
    if (!isObject(s) || typeof s.repo !== 'string' || typeof s.text !== 'string') return []
    try {
      return [customModel(s.repo, s.text)]
    } catch {
      return []
    }
  })
}

export function saveCustomModels(models: readonly ModelEntry[], storage: Pick<Storage, 'setItem'> = localStorage) {
  storage.setItem(STORAGE_KEY, JSON.stringify(models.map(({ repo, text }) => ({ repo, text }))))
}
