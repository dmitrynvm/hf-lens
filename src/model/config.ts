import config120b from './gpt-oss-120b.config.json'
import text120b from './gpt-oss-120b.config.json?raw'
import config20b from './gpt-oss-20b.config.json'
import text20b from './gpt-oss-20b.config.json?raw'

export type LayerType = 'sliding_attention' | 'full_attention'

/** The keys of config.json this app computes from; the files carry more (see `ModelEntry.json`). */
export interface GptOssConfig {
  hidden_size: number
  intermediate_size: number
  num_hidden_layers: number
  num_attention_heads: number
  num_key_value_heads: number
  head_dim: number
  num_local_experts: number
  num_experts_per_tok: number
  vocab_size: number
  sliding_window: number
  max_position_embeddings: number
  rope_theta: number
  rope_scaling: {
    rope_type: string
    factor: number
    original_max_position_embeddings: number
    beta_fast: number
    beta_slow: number
  }
  swiglu_limit: number
  rms_norm_eps: number
  attention_bias: boolean
  tie_word_embeddings: boolean
  layer_types: LayerType[]
  quantization_config: {
    quant_method: string
    modules_to_not_convert: string[]
  }
}

export const GPT_OSS_120B = config120b as GptOssConfig
export const GPT_OSS_20B = config20b as GptOssConfig

/** A checkpoint the app can show: its config.json parsed, typed and verbatim. */
export interface ModelEntry {
  /** Hash and storage key: '120b' / '20b' for the bundled checkpoints, the Hugging Face repo id for added ones. */
  id: string
  /** Hugging Face repo id, e.g. `openai/gpt-oss-120b`. */
  repo: string
  /** Short name for the model picker and diffs. */
  name: string
  config: GptOssConfig
  json: Record<string, unknown>
  /** config.json as published, for the Config tab. */
  text: string
}

/** Verbatim https://huggingface.co/openai/gpt-oss-{120b,20b}/raw/main/config.json, parsed and as text. */
export const BUILTIN_MODELS: readonly ModelEntry[] = [
  { id: '120b', repo: 'openai/gpt-oss-120b', name: 'gpt-oss-120b', config: GPT_OSS_120B, json: config120b, text: text120b },
  { id: '20b', repo: 'openai/gpt-oss-20b', name: 'gpt-oss-20b', config: GPT_OSS_20B, json: config20b, text: text20b },
]
export const DEFAULT_MODEL = '120b'

/** The model with this id, if the list has it. */
export const findModel = (models: readonly ModelEntry[], id: string | null) => models.find((m) => m.id === id)
