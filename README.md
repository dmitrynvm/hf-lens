# hf-lens

A lens on Hugging Face model checkpoints: paste a model link and see its architecture, memory, attention and expert routing, all computed from the repo's `config.json`. It runs entirely in the browser; nothing is downloaded except `config.json`.

Today it reads **gpt_oss** models with **MXFP4** experts. [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) and [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) are built in. Any other repo with the same architecture can be added from its link.

## Views

The main column has five tabs, switched with keys `1`–`5`.

| Tab | Shows |
|---|---|
| **Network** `1` | The whole model left to right. The residual stream is a thick line; every sublayer branches off it and rejoins at ⊕. Activations are drawn as slabs: height follows the width d, thickness follows the parameter count of the op that made them (both log scale), and parallel tensors (q/k/v, the top-k experts) are stacked in depth. Under the diagram, a weight tape gives each group a segment coloured like its sidebar tile and labelled with its type (`in`, `sliding`, `full`, `out`) and size; the segment splits into a bf16 part and an MXFP4 part, and hovering it shows the byte breakdown. Click a group to select it. |
| **Block** `2` | One decoder block: attention (GQA, YaRN RoPE, sinks) and MoE (router, top-k of E experts, clamped SwiGLU) with `[T, …]` shapes on every arrow. Below it are the block's memory table (params, active params, bytes, bytes read per token) and every tensor with its shape and dtype. |
| **Attention** `3` | Sliding-window and full causal masks with the sink, query heads grouped onto KV heads, and the KV cache per sequence length compared with no sliding window and with plain multi-head attention. |
| **Routing** `4` | A layer × expert grid of the top-k experts one token is routed to, shaded by softmax weight. Routing is simulated with a fixed seed, so every load and link shows the same experts. |
| **Config** `5` | The `config.json` as published, with notes on `quantization_config` and the `layer_types` pattern. |

Around the tabs:

- **Top bar**: params, active params, layers, experts, context and checkpoint size. Hover a number to see the sum it comes from.
- **Sidebar**: model picker, token-flow controls and the layer rod, which runs from the embedding through every block to lm_head. Arrow keys and Home/End move along it. Under the rod is the selected layer in words: what its attention sees, which experts the token used, its weight bytes and the bytes read per token.
- **Token flow**: step back, play/pause (`Space`), step forward and reset. Playback moves one layer every 400 ms and loops from lm_head back to the embedding until you pause; stepping forward by hand stops at lm_head.
- **Links**: the model, tab and selected layer live in the URL hash, e.g. `#model=20b&view=routing&layer=7` or `layer=embed` / `layer=head`. Editing the hash on an open page applies it straight away; invalid values fall back to 120b, Network and layer 0.
- **Themes**: light paper by default, night blue when the OS prefers dark.

## Adding a model

Click **+** in the sidebar and paste a Hugging Face link (`https://huggingface.co/owner/name`, a file link inside the repo, `hf.co/…`, or a bare `owner/name`). hf-lens fetches `config.json` from the main branch and accepts it only if:

- `model_type` is `gpt_oss`;
- `quantization_config.quant_method` is `mxfp4`;
- every number the views compute from is present, and `layer_types` has one entry per layer.

Otherwise it tells you what is missing. Gated and private repos can't be fetched. Added models are kept in `localStorage`, and the hash names them by repo id.

## Development

```sh
npm install
npm run dev        # dev server on http://localhost:5173
npm test           # vitest: params, tensors, KV cache, routing, flow, layout, hub parsing
npm run lint       # oxlint
npm run build      # tsc -b && vite build → dist/
npm run e2e        # Playwright: every tab in light, dark and 375 px, plus link round-trips
npm run e2e:update # re-record screenshots after an intended visual change
```

The e2e suite builds the app and serves it with `vite preview` on port 4174. It needs Chromium (`npx playwright install chromium`). Baselines live in `e2e/__screenshots__/<project>/`.

Code layout:

- `src/model/`: everything computed from a config: `params.ts` (counts and bytes), `tensors.ts`, `kvcache.ts`, `routing.ts` (seeded router), `network.ts` (diagram geometry), `hub.ts` (link parsing, fetching and validation), `facts.ts` (top-bar numbers).
- `src/components/`: one component per view plus the sidebar pieces.
- `src/flow.ts`: the token-flow state machine.

The visual system is described in `DESIGN.md`.
