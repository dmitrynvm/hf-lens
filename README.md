# hf-lens

A lens on Hugging Face model checkpoints: paste a model link and see its architecture, memory, attention and expert routing, all computed from the repo's `config.json`. It runs entirely in the browser; nothing is downloaded except `config.json`.

Today it reads **gpt_oss** models with **MXFP4** experts. [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) and [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) are built in. Any other repo with the same architecture can be added from its link.

## Views

The main column has five tabs, switched with keys `1`–`5`.

| Tab | Shows |
|---|---|
| **Network** `1` | The whole model left to right. The residual stream is a thick line; every sublayer branches off it with `RMSNorm` and rejoins at ⊕. Every tensor slab has its operation name centered above it and its dimensions and dtype (`int64`, `bf16`, `mxfp4`) below it. `L0` unpacks every activation in the attention and MoE branches; later blocks keep `RMSNorm`, the attention slab (`SWA` or `FA`), and the top-k expert stack with routed expert IDs. Under each block sit its attention mask (sliding band or causal triangle) and an expert grid with the simulated top-k experts lit. Below that, four candidate weight tapes are stacked for comparison: **A** block weights split `bf16`/`mxfp4` with bytes read per token, **B** active vs dormant bytes per tensor, **C** per-tensor byte magnitude (log height) by dtype, and **D** per-block KV cache at full context. Click a group to select it. |
| **Block** `2` | One decoder block: attention (GQA, YaRN RoPE, sinks) and MoE (router, top-k of E experts, clamped SwiGLU) with `[T, …]` shapes on every arrow. Below it are the block's memory table (params, active params, bytes, bytes read per token) and every tensor with its shape and dtype. |
| **Attention** `3` | Sliding-window and full causal masks with the sink, query heads grouped onto KV heads, and the KV cache per sequence length compared with no sliding window and with plain multi-head attention. |
| **Routing** `4` | A layer × expert grid of the top-k experts one token is routed to, shaded by softmax weight. Routing is simulated with a fixed seed, so every load and link shows the same experts. |
| **Config** `5` | The `config.json` as published, with notes on `quantization_config` and the `layer_types` pattern. |

Around the tabs:

- **Top bar**: params, active params, layers, experts, context and checkpoint size. Hover a number to see the sum it comes from.
- **Sidebar**: model picker (switching models starts over at L0); **DType** picker with each stored type's parameter count and size on disk as the Hugging Face model page's Safetensors box lists it (e.g. `BF16 · 2.17B (4.04 GiB)`, and `U8` for MXFP4 blocks and scales; added models take these numbers from the hub API and fall back to computing them from `config.json`); speed slider; token-flow controls; then the selected layer in words: what its attention sees, which experts the token used, its weight bytes and the bytes read per token. The DType picker starts at `BF16` (again after every model switch); on the Network diagram the dtype caption under every tensor stored in the picked type turns red (`bf16` for BF16, `mxfp4` for U8), and hovering the picker lists those weights with their layers (e.g. `routed experts · L0–L35`). A type `config.json` cannot place, like Kimi-K3's `F32`, marks nothing.
- **Token flow**: step back, play/pause (`Space`), step forward and reset. Playback moves one layer per step (1.2 s by default, 0.3–3 s on the speed slider) and loops from lm_head back to the embedding until you pause; stepping forward by hand stops at lm_head. Under each block on the Network tab, the attention mask and expert grid start empty (on load and after switching models) and stay empty until the token arrives; during its step the mask fills top-down over the first half, then the routed experts light up over the second half, heaviest first.
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

- `src/model/`: everything computed from a config: `quant.ts` (which weights `quantization_config` stores as MXFP4), `params.ts` (counts and bytes), `tensors.ts`, `kvcache.ts`, `routing.ts` (seeded router), `network.ts` (diagram geometry), `hub.ts` (link parsing, fetching and validation), `facts.ts` (top-bar numbers).
- `src/components/`: one component per view plus the sidebar pieces.
- `src/flow.ts`: the token-flow state machine.

The visual system is described in `DESIGN.md`.
