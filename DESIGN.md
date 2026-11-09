# Design System: hf-lens

Borrowed from `../capalo-ai/DESIGN.md` ("The Nordic Rail Timetable"): same tokens, type, shapes and rules. This file records only how that world maps onto a model visualizer; capalo's file is the reference for everything not listed here.

## Shell

- **Space** plays/pauses token flow from anywhere, as capalo's Space starts a run; ignored in fields and on a keyboard-focused button or link (a mouse-clicked one does not count).
- **Band** (`--band`): the Hugging Face repo (`openai/<model>`) plus model facts (params, active, checkpoint, layers, experts, context). Right side: token-flow status, "Idle" or a badge with the yellow dot (`L3 of 36`, `embedding`, `lm_head`).
- **Sidebar** (280 px, hairline right border, compact 13–14 px type and 28 px tiles): model select, token-flow form (seed, speed, icon controls step back / play-pause / step forward / reset, then "+" to add a gpt_oss MXFP4 model from its Hugging Face link — config.json is fetched, validated and kept in localStorage; the hash names it by repo id — then Trace…), then the layer stack as a departures board.
- **Main**: tabs `Network 1 · Block 2 · Attention 3 · Routing 4 · Parameters 5` (number keys switch tabs outside inputs; arrows move within the tab list; the tab is part of the URL hash as `view=`).
- ≤ 860 px: one column, sidebar above main, board capped at 380 px with its own scroll, tab key hints hidden.

## Network diagram

One grammar, each visual channel with one meaning:

- **The residual stream is the main road**: an ink line across the whole width, the Block tab's vertical residual turned sideways. Every sublayer is a branch that leaves it at a tap, runs along a line above it and rejoins at a ⊕ (same circle as the Block diagram). Every line is 2 px: the stream in ink, wires and connectors muted; in the group the token is in, all of them turn `--now` at the same width. Arrowheads only where a branch leaves or rejoins, plus one at the end of the stream.
- **Slabs are activations [T, d]**, never operations. Height = d, log scale (`10·(log₂ d − 5)`); thickness = parameters of the op that produced it, log scale, so experts are fat and norms paper-thin; depth = T, constant. Oblique boxes: front face, lighter top, darker side, dashed hidden edges.
- **Parallel tensors stack in depth** (q/k/v, the top-k experts); sequential ones follow along x.
- **Captions**: each tensor slab is a single vertical column with its operation name centered directly above the slab (`tokens`, `embed`, `RMSNorm`, `q·k·v`, `SWA`/`FA`, `o_proj`, `router`, `gate_up`, `SwiGLU`, `down`, `sum`, `MoE`, `unembed`, `probs`), its compact dimensions and dtype (`int64`, `bf16`, `mxfp4`) stacked directly below, and (for experts) routed expert ids in `--line`.
- **Show what differs**: L0 carries every activation; every later block keeps `RMSNorm` in both branches plus the attention slab (`SWA` in amber or `FA` in ink) and the `MoE` expert stack.
- **Grouping** by a hairline bracket under each group with its label (`embed`, `L0`…`L35`, `unembed`). Selection is `--select` + `--line` outline, token position `--now-wash` (no chip).
- **Block squares** under each decoder block, inside its selectable area, top-aligned: the attention mask under the attention branch (`--sliding` band for sliding, `--full` triangle for causal, sink bar on the left; KDA is an amber causal triangle fading from the diagonal toward the oldest keys by the decay ∏α, with no sink bar; captioned keys seen / tokens shown, e.g. `128/512`, `512/512`) and an expert grid under the MoE branch, as wide as the mask, its frame hugging the cells (one square cell per expert, ⌈√E⌉ columns, so 128 experts make a 120 × 110 frame; cells past E that fill the last row are `--paper` with no outline; the simulated top-k lit in `--line`, at 40% until the token reaches the block; captioned routed / total, e.g. `4/128`; dense blocks show no grid and `1/1`). Captions are digits only; words live in the hover title.
- **Weight tape** under the diagram, untitled: one bar per group coloured like the layer minimap (`--sliding` for sliding/KDA, `--full` for full attention, `--bf16` grey for embed/head), its `total · read/tok` label inside the bar in `--paper`. Hovering a segment shows its derivation.
- **DType picked**: the sidebar's DType picker (`BF16` by default) turns the dtype caption of every tensor stored in that type `--fail` red (`bf16` for BF16, `mxfp4` for U8); nothing else changes. Weightless ops (SwiGLU, sum, softmax, token ids) are never marked.
- **Explain on select**: a paragraph in the sidebar describes the selected layer and, once the token has passed it, its routed experts.

## Colour meaning

| Token | Means | Where |
|---|---|---|
| `--line` (route blue) | the token's route and anything interactive | chosen experts (block grid, heatmap), primary button, selected tab, focus, selected row tint, KV-cache bar |
| `--now` / `--now-wash` | where the token is right now, nothing else | current board row, current heatmap row, band status dot |
| `--sliding` (amber) | sliding-window attention | board tile underline, legend swatch, sliding mask |
| `--full` (ink) | full causal attention | board tile underline, legend swatch, full mask (80 % opacity) |
| `--mxfp4` (green) | MXFP4-stored weights | dtype badge, MoE box stroke, stacked share bar |
| `--bf16` (slate) | bf16-stored weights | dtype badge, attention/norm strokes, query heads, stacked share bar, GQA/MHA comparison bars |

Amber and green have no status meaning here (there is no pass/fail); they are the two domain hues.

## Rules kept from capalo

- Ruled paper: an ink rule opens each block (tables, facts, masks, board, heatmap), hairlines divide rows. No cards, no shadows.
- Barlow Semi Condensed speaks (titles, labels, tabs, badges, board ids); Public Sans reports values; mono only for tensor names and shapes. Tabular figures everywhere.
- 2 px corners on controls, badges and diagram boxes; circles only for the residual junctions and the status dot.
- No motion: the current position steps with the flow, no easing.

## Exports

SVG/PNG exports inline computed styles and fill the background with the page paper. Standalone files cannot load the self-hosted fonts, so exported text falls back to the system sans/mono.
