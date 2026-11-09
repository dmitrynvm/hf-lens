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

- **The residual stream is the main road**: a 4 px ink line across the whole width, the Block tab's vertical residual turned sideways. Every sublayer is a branch that leaves it at a tap, runs along a line above it and rejoins at a ⊕ (same circle as the Block diagram). Wires are muted 1.5 px; arrowheads only where a branch leaves or rejoins, plus one at the end of the stream.
- **Slabs are activations [T, d]**, never operations. Height = d, log scale (`10·(log₂ d − 5)`); thickness = parameters of the op that produced it, log scale, so experts are fat and norms paper-thin; depth = T, constant. Oblique boxes: front face, lighter top, darker side, dashed hidden edges.
- **Parallel tensors stack in depth** (q/k/v, the top-k experts); sequential ones follow along x.
- **Captions**: operations above the gaps in Barlow 600; dimensions below in mono, muted. Two fixed rows per level so neighbours never collide. Routed expert ids replace the experts' dimension in `--line`.
- **Show what differs**: L0 carries every activation; every later block is drawn too, keeping only the attention slab (amber/ink) and the expert stack.
- **Grouping** by a hairline bracket under each block with its label and a plain-language subtitle (`sliding · last 128 tokens`); each branch carries a sublayer tag naming its type above it (`sliding attention` in amber, `full attention` in ink, `MoE · top-4 of 128 experts` muted), and ops use plain names (`normalize`, `pick experts`, `weighted sum`) with code names in hover titles. Selection is `--select` + `--line` outline, token position `--now-wash` (no chip).
- **Weight tape** under the diagram: one gap-free strip, each group owning the stretch under it (boundaries halfway between groups), in its board-tile colour: pale for bf16 bytes (attention, router, norms), solid for MXFP4 experts, slate for the all-bf16 embedding and lm_head. No separators; each block's pale-then-solid step marks its start. A key row under the size labels names the shades; each label gives its group's size and its hover the sum behind it.
- **Explain on select**: a paragraph under the diagram (page UI, not in exports) describes the selected layer and, once the token has passed it, its routed experts.
- **Key inside the SVG** (height, thickness, colour, depth) so exports carry it.

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
