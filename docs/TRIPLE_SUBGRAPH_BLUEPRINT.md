# Per-Layer Triple-Subgraph MoE Pipeline: Implementation Blueprint

Branch `feat/moe-triple-subgraph` carries the build-side foundation. This blueprint
is the exact path to the 35 t/s target. It is the ONLY architecture that reaches it
(Oracle analysis, independently verified against code below).

## Goal

Wall clock per token ~= CUDA 22ms + merge ~= 24ms ~= 42-45 t/s, by overlapping the
hot expert chain (CUDA0) with the cold expert chain (ROCm0) layer by layer instead
of serializing them inside one whole-model graph.

## Architecture (Oracle design)

```
A_i   = [merge_{i-1}, attn_i, norm_i]            // CUDA0
hot_i = [hot MoE chain for layer i]              // CUDA0 (same sched as A_i)
cold_i= [cold MoE chain for layer i]             // ROCm0 (sched_cold)

pipeline per token:
  for il in 0..n_layer:
    submit A_il            (writes cur/selected/weights host mirrors)
    host-side wait ev_norm (norm_il output ready on host)
    submit hot_il  (CUDA0)  SUBMIT PARALLEL WITH  submit cold_il (ROCm0)
    host-side wait ev_cold (cold aggregate moe_host ready)
    # A_{il+1} head merges hot_moe_host + moe_host into the residual
```

Two graphs per layer (hot merges into the CUDA A_i graph since both are CUDA0),
i.e. 43 layers x 2 = 86 cgraphs, one per layer per device.

## Mechanisms already verified

1. ggml_backend_sched supports a per-graph sequence:
   `ggml_backend_sched_reset` clears hash_set + is_alloc=false but keeps the
   galloc buffers (alloc_splits only re-reserves when backend_ids change), so a
   loop of `reset -> alloc_graph -> graph_compute_async` per layer is viable and
   reuses buffers across layers.
   Evidence: ggml/src/ggml-backend.cpp reset (line ~1882) + alloc_splits (line ~1544).
2. Host-side event gating: ggml_backend_event_new/record/synchronize exist
   (ggml/include/ggml-backend.h 124-127) and are already used by sched for
   cross-split ordering. We use them from llama side between graph submissions.
3. The whole-model three-graph (gf + gf_hot + gf_cold, ALL layers in each) is
   DEAD: main graph contains merge, merge needs hot+cold outputs, hot/cold need
   main attention outputs -> circular -> mandatory replay. Verified:
   - alloc_graph asserts !is_alloc (single graph per sched at a time)
   - compute_splits reuses the LAST graph's splits when a different graph is
     submitted without reset (would compute garbage)
   - two scheds sharing the CUDA0 stream crash flash-attn ("illegal memory
     access", sched_cold comment in llama-context.cpp ~line 622)
   Per-layer breaks the cycle because merge moves to A_{i+1}'s head.

## Build-side foundation ALREADY COMMITTED on this branch

- 13c29c4: gf_hot third graph slot in llm_graph_result/llm_graph_context
  (+ dynamic metadata pool sizing for the enabled graph count).
- e3461fa: build_moe_ffn routes the hot chain into gf_hot with op==NONE host
  boundary mirrors (hot_cur/hot_sel/hot_w, mirroring cold_boundary) so
  visit_parents never pulls the CUDA0 attention chain into the hot graph, and an
  output mirror hot_moe_out for the merge. Zero behavior change when gf_hot==nullptr.

## Remaining work (1-2 days, split into shippable steps)

### Step 1: per-layer cgraph storage in llm_graph_result
- Replace the single `gf/gf_hot/gf_cold` with per-layer arrays for TRIPLE mode:
  std::vector<ggml_cgraph*> gf_a, gf_hot, gf_cold sized n_layer (or a struct).
- reset(): allocate 43x2 (or 43x3) cgraphs from the same compute ctx; size the
  metadata pool for n_layer graphs (per-layer max_nodes = max_nodes/n_layer + slack).

### Step 2: layer loop emits per-layer graphs (build side)
- dflash.cpp graph_dsv4 layer loop: inside `for il`, switch the active
  gf/gf_hot/gf_cold pointers to layer il's cgraphs before building attention and
  build_moe_ffn. The residual (cur/inpL) must cross graph boundaries -> carry it
  via op==NONE host mirrors (cold_boundary pattern) between A_il and A_{il+1}
  (also for the merge of the previous layer's hot+cold partials).
- build_moe_ffn already returns the hot partial via hot_moe_out (committed); the
  main-graph merge of hot_moe_host + moe_host then naturally belongs to the head
  of A_{il+1}.

### Step 3: graph_compute per-layer pipeline (llama-context.cpp)
- Replace the 2723-2780 dual-graph branch with a per-layer loop for
  GGML_MOE_TRIPLE:
  ```
  for il in 0..n_layer:
      ggml_backend_sched_graph_compute_async(sched, gf_a[il]);     // CUDA0
      ggml_backend_event_synchronize(ev_norm[il]);                  // host wait
      ggml_backend_sched_graph_compute_async(sched, gf_hot[il]);   // CUDA0, hot
      ggml_backend_sched_graph_compute_async(sched_cold, gf_cold[il]); // ROCm0
      ggml_backend_event_synchronize(ev_cold[il]);                  // host wait
  ```
  Events: create per-layer events on each sched's backends at init; record on the
  backend after each graph's split for A_il (norm) and cold_il (moe_host). The
  hot graph needs no event (host waits on cold; hot is same-device with A_i).
- process_ubatch: alloc each per-layer graph on its sched before the loop
  (or let graph_compute_async lazy-alloc after reset).

### Step 4: wiring sched_cold for per-layer cold graphs
- sched_cold already exists (ROCm0-only). It can hold the per-layer cold graph
  sequence with reset+alloc per layer (mechanism verified above). Only the
  ROCm0 stream is used, so no CUDA0 conflict.

### Step 5: validate + measure
- Build in moe-build container; deploy hot600_spare_iq3 to moe-srv on port 5234.
- Correctness: compare sampled logits/tokens against single-graph baseline.
- Speed: expect > 26.7 and toward 42-45; if slower, profile where the pipeline
  stalls (event wait latency, per-layer sched reset cost) before deeper changes.

## Hard constraints (do not violate)

- Default path (no GGML_MOE_TRIPLE env) must stay byte-for-byte identical.
- No cross-vendor event waits (ggml_backend_cuda_event_wait ABORTs on non-CUDA).
- No second sched over CUDA0 (flash-attn same-stream crash).
- Keep ggml_backend_event objects host-side synchronized; never rely on device
  order across scheds.

## References (source of truth)

- ggml/src/ggml-backend.cpp: sched reset 1882, alloc_splits 1544-1595,
  compute_async 1950, events created only when n_copies>1 (1840).
- src/llama-graph.cpp build_moe_ffn (1942+): hot chain (2129+), cold boundary
  (2261+), hot_moe_out mirror (2408+), merge (2440+).
- src/llama-context.cpp: sched_cold creation (622-650), process_ubatch alloc
  (1434+), graph_compute dual-graph branch (2723-2780).
- src/models/dflash.cpp graph_dsv4 layer loop (563-669).

## Session completion note

This session: verified the whole-model triple-graph is dead, committed the
build-side foundation (2 commits), restored the hot600 baseline (healthy,
26.31 t/s). The per-layer pipeline above is the approved 1-2 day architecture
change; it was NOT implemented in this session (context budget). Continue from
Step 1 in a fresh session with this blueprint.