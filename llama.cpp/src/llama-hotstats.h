// llama-hotstats.h - hot expert statistics module
//
// Purpose: collect per-(layer, expert) selection frequency from the MoE
// routing tensor (selected_experts, passed as src1 to every MUL_MAT_ID
// node in build_moe_ffn). This data is the ground truth for deciding
// which experts are "hot" under a real workload, and for simulating
// top-N expert offload coverage.
//
// Enabled with env: LLAMA_HOT_STAT=1
// Output path:      LLAMA_HOT_STAT_FILE=/path/to/hot_experts.json (default: hot_experts.json)
// Interval:         LLAMA_HOT_STAT_INTERVAL=<seconds> (default: 60)
//
// The module writes an incremental JSON snapshot every INTERVAL seconds
// and a final one on deinit, so a long-running server can be sampled
// mid-flight.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llama_hotstats {

    // initialize the collector; returns false if disabled
    bool init();

    // record a batch of selected expert ids for one layer
    //   layer:  0-based block index (from tensor name blk.<layer>.ffn_*_exps.weight)
    //   ids:     pointer to n_expert_used * n_tokens int32 selected expert indices
    //   n_used:  number of experts per token (e.g. 6)
    //   n_tokens: number of tokens in the ubatch
    //   n_layer_total: total number of transformer layers (for layer-name normalization)
    void record(int layer, const int32_t * ids, int n_used, int n_tokens);

    // write current snapshot to the configured file (or given path)
    void dump(const std::string & path);

    // dump if the configured interval has elapsed since the last dump
    void maybe_dump();

    // graph-build-time snapshot registry: routing ids tensors whose device
    // data is read back after graph compute (see llama_context decode)
    void register_snap(int layer, const void * ids_tensor);
    void clear_snaps();
    const std::vector<std::pair<int, const void *>> & snaps();

    // final flush on exit
    void fini();

    bool enabled();

} // namespace llama_hotstats