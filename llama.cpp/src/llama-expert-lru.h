// llama-expert-lru.h - LRU expert cache ring for the MoE hot/cold offload
//
// Enabled with env LLAMA_EXPERT_LRU_SLOTS_PER_LAYER=<K> (K spare columns per
// layer allocated at model load, see llama_model_base::setup_moe_hot_exps).
// After each decode, llm_expert_lru::on_decode() is called with the per-layer
// selected expert ids. For every selected expert not currently in the hot set
// (pinned or spare), the manager copies its weight columns from the cold
// tensor into a spare hot column, LRU-evicting the least recently used spare
// occupant if necessary, and updates the remap/flag/cold_remap tables so the
// next decode routes that expert through the hot (CUDA) branch.
//
// The hot tensor layout per layer is [n_hot pinned][K spare][1 zero mask], so
// the mask column index is n_hot + K (matching setup_moe_hot_exps).

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

struct ggml_tensor;

namespace llm_expert_lru {

    // per-layer tensors the swap manager writes to / reads from
    struct LayerTensors {
        ggml_tensor * hot_up    = nullptr; // [n_embd, n_ff, n_hot+1+K] on CUDA
        ggml_tensor * hot_gate  = nullptr;
        ggml_tensor * hot_down  = nullptr;
        ggml_tensor * cold_up   = nullptr; // [n_embd, n_ff, n_expert] on ROCm
        ggml_tensor * cold_gate = nullptr;
        ggml_tensor * cold_down = nullptr;
        ggml_tensor * remap     = nullptr; // I32 [n_expert]
        ggml_tensor * cold_remap= nullptr; // I32 [n_expert]
        ggml_tensor * flag      = nullptr; // F32 [n_expert]
        int32_t  n_hot    = 0;             // pinned hot expert count (mask at n_hot+K)
        int32_t  n_spare  = 0;             // K
        int32_t  n_mask   = 0;             // n_hot + K
    };

    // per-layer spare-slot bookkeeping
    struct LayerState {
        std::unordered_map<int32_t, int32_t> expert_to_slot; // global expert id -> hot col index
        std::vector<int32_t> slot_expert;                    // hot col index -> global expert id (-1 = free)
        std::vector<uint64_t> slot_hits;                     // cumulative hits per slot (LFU eviction)
        std::vector<uint64_t> slot_enter_tick;               // decode tick at swap-in (grace period)
        std::vector<uint64_t> slot_last_tick;                // decode tick of last hit (idle eviction)
        // occurrence evidence for cold experts that have not been swapped in
        // yet; only experts seen >= min_occurrence times attempt a swap-in,
        // so single-occurrence routing noise never pays for a 19MB copy
        std::unordered_map<int32_t, int32_t> candidate_count;
        // host-side mirror of the device remap table (I32 [n_expert]): the
        // swap manager is its only writer and keeps this in sync on evict/bind
        std::vector<int32_t> remap_host;
        uint64_t clock = 0;
    };

    // reset the occurrence evidence (called at each sliding-window boundary,
    // e.g. every 2048 decoded tokens) so stale cold experts can re-qualify
    void reset_window(std::vector<LayerState> & states);

    // called once per decode with the per-layer selected expert ids
    // swap_budget (optional): remaining number of expert swaps allowed for
    // this batch across all layers; decremented on every actual swap-in.
    // Pass nullptr to disable the budget limit.
    // swap_enable=false: count selected experts into candidate_count only,
    // never swap (used for prefill: harvest routing statistics cheaply, the
    // actual swap-in happens later during decode).
    void on_decode(
        std::vector<LayerState> & states,
        const LayerTensors & tensors,
        int layer,
        const int32_t * ids,
        int n_used,
        int * swap_budget = nullptr,
        bool swap_enable = true);

    // returns true when the LRU ring is enabled (env set)
    bool enabled();

    // per-phase (prefill vs decode) cumulative hit counters for the /metrics
    // endpoint; zero when the LRU ring is disabled
    void get_hit_counts(uint64_t & prefill_sel, uint64_t & prefill_hit,
                        uint64_t & decode_sel,  uint64_t & decode_hit);

} // namespace llm_expert_lru