// llama-expert-lru.cpp - LRU expert cache ring implementation
//
// During decode (batched every LLAMA_EXPERT_LRU_APPLY_EVERY tokens), the
// caller feeds the per-layer selected expert ids in. For each selected expert
// not already pinned hot or in a spare slot, the manager:
//   1. picks a spare slot (free, or LRU-evicts the oldest occupant)
//   2. copies the three weight columns (up/gate/down) from the cold tensor
//      into the spare hot columns (host-staged, like model-load time)
//   3. updates remap/flag/cold_remap so the next compute routes that expert
//      through the hot branch and skips it in the cold branch
// Evicted experts revert to the cold path.
//
// Performance: the remap table (1 KiB) is read back once per layer and cached
// host-side (the swap manager is its only writer); the column copy buffer is
// allocated lazily, only when an actual swap pays for a 19MB copy.

#include "llama-expert-lru.h"
#include "llama-model.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace llm_expert_lru {

namespace {
    // cumulative hit-rate accounting (printed on shutdown / every N batches)
    uint64_t g_selected   = 0; // total selected-expert slots seen
    uint64_t g_spare_hit  = 0; // selected expert found in a spare slot
    uint64_t g_pinned_hit = 0; // selected expert found in the pinned hot set
    uint64_t g_swapped    = 0; // experts swapped into spare slots
    uint64_t g_cold       = 0; // selected expert computed on the cold path
    uint64_t g_print_n   = 0; // last printed selected-count bucket (hit summary)
    // phase-split hit accounting for the /metrics endpoint: prefill runs with
    // swap_enable=false (count-only), decode with swap_enable=true
    uint64_t g_psel = 0, g_phit = 0; // prefill: selected / hit (pinned or spare)
    uint64_t g_dsel = 0, g_dhit = 0; // decode:  selected / hit
}

void get_hit_counts(uint64_t & prefill_sel, uint64_t & prefill_hit,
                    uint64_t & decode_sel,  uint64_t & decode_hit) {
    prefill_sel = g_psel; prefill_hit = g_phit;
    decode_sel  = g_dsel; decode_hit  = g_dhit;
}

bool enabled() {
    const char * v = getenv("LLAMA_EXPERT_LRU_SLOTS_PER_LAYER");
    return v != nullptr && std::atoi(v) > 0;
}

    // copy one column of src tensor into the dst tensor at column `dst_col`
    static void copy_expert_column(ggml_tensor * dst, ggml_tensor * src,
            int32_t src_col, int32_t dst_col, const size_t col_size, std::vector<char> & buf) {
        if (dst == nullptr || src == nullptr) {
            return;
        }
        ggml_backend_tensor_get(src, buf.data(), (size_t) src_col * col_size, col_size);
        ggml_backend_tensor_set(dst, buf.data(), (size_t) dst_col * col_size, col_size);
    }

    void on_decode(
        std::vector<LayerState> & states,
        const LayerTensors & t,
        int layer,
        const int32_t * ids,
        int n_used,
        int * swap_budget,
        bool swap_enable) {
        if (!enabled() || ids == nullptr || n_used <= 0) {
            return;
        }
        if (t.n_spare <= 0 || t.remap == nullptr || t.cold_remap == nullptr || t.flag == nullptr) {
            return;
        }
        if (layer >= (int) states.size()) {
            states.resize(layer + 1);
        }
        LayerState & st = states[layer];
        if (st.slot_expert.empty()) {
            st.slot_expert.assign(t.n_spare, -1);
            st.slot_hits.assign(t.n_spare, 0);
            st.slot_enter_tick.assign(t.n_spare, 0);
            st.slot_last_tick.assign(t.n_spare, 0);
        }

        const int64_t n_expert = t.remap->ne[0];

        // env knobs: read once per call, not once per selected expert
        const int min_occ = std::max(1, getenv("LLAMA_EXPERT_LRU_MIN_OCCURRENCE") ? std::atoi(getenv("LLAMA_EXPERT_LRU_MIN_OCCURRENCE")) : 3);
        const int grace   = std::max(0, getenv("LLAMA_EXPERT_LRU_GRACE_TICKS") ? std::atoi(getenv("LLAMA_EXPERT_LRU_GRACE_TICKS")) : 32);
        // calm period: a slot whose occupant has not been activated for
        // CALM_IDLE_TICKS decode ticks is dead weight; only then evict. When
        // every occupant is active the ring stays frozen - zero 19MB copies.
        const uint64_t calm_idle = getenv("LLAMA_EXPERT_LRU_CALM_IDLE_TICKS") ? (uint64_t) std::max(0, std::atoi(getenv("LLAMA_EXPERT_LRU_CALM_IDLE_TICKS"))) : 128;

        // remap cached in LayerState: read back only on first use per layer;
        // evict/bind keep the host copy in sync with the device table
        if ((int64_t) st.remap_host.size() != n_expert) {
            st.remap_host.resize((size_t) n_expert);
            ggml_backend_tensor_get(t.remap, st.remap_host.data(), 0, sizeof(int32_t) * n_expert);
            // backfill slots that were pre-warmed into the hot tensor at model
            // load (remap points them at column n_hot+s); otherwise the ring
            // sees them as free and overwrites the pre-loaded weights
            for (int32_t e = 0; e < n_expert; ++e) {
                const int32_t col = st.remap_host[(size_t) e];
                if (col >= t.n_hot && col < t.n_mask) {
                    const int32_t s = col - t.n_hot;
                    if (s < t.n_spare) {
                        st.slot_expert[(size_t) s] = e;
                        st.slot_last_tick[(size_t) s] = st.clock;
                        st.expert_to_slot[e] = col;
                    }
                }
            }
            // pre-warmed slots are protected like fresh swaps during grace
            for (int32_t s = 0; s < t.n_spare; ++s) {
                if (st.slot_expert[(size_t) s] >= 0) {
                    st.slot_enter_tick[(size_t) s] = st.clock;
                }
            }
        }

        // column sizes are constant per layer; compute once, allocate the
        // copy buffer lazily only when an actual swap needs it
        const size_t col_up   = t.hot_up   ? ggml_row_size(t.hot_up->type,   t.hot_up->ne[0])   * t.hot_up->ne[1]   : 0;
        const size_t col_gate = t.hot_gate ? ggml_row_size(t.hot_gate->type, t.hot_gate->ne[0]) * t.hot_gate->ne[1] : 0;
        const size_t col_down = t.hot_down ? ggml_row_size(t.hot_down->type, t.hot_down->ne[0]) * t.hot_down->ne[1] : 0;
        const size_t max_col  = std::max(std::max(col_up, col_gate), col_down);
        std::vector<char> buf;

        for (int i = 0; i < n_used; ++i) {
            const int32_t e = ids[i];
            if (e < 0 || e >= n_expert) {
                continue;
            }
            ++g_selected;
            if (swap_enable) {
                ++g_dsel;
            } else {
                ++g_psel;
            }
            // already resident (pinned hot or spare slot)?
            const int32_t cur_col = st.remap_host[(size_t) e];
            if (cur_col >= 0 && cur_col < t.n_hot) {
                ++g_pinned_hit;
                if (swap_enable) {
                    ++g_dhit;
                } else {
                    ++g_phit;
                }
                continue; // pinned hot
            }
            auto it = st.expert_to_slot.find(e);
            if (it != st.expert_to_slot.end() && it->second == cur_col) {
                ++g_spare_hit;
                if (swap_enable) {
                    ++g_dhit;
                } else {
                    ++g_phit;
                }
                // in a spare slot: bump its hit count and refresh the last-use tick
                const int32_t s = it->second - t.n_hot;
                if (s >= 0 && s < t.n_spare) {
                    ++st.slot_hits[(size_t) s];
                    st.slot_last_tick[(size_t) s] = st.clock;
                }
                continue;
            }
            if (cur_col >= t.n_hot && cur_col < t.n_mask) {
                // already bound to a spare slot but host bookkeeping is stale
                const int32_t s = cur_col - t.n_hot;
                st.slot_expert[(size_t) s] = e;
                st.slot_hits[(size_t) s]++;
                st.slot_last_tick[(size_t) s] = st.clock;
                st.expert_to_slot[e] = cur_col;
                ++g_spare_hit;
                if (swap_enable) {
                    ++g_dhit;
                } else {
                    ++g_phit;
                }
                continue;
            }
            ++g_cold;

            // occurrence gate: only swap in experts seen often enough. Routing
            // is highly dispersed (single-occurrence experts dominate), so a
            // fresh cold expert is evidence-less noise - counting filters it
            // without paying for a 19MB copy per hit.
            int occ = (++st.candidate_count[e]);
            if (occ < min_occ) {
                continue;
            }
            // threshold reached: stop counting (we only need the bucket)
            st.candidate_count[e] = min_occ;
            if (!swap_enable) {
                continue; // prefill: count only, swap happens during decode
            }

            // pick a spare slot: free first, else longest-idle occupant
            if (swap_budget != nullptr && *swap_budget <= 0) {
                continue; // batch swap budget exhausted
            }
            int slot = -1;
            uint64_t max_idle = 0;
            bool all_occupied = true;
            for (int s = 0; s < t.n_spare; ++s) {
                if (st.slot_expert[(size_t) s] < 0) {
                    slot = s;
                    all_occupied = false;
                    break; // free slot: fill it immediately
                }
                // grace period: a freshly swapped-in expert must be given time
                // to accumulate hits, otherwise it is evicted before it can
                // pay for the 19MB copy (thrash)
                if (st.clock - st.slot_enter_tick[(size_t) s] < (uint64_t) grace) {
                    continue;
                }
                const uint64_t idle = st.clock - st.slot_last_tick[(size_t) s];
                if (idle > max_idle) {
                    max_idle = idle;
                    slot = s;
                }
            }
            if (slot < 0) {
                continue;
            }
            // calm period: with every slot occupied, only evict when the
            // longest-idle occupant has been inactive for CALM_IDLE_TICKS;
            // otherwise keep the active set and burn zero 19MB copy bandwidth.
            if (all_occupied && max_idle < calm_idle) {
                continue;
            }
            const int32_t hot_col = t.n_hot + slot;
            const int32_t victim = st.slot_expert[(size_t) slot];
            if (victim >= 0) {
                // evict the victim back to the cold path
                const int32_t mask_only[1] = { t.n_mask };
                ggml_backend_tensor_set(t.remap,      mask_only,  (size_t) victim * sizeof(int32_t), sizeof(int32_t));
                const float zero = 0.f;
                ggml_backend_tensor_set(t.flag,       &zero,      (size_t) victim * sizeof(float),   sizeof(float));
                const int32_t cold_id[1] = { victim };
                ggml_backend_tensor_set(t.cold_remap, cold_id,    (size_t) victim * sizeof(int32_t), sizeof(int32_t));
                st.expert_to_slot.erase(victim);
                st.remap_host[(size_t) victim] = t.n_mask;
            }

            // copy the cold expert's weight columns into the spare hot columns
            if (buf.size() < max_col) {
                buf.resize(max_col);
            }

            copy_expert_column(t.hot_up,   t.cold_up,   e, hot_col, col_up,   buf);
            copy_expert_column(t.hot_gate, t.cold_gate, e, hot_col, col_gate, buf);
            copy_expert_column(t.hot_down, t.cold_down, e, hot_col, col_down, buf);

            // bind: remap[e] = hot_col, flag[e] = 1, cold_remap[e] = -1
            const int32_t hot_col_arr[1] = { hot_col };
            const float   one[1]         = { 1.0f };
            const int32_t neg1[1]        = { -1 };
            ggml_backend_tensor_set(t.remap,      hot_col_arr, (size_t) e * sizeof(int32_t), sizeof(int32_t));
            ggml_backend_tensor_set(t.flag,       one,         (size_t) e * sizeof(float),   sizeof(float));
            ggml_backend_tensor_set(t.cold_remap, neg1,        (size_t) e * sizeof(int32_t), sizeof(int32_t));

            st.remap_host[(size_t) e] = hot_col;
            st.slot_expert[(size_t) slot] = e;
            st.slot_hits[(size_t) slot]   = 1; // reset: newcomer starts its own hit count
            st.slot_enter_tick[(size_t) slot] = st.clock; // clock advances inside llama_context::process_ubatch
            st.slot_last_tick[(size_t) slot]  = st.clock;
            st.expert_to_slot[e] = hot_col;
            ++g_swapped;
            if (swap_budget != nullptr) {
                --(*swap_budget);
            }
        }
    // print one summary line per PRINT_EVERY selected slots (env-tunable so the
        // hit rates are visible on a live terminal even during short generation)
        const uint64_t print_every = std::max<uint64_t>(1, getenv("LLAMA_EXPERT_LRU_PRINT_EVERY") ? (uint64_t) std::max(0, std::atoi(getenv("LLAMA_EXPERT_LRU_PRINT_EVERY"))) : 256);
        if (g_selected / print_every != g_print_n) {
            g_print_n = g_selected / print_every;
            const auto pct = [](uint64_t n, uint64_t d) -> double { return d ? 100.0 * (double) n / (double) d : 0.0; };
            fprintf(stderr, "EXPERTLRU sel=%llu pinned=%.1f%% sparehit=%.1f%% cold=%.1f%% swaps=%llu totalhit=%.1f%%\n",
                    (unsigned long long) g_selected,
                    pct(g_pinned_hit, g_selected), pct(g_spare_hit, g_selected),
                    pct(g_cold, g_selected), (unsigned long long) g_swapped,
                    pct(g_pinned_hit + g_spare_hit, g_selected));
        }
    }

    void reset_window(std::vector<LayerState> & states) {
        for (auto & st : states) {
            st.candidate_count.clear();
        }
    }

} // namespace llm_expert_lru