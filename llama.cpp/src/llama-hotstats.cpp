// llama-hotstats.cpp - hot expert statistics (see llama-hotstats.h)

#include "llama-hotstats.h"

#include "llama-impl.h" // LLAMA_LOG_INFO etc.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace llama_hotstats {

namespace {

    struct LayerCounts {
        // expert id -> times selected
        std::unordered_map<int, uint64_t> expert_counts;
        // cumulative selected-expert slots seen (n_used * n_tokens summed)
        uint64_t total_slots = 0;
        // prefill vs decode tokens (approx: batch>1 => prefill)
        uint64_t prefill_slots = 0;
        uint64_t decode_slots  = 0;
    };

    std::mutex g_mutex;
    std::vector<LayerCounts> g_layers;          // indexed by layer id (0..n_layer)
    int g_n_layer = 0;                          // max layer index seen + 1
    std::atomic<bool> g_enabled(false);
    std::string g_file_path = "hot_experts.json";
    std::chrono::steady_clock::time_point g_last_dump;
    std::vector<std::pair<int, const void *>> g_snaps;
    std::chrono::seconds g_interval{60};

    int parse_env_int(const char * name, int def) {
        const char * val = std::getenv(name);
        if (!val || val[0] == '\0') {
            return def;
        }
        return std::atoi(val);
    }

    const char * parse_env_str(const char * name, const char * def) {
        const char * val = std::getenv(name);
        return (val && val[0] != '\0') ? val : def;
    }

    // serialize snapshot; simulates top-N coverage for the union of hot experts
    void write_snapshot(std::ofstream & out) {
        out << "{\n";
        out << "  \"version\": 1,\n";
        out << "  \"n_layers\": " << g_n_layer << ",\n";
        out << "  \"interval_s\": " << g_interval.count() << ",\n";

        // global expert frequency across all layers (for top-N simulation)
        std::unordered_map<int, uint64_t> global_counts;
        uint64_t global_total = 0;
        for (const auto & lc : g_layers) {
            for (const auto & kv : lc.expert_counts) {
                global_counts[kv.first] += kv.second;
                global_total += kv.second;
            }
        }

        out << "  \"global_total_slots\": " << global_total << ",\n";

        // top-N coverage simulation (N = 1..64, and a few larger values)
        out << "  \"top_n_coverage\": {\n";
        std::vector<std::pair<int, uint64_t>> ranked(global_counts.begin(), global_counts.end());
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto & a, const auto & b) { return a.second > b.second; });
        std::vector<int> ns = {1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256};
        uint64_t cum = 0;
        int slot = 0;
        for (int i = 0; i < (int) ranked.size(); ++i) {
            cum += ranked[i].second;
            slot = ranked.size() > (size_t) i ? i : slot;
            int rank = i + 1;
            if (std::find(ns.begin(), ns.end(), rank) != ns.end() || rank == (int) ranked.size()) {
                out << "    \"top" << rank << "\": {"
                    << "\"slots\": " << cum << ","
                    << "\"coverage\": " << (global_total ? (double) cum / (double) global_total : 0.0)
                    << "},\n";
            }
        }
        out << "    \"all\": {\"slots\": " << global_total
            << ", \"coverage\": 1.0}\n";
        out << "  },\n";

        // full expert frequency list (sorted desc); hotcfg slices top-N from this
        out << "  \"global_top\": [\n";
        for (int i = 0; i < (int) ranked.size(); ++i) {
            out << "    {\"expert\": " << ranked[i].first
                << ", \"count\": " << ranked[i].second << "}";
            out << (i + 1 < (int) ranked.size() ? ",\n" : "\n");
        }
        out << "  ],\n";

        // per-layer detail
        std::vector<int> nonempty;
        for (int layer = 0; layer < g_n_layer; ++layer) {
            if (g_layers[layer].total_slots > 0) {
                nonempty.push_back(layer);
            }
        }
        out << "  \"layers\": {\n";
        for (size_t li = 0; li < nonempty.size(); ++li) {
            const int layer = nonempty[li];
            const auto & lc = g_layers[layer];
            out << "    \"blk." << layer << "\": {"
                << "\"total_slots\": " << lc.total_slots << ","
                << "\"prefill\": " << lc.prefill_slots << ","
                << "\"decode\": " << lc.decode_slots << ","
                << "\"n_experts_seen\": " << lc.expert_counts.size() << ","
                << "\"top\": [";
            std::vector<std::pair<int, uint64_t>> lrank(lc.expert_counts.begin(), lc.expert_counts.end());
            std::sort(lrank.begin(), lrank.end(),
                      [](const auto & a, const auto & b) { return a.second > b.second; });
            for (int i = 0; i < (int) lrank.size(); ++i) {
                out << "{\"e\": " << lrank[i].first << ", \"c\": " << lrank[i].second << "}";
                if (i + 1 < (int) lrank.size()) out << ", ";
            }
            out << "]}";
            if (li + 1 < nonempty.size()) out << ",\n";
        }
        out << "\n  }\n";
        out << "}\n";
    }

} // namespace

bool init() {
    g_enabled.store(parse_env_int("LLAMA_HOT_STAT", 0) != 0);
    if (!g_enabled.load()) {
        return false;
    }
    g_file_path = parse_env_str("LLAMA_HOT_STAT_FILE", "hot_experts.json");
    g_interval = std::chrono::seconds(parse_env_int("LLAMA_HOT_STAT_INTERVAL", 60));
    g_last_dump = std::chrono::steady_clock::now();
    fprintf(stderr, "[hotstats] enabled, file=%s interval=%llds\n",
            g_file_path.c_str(), (long long) g_interval.count());
    return true;
}

bool enabled() {
    return g_enabled.load();
}

void record(int layer, const int32_t * ids, int n_used, int n_tokens) {
    if (!g_enabled.load() || ids == nullptr || n_used <= 0 || n_tokens <= 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);

    if (layer >= (int) g_layers.size()) {
        g_layers.resize(layer + 1);
    }
    if (layer + 1 > g_n_layer) {
        g_n_layer = layer + 1;
    }

    auto & lc = g_layers[layer];
    const int64_t n_slots = (int64_t) n_used * n_tokens;

    for (int64_t i = 0; i < n_slots; ++i) {
        const int32_t e = ids[i];
        if (e < 0) {
            continue; // masked / unused slot
        }
        lc.expert_counts[e]++;
    }
    lc.total_slots += (uint64_t) n_slots;
    if (n_tokens > 1) {
        lc.prefill_slots += (uint64_t) n_slots;
    } else {
        lc.decode_slots  += (uint64_t) n_slots;
    }
}

void dump(const std::string & path) {
    if (!g_enabled.load()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    std::ofstream out(path);
    if (!out.is_open()) {
        LLAMA_LOG_ERROR("%s: cannot open output file %s\n", __func__, path.c_str());
        return;
    }
    write_snapshot(out);
    out.close();
    LLAMA_LOG_INFO("%s: hot expert stats written to %s\n", __func__, path.c_str());
}

void register_snap(int layer, const void * ids_tensor) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_snaps.emplace_back(layer, ids_tensor);
}

void clear_snaps() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_snaps.clear();
}

const std::vector<std::pair<int, const void *>> & snaps() {
    return g_snaps;
}

void maybe_dump() {
    if (!g_enabled.load()) {
        return;
    }
    auto now = std::chrono::steady_clock::now();
    if (now - g_last_dump >= g_interval) {
        g_last_dump = now;
        dump(g_file_path);
    }
}

void fini() {
    if (!g_enabled.load()) {
        return;
    }
    dump(g_file_path);
    g_enabled.store(false);
}

} // namespace llama_hotstats