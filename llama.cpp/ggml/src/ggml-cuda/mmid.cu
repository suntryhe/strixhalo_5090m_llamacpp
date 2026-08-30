#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used >= 0 && expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
        const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used; // Padded to next higher power of 2.
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used >= 0 && expert_used < expert;

            // MoE hot-expert offload: a token may map multiple slots to the same expert
            // (the mask column of a compact tensor). Count every matching slot so each
            // gets its own compact row. For standard MoE (<= 1 slot per token per expert)
            // this reduces to the original behavior.
            const int match = iex_used != -1 ? 1 : 0;

            // Use ballot + popc for intra-group exclusive prefix sum and group match count.
            // This is more robust than shfl for the segmented layout (neu_padded lanes per token).
            const uint64_t ballot = __ballot_sync(~0ull, match);
            const int seg_base = (threadIdx.x / neu_padded) * neu_padded;
            const uint64_t seg_mask = ((1ull << neu_padded) - 1) << seg_base;
            const uint64_t low_mask  = ((1ull << (threadIdx.x % neu_padded)) - 1) << seg_base;
            const int intra_rank = __popcll(ballot & low_mask);
            const int group_match = __popcll(ballot & seg_mask);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, group_match, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower + intra_rank] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + group_match, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    // MoE hot-expert offload (cold chain, -1 slots): the write_inverse map only fills
    // rows for slots that matched an expert. Slots with id == -1 keep pool garbage,
    // and the quantize scatter later indexes src1_q8_1 with those stale values,
    // corrupting the quantized y buffer. Mark every -1 slot so scatter can skip it.
    if (write_inverse) {
        for (int it = 0; it < n_tokens; ++it) {
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                if (expert_used < 0) {
                    ids_src1[it*n_expert_used + iex] = -1;
                }
            }
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, const bool has_mask, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    // Staging size: the generic path (template 0) stages at most one entry per
    // token, and the template path without a hot-expert mask column sees at most
    // one matching slot per token, so n_tokens entries always fit. Only the
    // template path with a mask column can pile every slot onto one expert,
    // which needs the worst-case n_tokens*n_expert_used room.
    const bool worst_case = has_mask && n_expert_used_template != 0;
    const size_t nbytes_shared = (worst_case ? (size_t)n_tokens*n_expert_used_var : (size_t)n_tokens)*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

static void launch_mm_ids_helper_legacy(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, const bool has_mask, cudaStream_t stream) {
    // Reached only via GGML_IDS_HELPER_SMEM=full; keep worst-case sizing when a
    // mask column can pile duplicate slots onto one expert.
    const bool mask_eff = has_mask;

    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, mask_eff, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, mask_eff, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, mask_eff, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, mask_eff, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, mask_eff, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, mask_eff, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, mask_eff, stream);
            break;
    }
}

// Counting-sort pipeline: histogram -> exclusive scan -> scatter. Two linear
// passes over the slot table instead of one rescan per expert (n_experts x
// table work). Row order inside an expert slice follows atomic arrival order;
// each output element is an order-independent dot product, so results match
// the legacy kernel. The scan reuses hist as the scatter cursor.

static __global__ void mm_ids_hist(
        const int32_t * __restrict__ ids, int * __restrict__ hist,
        const int ne_get_rows, const int n_expert_used, const int si1) {
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < ne_get_rows; i += gridDim.x*blockDim.x) {
        const int e = ids[(i / n_expert_used)*si1 + (i % n_expert_used)];
        if (e >= 0) {
            atomicAdd(hist + e, 1);
        }
    }
}

static __global__ void mm_ids_scan(
        int * __restrict__ hist, int * __restrict__ starts, int32_t * __restrict__ expert_bounds, const int n_experts) {
    if (threadIdx.x != 0) {
        return;
    }
    int acc = 0;
    for (int e = 0; e < n_experts; ++e) {
        starts[e] = acc;
        expert_bounds[e] = acc;
        acc += hist[e];
        hist[e] = 0;
    }
    expert_bounds[n_experts] = acc;
}

static __global__ void mm_ids_scatter(
        const int32_t * __restrict__ ids, int * __restrict__ cursor, const int * __restrict__ starts,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        const int ne_get_rows, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < ne_get_rows; i += gridDim.x*blockDim.x) {
        const int it  = i / n_expert_used;
        const int iex = i % n_expert_used;
        const int e   = ids[it*si1 + iex];
        if (e < 0) {
            if (write_inverse) {
                ids_src1[i] = -1;
            }
            continue;
        }
        const int pos = starts[e] + atomicAdd(cursor + e, 1);
        ids_dst[pos] = i;
        if (write_inverse) {
            ids_src1[i] = pos;
        } else {
            ids_src1[pos] = it*sis1 + iex % nchannels_y;
        }
    }
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, const bool has_mask, ggml_cuda_pool & pool, cudaStream_t stream) {
    // GGML_IDS_HELPER_SMEM=full falls back to the legacy per-expert rescan kernel.
    const char * env_smem = getenv("GGML_IDS_HELPER_SMEM");
    if (env_smem != nullptr && env_smem[0] == 'f') {
        launch_mm_ids_helper_legacy(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, has_mask, stream);
        return;
    }

    const int ne_get_rows = n_tokens*n_expert_used;
    ggml_cuda_pool_alloc<int32_t> hist(pool, n_experts);
    ggml_cuda_pool_alloc<int32_t> starts(pool, n_experts);
    CUDA_CHECK(cudaMemsetAsync(hist.get(), 0, n_experts*sizeof(int32_t), stream));

    int blocks = (ne_get_rows + 255) / 256;
    if (blocks > 2048) {
        blocks = 2048;
    }
    mm_ids_hist<<<blocks, 256, 0, stream>>>(ids, hist.get(), ne_get_rows, n_expert_used, si1);
    mm_ids_scan<<<1, 32, 0, stream>>>(hist.get(), starts.get(), expert_bounds, n_experts);
    mm_ids_scatter<<<blocks, 256, 0, stream>>>(ids, hist.get(), starts.get(), ids_src1, ids_dst,
        ne_get_rows, n_expert_used, nchannels_y, si1, sis1, write_inverse);
}
