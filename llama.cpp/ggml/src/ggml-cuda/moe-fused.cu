#include "moe-fused.cuh"
#include "mmvq.cuh"

#include <cstring>

static __device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

static ggml_tensor make_f32_tensor(float * data, int64_t ne0, int64_t ne1, int64_t ne2) {
    ggml_tensor t = {};
    t.type = GGML_TYPE_F32;
    t.data  = data;
    t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = 1;
    t.nb[0] = sizeof(float); t.nb[1] = t.nb[0]*ne0; t.nb[2] = t.nb[1]*ne1; t.nb[3] = t.nb[2]*ne2;
    return t;
}

// weighted sum over used expert slots; slots whose weight is 0 are skipped
// without reading the (possibly stale) per-slot result.
static __global__ void moe_combine_kernel(
    const char * __restrict__ experts,
    const char * __restrict__ weights,
    char * __restrict__ output,
    const int n_embd,
    const int n_used,
    const int n_tokens,
    const size_t experts_nb0,
    const size_t experts_nb1,
    const size_t experts_nb2,
    const size_t weights_nb0,
    const size_t weights_nb1,
    const size_t output_nb0,
    const size_t output_nb1) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_embd * n_tokens;
    if (idx >= total) return;

    const int h = idx % n_embd;
    const int t = idx / n_embd;
    float sum = 0.0f;
    for (int e = 0; e < n_used; ++e) {
        const float w = *(const float *)(weights +
            (size_t) e * weights_nb0 +
            (size_t) t * weights_nb1);
        if (w == 0.0f) {
            if (e == 0) sum = 0.0f;
            continue;
        }
        const float v = *(const float *)(experts +
            (size_t) h * experts_nb0 +
            (size_t) e * experts_nb1 +
            (size_t) t * experts_nb2);
        const float prod = __fmul_rn(v, w);
        sum = (e == 0) ? prod : __fadd_rn(sum, prod);
    }
    *(float *)(output +
        (size_t) h * output_nb0 +
        (size_t) t * output_nb1) = sum;
}

// one block computes the whole FFN of one token: gate/up projections from the
// quantized expert rows into shared memory, then the down projection and the
// weighted expert sum straight into the output. supports IQ2_XS (17),
// IQ3_XXS (18) and IQ4_XS (23, down only). single token (decode) only.
template<int BLOCK_SIZE>
static __global__ void moe_fused_kernel(
    const float * __restrict__ input,
    const void * __restrict__ gate_w,
    const void * __restrict__ up_w,
    const void * __restrict__ down_w,
    const int32_t * __restrict__ expert_ids,
    const float * __restrict__ expert_wts,
    const void * __restrict__ sh_gate_w,
    const void * __restrict__ sh_up_w,
    const void * __restrict__ sh_down_w,
    const void * __restrict__ sh_gate_inp_w,
    float * __restrict__ output,
    const int n_embd,
    const int ff_dim,
    const int n_expert_used,
    const int n_expert_total,
    const int gate_type,
    const int up_type,
    const int down_type,
    const size_t gate_row_stride,
    const size_t up_row_stride,
    const size_t down_row_stride,
    const size_t gate_expert_stride,
    const size_t up_expert_stride,
    const size_t down_expert_stride,
    const size_t gate_block_stride,
    const size_t up_block_stride,
    const size_t down_block_stride) {

    extern __shared__ float gu_smem[];

    const int tid = threadIdx.x;
    const int nblocks_in = n_embd / QK_K;
    const int nblocks_ff = ff_dim / QK_K;


    for (int e = 0; e < n_expert_used; e++) {
        // cold_ids marks hot-owned slots as -1 (their expert_wts entry is 0);
        // clamp to row 0 so the unused read stays in-bounds and NaN-free.
        const int eidx = expert_ids[e] < 0 ? 0 : expert_ids[e];
        const char * g_exp = (const char *) gate_w + (size_t) eidx * gate_expert_stride;
        const char * u_exp = (const char *) up_w   + (size_t) eidx * up_expert_stride;

        for (int k = tid; k < ff_dim; k += BLOCK_SIZE) {
            float g_sum = 0.0f;
            float u_sum = 0.0f;

            const char * g_row = g_exp + k * gate_row_stride;
            const char * u_row = u_exp + k * up_row_stride;

            if (gate_type == 17) {
                for (int b = 0; b < nblocks_in; b++) {
                    const block_iq2_xs * g_blk = (const block_iq2_xs *)(g_row + b * sizeof(block_iq2_xs));
                    const float d = (float)g_blk->d;
                    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                        for (int l = 0; l < 4; ++l) {
                            const float db = d * (0.5f + ((g_blk->scales[ib32] >> 4*(l/2)) & 0xf)) * 0.25f;
                            const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (g_blk->qs[4*ib32 + l] & 511));
                            const uint8_t signs = ksigns_iq2xs[g_blk->qs[4*ib32 + l] >> 9];
                            const int off = b*QK_K + 32*ib32 + 8*l;
                            for (int j = 0; j < 8; ++j)
                                g_sum += db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f) * input[off + j];
                        }
                    }
                }
            } else if (gate_type == 18) {
                for (int b = 0; b < nblocks_in; b++) {
                    const block_iq3_xxs * g_blk = (const block_iq3_xxs *)(g_row + b * sizeof(block_iq3_xxs));
                    const float gd = (float)g_blk->d;
                    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                        const uint8_t * qs = g_blk->qs + 8*ib32;
                        const uint16_t * gas = (const uint16_t *)(g_blk->qs + QK_K/4) + 2*ib32;
                        const uint32_t aux32 = gas[0] | (gas[1] << 16);
                        const float gdb = gd * (0.5f + (aux32 >> 28)) * 0.5f;
                        for (int l = 0; l < 4; ++l) {
                            const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                            const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + qs[2*l+0]);
                            const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + qs[2*l+1]);
                            const int off = b*QK_K + 32*ib32 + 8*l;
                            for (int j = 0; j < 4; ++j) {
                                g_sum += gdb * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f) * input[off + j];
                                g_sum += gdb * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f) * input[off + 4 + j];
                            }
                        }
                    }
                }
            }

            if (up_type == 17) {
                for (int b = 0; b < nblocks_in; b++) {
                    const block_iq2_xs * u_blk = (const block_iq2_xs *)(u_row + b * sizeof(block_iq2_xs));
                    const float d = (float)u_blk->d;
                    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                        for (int l = 0; l < 4; ++l) {
                            const float db = d * (0.5f + ((u_blk->scales[ib32] >> 4*(l/2)) & 0xf)) * 0.25f;
                            const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (u_blk->qs[4*ib32 + l] & 511));
                            const uint8_t signs = ksigns_iq2xs[u_blk->qs[4*ib32 + l] >> 9];
                            const int off = b*QK_K + 32*ib32 + 8*l;
                            for (int j = 0; j < 8; ++j)
                                u_sum += db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f) * input[off + j];
                        }
                    }
                }
            } else if (up_type == 18) {
                for (int b = 0; b < nblocks_in; b++) {
                    const block_iq3_xxs * u_blk = (const block_iq3_xxs *)(u_row + b * sizeof(block_iq3_xxs));
                    const float ud = (float)u_blk->d;
                    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                        const uint8_t * qs = u_blk->qs + 8*ib32;
                        const uint16_t * gas = (const uint16_t *)(u_blk->qs + QK_K/4) + 2*ib32;
                        const uint32_t aux32 = gas[0] | (gas[1] << 16);
                        const float udb = ud * (0.5f + (aux32 >> 28)) * 0.5f;
                        for (int l = 0; l < 4; ++l) {
                            const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                            const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + qs[2*l+0]);
                            const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + qs[2*l+1]);
                            const int off = b*QK_K + 32*ib32 + 8*l;
                            for (int j = 0; j < 4; ++j) {
                                u_sum += udb * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f) * input[off + j];
                                u_sum += udb * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f) * input[off + 4 + j];
                            }
                        }
                    }
                }
            }

            gu_smem[e * ff_dim + k] = silu_f32(g_sum) * u_sum;
        }
    }
    __syncthreads();

    // Phase 2: down projection + weighted sum
    const char * d_base = (const char *) down_w;
    for (int n = tid; n < n_embd; n += BLOCK_SIZE) {
        float sum = 0.0f;

        for (int e = 0; e < n_expert_used; e++) {
            // same clamp as phase 1: -1 rows carry zero weight
            const int eidx = expert_ids[e] < 0 ? 0 : expert_ids[e];
            const char * d_exp = d_base + (size_t) eidx * down_expert_stride;
            const char * d_row = d_exp + n * down_row_stride;
            float d_sum = 0.0f;

            if (down_type == 17) {
                for (int b = 0; b < nblocks_ff; b++) {
                    const block_iq2_xs * d_blk = (const block_iq2_xs *)(d_row + b * sizeof(block_iq2_xs));
                    const float d = (float)d_blk->d;
                    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                        for (int l = 0; l < 4; ++l) {
                            const float db = d * (0.5f + ((d_blk->scales[ib32] >> 4*(l/2)) & 0xf)) * 0.25f;
                            const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (d_blk->qs[4*ib32 + l] & 511));
                            const uint8_t signs = ksigns_iq2xs[d_blk->qs[4*ib32 + l] >> 9];
                            const int off = b*QK_K + 32*ib32 + 8*l;
                            for (int j = 0; j < 8; ++j)
                                d_sum += db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f) * gu_smem[e * ff_dim + off + j];
                        }
                    }
                }
            } else if (down_type == 18) {
                for (int b = 0; b < nblocks_ff; b++) {
                    const block_iq3_xxs * d_blk = (const block_iq3_xxs *)(d_row + b * sizeof(block_iq3_xxs));
                    const float dd = (float)d_blk->d;
                    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                        const uint8_t * qs = d_blk->qs + 8*ib32;
                        const uint16_t * gas = (const uint16_t *)(d_blk->qs + QK_K/4) + 2*ib32;
                        const uint32_t aux32 = gas[0] | (gas[1] << 16);
                        const float ddb = dd * (0.5f + (aux32 >> 28)) * 0.5f;
                        for (int l = 0; l < 4; ++l) {
                            const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                            const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + qs[2*l+0]);
                            const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + qs[2*l+1]);
                            const int off = b*QK_K + 32*ib32 + 8*l;
                            for (int j = 0; j < 4; ++j) {
                                d_sum += ddb * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f) * gu_smem[e * ff_dim + off + j];
                                d_sum += ddb * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f) * gu_smem[e * ff_dim + off + 4 + j];
                            }
                        }
                    }
                }
            } else if (down_type == 23) {
                for (int b = 0; b < nblocks_ff; b++) {
                    const block_iq4_xs * d_blk = (const block_iq4_xs *)(d_row + b * sizeof(block_iq4_xs));
                    const float dd = (float)d_blk->d;
                    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                        const int ls = ((d_blk->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf)
                                     | (((d_blk->scales_h >> 2*ib32) & 3) << 4);
                        const float ddb = dd * (ls - 32);
                        const uint8_t * q4 = d_blk->qs + 16*ib32;
                        for (int l = 0; l < 4; ++l) {
                            const int off = b*QK_K + 32*ib32 + 8*l;
                            d_sum += ddb * kvalues_iq4nl[q4[4*l+0] & 0xf] * gu_smem[e * ff_dim + off + 0];
                            d_sum += ddb * kvalues_iq4nl[q4[4*l+1] & 0xf] * gu_smem[e * ff_dim + off + 1];
                            d_sum += ddb * kvalues_iq4nl[q4[4*l+2] & 0xf] * gu_smem[e * ff_dim + off + 2];
                            d_sum += ddb * kvalues_iq4nl[q4[4*l+3] & 0xf] * gu_smem[e * ff_dim + off + 3];
                            d_sum += ddb * kvalues_iq4nl[q4[4*l+0] >>  4] * gu_smem[e * ff_dim + off + 4];
                            d_sum += ddb * kvalues_iq4nl[q4[4*l+1] >>  4] * gu_smem[e * ff_dim + off + 5];
                            d_sum += ddb * kvalues_iq4nl[q4[4*l+2] >>  4] * gu_smem[e * ff_dim + off + 6];
                            d_sum += ddb * kvalues_iq4nl[q4[4*l+3] >>  4] * gu_smem[e * ff_dim + off + 7];
                        }
                    }
                }
            }
            sum += expert_wts[e] * d_sum;
        }

        output[n] = sum;
    }
}

static __global__ void ds4_peer_copy_f32_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}

void ggml_cuda_op_moe_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int mode = ggml_get_op_params_i32(dst, 0);
    if (mode == GGML_MOE_FUSED_DEFERRED_PEER_COPY) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(dst));

        void * event_context = nullptr;
        memcpy(&event_context,
               &dst->op_params[GGML_MOE_FUSED_DEFERRED_EVENT_WORD],
               sizeof(event_context));
        GGML_ASSERT(event_context);

        void * source_data = nullptr;
        memcpy(&source_data,
               &dst->op_params[GGML_MOE_FUSED_DEFERRED_SOURCE_WORD],
               sizeof(source_data));
        GGML_ASSERT(source_data);

        if (ggml_get_op_params_i32(
                dst, GGML_MOE_FUSED_DEFERRED_EXTERNAL_WAIT_WORD) == 0) {
            // Same-split control: capture the wait at its exact position in
            // the main GPU graph.
            CUDA_CHECK(cudaStreamWaitEvent(
                ctx.stream(), (cudaEvent_t) event_context, 0));
        }

        // Peer read on the consumer stream: the producer wrote to its own
        // memory, the event above gates this copy.
        const int64_t n = ggml_nelements(dst);
        const int block = 256;
        const int grid = (int) ((n + block - 1) / block);
        ds4_peer_copy_f32_kernel<<<grid, block, 0, ctx.stream()>>>(
            (const float *) source_data, (float *) dst->data, n);
        return;
    }
    if (mode < 0) {
        GGML_ABORT("%s: unsupported MOE_FUSED mode %d", __func__, mode);
    }

// default (mode >= 0 carries n_embd; op_params_f32[1] carries the swiglu
// clamp limit, 0 = off): fused MoE FFN via multi-block mmvq, decode only.
// 3 kernels: fused gate+up+swiglu, down projection, weighted combine.
    const ggml_tensor * input      = dst->src[0];
    const ggml_tensor * gate_w     = dst->src[1];
    const ggml_tensor * up_w       = dst->src[2];
    const ggml_tensor * down_w     = dst->src[3];
    const ggml_tensor * expert_ids = dst->src[4];
    const ggml_tensor * expert_wts = dst->src[5];

    const int n_embd        = (int) gate_w->ne[0];
    const int ff_dim        = (int) gate_w->ne[1];
    const int n_expert_used = (int) expert_ids->ne[0];
    const float clamp_limit = ggml_get_op_params_f32(dst, 1);

    GGML_ASSERT(input->type == GGML_TYPE_F32 && ggml_is_contiguous(input));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(up_w->type == gate_w->type);
    GGML_ASSERT(ggml_are_same_stride(up_w, gate_w));

    // stage 1: fused gate+up mmvq with clamped swiglu -> gu [n_ff, n_used, 1]
    ggml_cuda_pool_alloc<float> gu_alloc(ctx.pool(), (size_t) ff_dim * n_expert_used);
    ggml_tensor gu = make_f32_tensor(gu_alloc.ptr, ff_dim, n_expert_used, 1);
    GGML_ABORT("moe fused gate kernel requires the hot-expert mmvq path (not compiled)");

    // stage 2: down projection per slot -> experts [n_embd, n_used, 1]
    ggml_cuda_pool_alloc<float> ex_alloc(ctx.pool(), (size_t) n_embd * n_expert_used);
    ggml_tensor experts = make_f32_tensor(ex_alloc.ptr, n_embd, n_expert_used, 1);
    ggml_cuda_mul_mat_vec_q(ctx, down_w, &gu, expert_ids, &experts, nullptr);

    // stage 3: weighted combine over used slots -> dst [n_embd, 1]
    const int block = 256;
    const int total = n_embd;
    moe_combine_kernel<<<(total + block - 1) / block, block, 0, ctx.stream()>>>(
        (const char *) experts.data,
        (const char *) expert_wts->data,
        (char *) dst->data,
        n_embd, n_expert_used, 1,
        experts.nb[0], experts.nb[1], experts.nb[2],
        expert_wts->nb[0], expert_wts->nb[1],
        dst->nb[0], dst->nb[1]);
    CUDA_CHECK(cudaGetLastError());
}
