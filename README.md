# strixhalo_5090m_llamacpp

**English** | [简体中文](README.zh-CN.md)

llama.cpp fork for split inference on one machine: AMD Strix Halo (Ryzen AI Max 395, gfx1151) + NVIDIA RTX 5090 — one `llama-server`, two backends (CUDA attention + ROCm experts). Focused on `qwen4exp` MoE-hybrid models (Qwen3.8-Flash-Next) with lightning-indexer QSA sparse attention.

**Repo layout / 仓库结构**: the patched llama.cpp tree lives in [`llama.cpp/`](llama.cpp/) (based on upstream `main`, ~Aug 2026, post #27886); fork assets are at the root:

```
llama.cpp/            patched upstream tree (all model/backend code changes inside)
configs/              mmq_settings + expert heat tables (DeepSeek-V4-Flash, Qwen3.8-Next)
docs/                 design notes and production patches
scripts/              one-off tuning tools
Dockerfile.dual       self-contained dual-backend build (CUDA sm_120 + ROCm gfx1151)
Containerfile.qwen-prod  reproducible build on a proven toolchain base
```

## Changes vs upstream (22 commits)

**Dual-backend execution**
- `ggml-backend-meta`: mirrored/split graph scheduler — one graph, layers split across CUDA0+ROCm0 (`-dev CUDA0,ROCm0 -ts 1,0`), cross-backend tensor relay, split-state sync
- MoE hot-expert system: `--hot-exp-file` heat tables, expert LRU harvest throttling, counting-sort ids pipeline, per-model `mmq_settings.json`

**Strix Halo / UMA tuning**
- gfx1151 rdna3-5 downblock fix (critical for 8060S), UMA HIP host-addressable buffers, `GGML_HIP_MEM_MODE`/`MEM_ADVISE` runtime switches, value-based `GGML_HIP_DISABLE_GRAPHS`

**qwen4exp QSA correctness + prefill/decode performance**
- `llama-memory-hybrid-idx`: pooled indexer-key cache (incremental block pooling, cross-layer shared QSA input, graph reuse)
- windowed n-gram predecessor scan, indexer head-sum as strided slice adds
- gathered QSA **decode** attention; **hot-block gather prefill** (headline): prefill attention over one shared hot window per layer — block-level top-k over indexer scores + the chunk's own cells
- NextN/MTP draft head (`--spec-type draft-mtp`)

**Env switches**: `LLAMA_QSA_DISABLE`, `LLAMA_QSA_HOTBLOCK`, `LLAMA_QSA_HOTBLOCK_CELLS`, `LLAMA_QSA_GATHER`, `LLAMA_QSA_NO_POOLED_CACHE`, `LLAMA_QSA_HOTBLOCK_DEBUG`, `GGML_HIP_DISABLE_GRAPHS`, `GGML_HIP_MEM_MODE`, `LLAMA_EXPERT_LRU_HARVEST_EVERY`

## Measured gains (Qwen3.8-Flash-Next-UD-IQ4_XS, APU 8060S)

| prefill t/s | 16k ctx | 32k ctx | 48k ctx | 16k->48k decay |
|---|---|---|---|---|
| dense QSA (env off) | 278 | 244 | 224 | -23% |
| hot-block (default) | 322 | 369 | **385** | **-8%** |

PPL: 2048 ctx `3.7872 +/- 0.033` (baseline `3.7892`, chunk-identical); 49k ctx `3.53` vs `3.88` full-attention ground truth. Decode: unchanged.

## Build

```bash
# self-contained dual-backend build (CUDA sm_120 + ROCm gfx1151)
podman build -f Dockerfile.dual -t localhost/llamacpp:dual .
# native cmake
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON \
      -DGGML_CUDA=ON -DGGML_HIP=ON -DGGML_NATIVE=OFF -DGGML_CPU=x86-64-v3 \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build --target llama-server -j16
```

## Configs & launch commands

**Qwen3.8-Flash-Next (qwen4exp), 5090 + Strix Halo dual-card:**

```bash
export LD_LIBRARY_PATH=<build>/bin:/nvidia-libs
export GGML_QSA_HOTBLOCK=1            # hot-block gather prefill (default on; 0 = dense QSA)

llama-server \
  -m Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  -dev CUDA0,ROCm0 -ngl 999 -ts 1,0 \
  -ot "blk\.([1-9]|[1-2][0-9]|3[0-9]|4[0-7])\.ffn_.*_exps\.weight=ROCm0" \
  --hot-exp-file configs/qwen3.8-next/hot2000_spare.json \
  --host 0.0.0.0 --port 5234 -n 24000 -c 256000 -fa on -b 2048 -ub 2048 \
  -t 2 -tb 2 --parallel 1 --no-mmproj --no-mmap \
  --temp 0.7 --top-p 0.95 --top-k 20 --jinja --fit off \
  -ctv q8_0 -ctk q8_0 --numa distribute --metrics
```

Config: `configs/qwen3.8-next/mmq_settings.json` = `max_j_occ2=64, ubatch=2048, ids_helper_smem=auto`.

**DeepSeek-V4-Flash (MoE), 5090 + Strix Halo dual-card:**

```bash
export LD_LIBRARY_PATH=<build>/bin:/nvidia-libs
export HSA_OVERRIDE_GFX_VERSION=11.5.1 HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode
export GGML_HIP_ENABLE_UNIFIED_MEMORY=1 GGML_HIP_DISABLE_GRAPHS=1 GGML_MMQ_MAX_J_OCC2=64
export LLAMA_EXPERT_LRU_SLOTS_PER_LAYER=1 LLAMA_EXPERT_LRU_APPLY_EVERY=32
export LLAMA_EXPERT_LRU_MAX_SWAP_PER_BATCH=1 LLAMA_EXPERT_LRU_CALM_HITS=2

llama-server \
  -m DeepSeek-V4-Flash-0731-UD-IQ4_XS-00001-of-00004.gguf \
  -dev CUDA0,ROCm0 -ngl 999 -ts 1,0 \
  -ot "blk\.([2-9]|[1-2][0-9]|3[0-9]|4[0-2])\.ffn_.*_exps\.weight=ROCm0" \
  --hot-exp-file configs/deepseek-v4-flash/hot250_spare.json \
  --host 0.0.0.0 --port 5234 -n 24000 -c 132384 -fa on -b 2047 -ub 2047 \
  -t 2 -tb 2 --parallel 1 --no-mmproj --no-mmap \
  --temp 0.7 --top-p 0.95 --top-k 20 --jinja --fit off \
  -ctv q8_0 -ctk q8_0 --numa distribute --metrics
```

Config: `configs/deepseek-v4-flash/mmq_settings.json` = `max_j_occ2=64, ubatch=2047, ids_helper_smem=full` (conservative; `auto` also works on newer builds). Heat tables: `hot250_spare.json` (default) or `hot600_spare.json`.

Heat-table device keys follow your `-dev` mapping (`"CUDA0": {...}` pins those expert ids onto CUDA0); regenerate with your own traffic if the tables do not match your workload.

Upstream llama.cpp documentation: [`llama.cpp/README.md`](llama.cpp/README.md). 上中文文档:[README.zh-CN.md](README.zh-CN.md)

License: follows upstream llama.cpp ([MIT](llama.cpp/LICENSE)).
