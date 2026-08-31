# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

**English** | [简体中文](README.zh-CN.md)

---

## This fork: Strix Halo + RTX 5090 dual-backend inference

llama.cpp for split inference on one machine: AMD Strix Halo (Ryzen AI Max 395, gfx1151) + NVIDIA RTX 5090 — one `llama-server`, two backends (CUDA attention + ROCm experts). Focused on `qwen4exp` MoE-hybrid models (Qwen3.8-Flash-Next) that use lightning-indexer QSA sparse attention. Based on upstream `main` (~Aug 2026, post #27886); all fork changes are additive or env-gated. 上中文说明:[README.zh-CN.md](README.zh-CN.md)

### Changes vs upstream (22 commits)

**Dual-backend execution**
- `ggml-backend-meta`: mirrored/split graph scheduler — one graph, layers split across CUDA0+ROCm0 (`-dev CUDA0,ROCm0 -ts 1,0`), cross-backend tensor relay, split-state sync
- MoE hot-expert system: `--hot-exp-file` heat tables pin hot expert weights per device, expert LRU harvest throttling (`LLAMA_EXPERT_LRU_HARVEST_EVERY`), counting-sort ids pipeline replacing per-expert rescan, per-model `mmq_settings.json`

**Strix Halo / UMA tuning**
- gfx1151 rdna3-5 downblock fix (critical for 8060S), UMA HIP host-addressable buffers, `GGML_HIP_MEM_MODE`/`MEM_ADVISE` runtime switches, value-based `GGML_HIP_DISABLE_GRAPHS`

**qwen4exp QSA correctness + prefill/decode performance**
- `llama-memory-hybrid-idx`: pooled indexer-key cache (incremental block pooling, cross-layer shared QSA input, graph reuse via `can_reuse`)
- windowed n-gram predecessor scan in `get_prev_tokens`
- indexer head-sum as strided slice adds (removes full-size permutes)
- gathered QSA **decode** attention (attend exactly the selected rows)
- **hot-block gather prefill** (this fork's headline): prefill attention over one shared hot window per layer — block-level top-k over indexer scores + the chunk's own cells; replaces the `[n_tps, n_kv]` mask-expansion chain
- NextN/MTP draft head (`--spec-type draft-mtp`)

**Env switches**: `LLAMA_QSA_DISABLE`, `LLAMA_QSA_HOTBLOCK`, `LLAMA_QSA_HOTBLOCK_CELLS`, `LLAMA_QSA_GATHER`, `LLAMA_QSA_NO_POOLED_CACHE`, `LLAMA_QSA_HOTBLOCK_DEBUG`, `GGML_HIP_DISABLE_GRAPHS`, `GGML_HIP_MEM_MODE`, `LLAMA_EXPERT_LRU_HARVEST_EVERY`

### Measured gains (Qwen3.8-Flash-Next-UD-IQ4_XS, APU 8060S)

| prefill t/s | 16k ctx | 32k ctx | 48k ctx | 16k->48k decay |
|---|---|---|---|---|
| dense QSA (env off) | 278 | 244 | 224 | -23% |
| hot-block (default) | 322 | 369 | **385** | **-8%** |

PPL: 2048 ctx `3.7872 +/- 0.033` (baseline `3.7892`, chunk-identical); 49k ctx `3.53` vs `3.88` full-attention ground truth. Decode: unchanged (hot-block is prefill-only; QSA decode gather lands separately).

### Build

```bash
# self-contained dual-backend build (CUDA sm_120 + ROCm gfx1151)
podman build -f Dockerfile.dual -t localhost/llamacpp:dual .
# or the proven local toolchain-base path
podman build -f Containerfile.qwen-prod -t localhost/llamacpp:qwen-prod-v6 .
# native cmake
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON \
      -DGGML_CUDA=ON -DGGML_HIP=ON -DGGML_NATIVE=OFF -DGGML_CPU=x86-64-v3 \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build --target llama-server -j16
```

Run: `-dev CUDA0,ROCm0 -ts 1,0 -ot "blk\.([1-9]|...)\.ffn_.*_exps\.weight=ROCm0" --hot-exp-file <heat.json> -fa on` (see `Containerfile.qwen-prod` header for a full example).

### Configs & launch commands

Ready-to-use configs live in [`configs/`](configs/): per-model `mmq_settings.json` (auto-loaded by `common_init` when placed next to the model as `mmq_settings.json`) and heat tables for `--hot-exp-file`.

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

Config: `configs/deepseek-v4-flash/mmq_settings.json` = `max_j_occ2=64, ubatch=2047, ids_helper_smem=full` (conservative; `auto` also works on newer builds). Heat tables: `hot250_spare.json` (default) or `hot600_spare.json` (more hot experts, +capacity).

Heat-table device keys follow your `-dev` mapping (`"CUDA0": {...}` pins those expert ids onto CUDA0); regenerate with your own traffic if the tables do not match your workload.

Upstream llama.cpp documentation continues below. 上游 llama.cpp 文档见下方（中文分支说明见 [README.zh-CN.md](README.zh-CN.md)）。

---

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
