# strixhalo_5090m_llamacpp

**简体中文** | [English](README.md)

AMD Strix Halo（Ryzen AI Max 395，gfx1151）+ NVIDIA RTX 5090 单机双卡推理分支：一个 `llama-server` 进程、两个后端（CUDA 跑 attention，ROCm 跑 MoE 专家），面向 `qwen4exp` 混合架构模型（Qwen3.8-Flash-Next）与 lightning-indexer QSA 稀疏注意力。

基于上游 llama.cpp `main`（2026-08 末，#27886 之后），全部改动为增量式或带环境变量开关。上游 llama.cpp 的完整文档（英文）见 [README 英文版](README.md#llamacpp)。

## 与主线差异（22 个提交）

### 双后端执行
- `ggml-backend-meta`：镜像/切分图调度器——一张计算图按层切到 CUDA0+ROCm0（`-dev CUDA0,ROCm0 -ts 1,0`），跨后端张量接力与分片状态同步
- MoE 热专家系统：`--hot-exp-file` 热表按设备固定热门专家权重；专家 LRU 采集节流（`LLAMA_EXPERT_LRU_HARVEST_EVERY`）；counting-sort ids 流水线替换逐专家重扫；按模型加载 `mmq_settings.json`

### Strix Halo / UMA 调优
- gfx1151 rdna3-5 downblock 修复（8060S 关键性能修复）；UMA HIP 主机可寻址缓冲；`GGML_HIP_MEM_MODE` / `MEM_ADVISE` 运行时开关；`GGML_HIP_DISABLE_GRAPHS` 支持按值配置

### qwen4exp QSA 正确性与 prefill/decode 性能
- `llama-memory-hybrid-idx`：池化 indexer-key 缓存（按块增量池化、跨层共享 QSA 输入、`can_reuse` 图复用）
- `get_prev_tokens` 窗口化 n-gram 前驱扫描
- indexer head 求和改跨步分片相加（去掉整幅 permute）
- QSA **decode** gather 注意力（只读被选中行）
- **hot-block gather prefill**（本分支主打）：prefill 注意力改为每层一个共享热点窗口——对 indexer 分数做块级 top-k，加上本 chunk 自身 cells；替代整幅 `[n_tps, n_kv]` mask 展开链
- NextN/MTP draft head（`--spec-type draft-mtp`）

### 环境开关一览

| 开关 | 作用 |
|---|---|
| `LLAMA_QSA_HOTBLOCK` | 热窗口 prefill，默认开；`0` 关闭；整数=最小激活历史 cells（默认 8192） |
| `LLAMA_QSA_HOTBLOCK_CELLS` | 热窗口历史预算 cells（默认 2048） |
| `LLAMA_QSA_DISABLE` | `1` = 关闭 QSA，纯全注意力（数值对照基准） |
| `LLAMA_QSA_GATHER` | decode gather，默认 32768 起，`0` 关 |
| `LLAMA_QSA_NO_POOLED_CACHE` | `1` = 关闭池化缓存（全量重池化对照） |
| `LLAMA_QSA_HOTBLOCK_DEBUG` | `1` = 打印热窗口形状 |
| `GGML_HIP_DISABLE_GRAPHS` | ROCm 图捕获开关（按值） |
| `GGML_HIP_MEM_MODE` / `MEM_ADVISE` | UMA 内存模式运行时 A/B |
| `LLAMA_EXPERT_LRU_HARVEST_EVERY` | 专家 LRU 采集节流（默认 32） |

## 实测数据（Qwen3.8-Flash-Next-UD-IQ4_XS，APU 8060S）

| prefill t/s | 16k ctx | 32k ctx | 48k ctx | 16k→48k 衰减 |
|---|---|---|---|---|
| dense QSA（开关关闭） | 278 | 244 | 224 | -23% |
| hot-block（默认开） | 322 | 369 | **385** | **-8%** |

困惑度：2048 ctx `3.7872 ± 0.033`（基线 `3.7892`，逐 chunk 一致）；49k ctx `3.53`，对照全注意力真值 `3.88`。decode 无变化（hot-block 只动 prefill）。

## 配置与启动命令

配置在 [`configs/`](configs/) 目录：每模型 `mmq_settings.json`（放到模型同目录会被 `common_init` 自动加载）和 `--hot-exp-file` 热表。

**Qwen3.8-Flash-Next（qwen4exp），5090 + Strix Halo 双卡：**

```bash
export LD_LIBRARY_PATH=<构建目录>/bin:/nvidia-libs
export GGML_QSA_HOTBLOCK=1            # 热窗口 prefill（默认开；0 = dense QSA）

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

配置：`configs/qwen3.8-next/mmq_settings.json` = `max_j_occ2=64, ubatch=2048, ids_helper_smem=auto`。

**DeepSeek-V4-Flash（MoE），5090 + Strix Halo 双卡：**

```bash
export LD_LIBRARY_PATH=<构建目录>/bin:/nvidia-libs
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

配置：`configs/deepseek-v4-flash/mmq_settings.json` = `max_j_occ2=64, ubatch=2047, ids_helper_smem=full`（保守值；新构建也可用 `auto`）。热表：`hot250_spare.json`（默认）或 `hot600_spare.json`（更多热专家，更大容量）。

热表的设备键跟随你的 `-dev` 映射（`"CUDA0": {...}` 表示这批专家 id 钉在 CUDA0）；流量特征不同建议用自带工具重新生成。

## 构建

```bash
# 自包含双后端构建（CUDA sm_120 + ROCm gfx1151）
podman build -f Dockerfile.dual -t localhost/llamacpp:dual .
# 或走本地已验证的工具链基镜像
podman build -f Containerfile.qwen-prod -t localhost/llamacpp:qwen-prod-v6 .
# 原生 cmake
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON \
      -DGGML_CUDA=ON -DGGML_HIP=ON -DGGML_NATIVE=OFF -DGGML_CPU=x86-64-v3 \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build --target llama-server -j16
```

## 许可

本分支遵循上游 [llama.cpp 的 MIT 许可证](LICENSE)。
