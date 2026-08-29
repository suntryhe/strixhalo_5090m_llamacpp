#!/bin/bash
# LD_LIBRARY_PATH 必须含系统库路径(镜像内 /usr/local/lib64 /usr/lib64),
# 否则 llama-server 启动即 SIGSEGV(链接到错误/缺失的库)
export LD_LIBRARY_PATH=/nvidia-libs:/build-moe/bin:/usr/local/lib64:/usr/lib64:/opt/rocm/lib
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export GGML_HIP_ENABLE_UNIFIED_MEMORY=1
export GGML_CUDA_GRAPH_OPT=1
exec /build-moe/bin/llama-server \
  -m "/models/DeepSeek/DeepSeek-V4-Flash-0731-UD-IQ4_XS-00001-of-00004.gguf" \
  -dev CUDA0,ROCm0 -ngl 999 -ts 1,0 \
  -ot 'blk\.([3-9]|[1-2][0-9]|3[0-9]|4[0-2])\.ffn_.*_exps\.weight=ROCm0' \
  --hot-exp-file /build-moe/bin/hot_cfg.json \
  --host 0.0.0.0 --port 5235 -n 24000 -c 8192 -fa on -b 2047 -ub 2047 -t 2 -tb 2 \
  --parallel 1 --no-mmproj --no-mmap --metrics --temp 0.7 --top-p 0.95 --top-k 20 \
  --jinja --fit off -ctv q8_0 -ctk q8_0 --numa distribute
