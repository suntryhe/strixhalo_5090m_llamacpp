#!/bin/bash
# 固化 MoE hot 卸载部署流程(重启后恢复用)
# 用法: bash tools/moe_deploy.sh <hot_cfg.json路径>
set -e
CFG="${1:-hot_cfg.json}"

echo "=== 1. 重建 hotlibs 目录 ==="
mkdir -p /tmp/opencode/moe-out/hotlibs

echo "=== 2. 从构建容器拷三件套+全部库 ==="
podman start moe-build 2>/dev/null || true
sleep 2
podman cp moe-build:/build-moe/bin/libllama.so.0.0.19 /tmp/opencode/moe-out/hotlibs/
podman cp moe-build:/build-moe/bin/libllama-common.so.0.0.19 /tmp/opencode/moe-out/hotlibs/
podman cp moe-build:/build-moe/bin/libllama-server-impl.so /tmp/opencode/moe-out/hotlibs/
podman cp moe-build:/build-moe/bin/libggml.so.0.18.0 /tmp/opencode/moe-out/hotlibs/
podman cp moe-build:/build-moe/bin/libggml-base.so.0.18.0 /tmp/opencode/moe-out/hotlibs/
podman cp moe-build:/build-moe/bin/libggml-cuda.so /tmp/opencode/moe-out/hotlibs/
podman cp moe-build:/build-moe/bin/libggml-hip.so /tmp/opencode/moe-out/hotlibs/
podman cp moe-build:/build-moe/bin/libmtmd.so.0.0.19 /tmp/opencode/moe-out/hotlibs/
podman cp moe-build:/build-moe/bin/llama-server /tmp/opencode/moe-out/hotlibs/

echo "=== 3. 补齐主版本 symlink(缺则链接器回退镜像旧库=SIGSEGV)==="
cd /tmp/opencode/moe-out/hotlibs
ln -sf libllama.so.0.0.19 libllama.so.0
ln -sf libllama.so.0.0.19 libllama.so
ln -sf libllama-common.so.0.0.19 libllama-common.so.0
ln -sf libllama-common.so.0.0.19 libllama-common.so
ln -sf libggml.so.0.18.0 libggml.so.0
ln -sf libggml.so.0.18.0 libggml.so
ln -sf libggml-base.so.0.18.0 libggml-base.so.0
ln -sf libggml-base.so.0.18.0 libggml-base.so

echo "=== 4. 拷贝 hot 配置 ==="
cp "$CFG" /tmp/opencode/moe-out/hotlibs/hot_cfg.json
chmod 755 /tmp/opencode/moe-out/hotlibs/llama-server

echo "=== 5. 重建 nvidia-libs(宿主驱动库)==="
mkdir -p /tmp/opencode/nvidia-libs
cp -L /usr/lib64/libcuda.so.580.173.02 /tmp/opencode/nvidia-libs/ 2>/dev/null || true
for f in /usr/lib64/libnvidia-*.so*; do [ -f "$f" ] && cp -L "$f" /tmp/opencode/nvidia-libs/ 2>/dev/null; done
cd /tmp/opencode/nvidia-libs
ln -sf libcuda.so.580.173.02 libcuda.so
ln -sf libcuda.so.580.173.02 libcuda.so.1

echo "=== 完成: 目录就绪 ==="
ls /tmp/opencode/moe-out/hotlibs/ | head
