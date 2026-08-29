#!/bin/bash
# Reliable deploy: build -> pick latest soname -> deploy -> restart server.
# Usage: bash tools/deploy_lru.sh
set -e
SRC_DIR=/home/suntryho/opencode/llamacpp-moe
HL=/tmp/opencode/moe-out/hotlibs-prod

echo "=== 1. build ==="
podman start moe-build >/dev/null 2>&1 || true
podman exec moe-build bash -c "cd /src-moe/build && cmake --build . --config Release --target llama-server -j\$(nproc) 2>&1 | tail -2"

echo "=== 2. pick latest libllama soname ==="
LLAMA_SO=$(podman exec moe-build bash -c 'ls /src-moe/build/bin/libllama.so.0.0.* | sort -V | tail -1')
LLAMA_VER=$(basename "$LLAMA_SO")
echo "newest: $LLAMA_VER"

echo "=== 3. deploy ==="
podman cp moe-build:"$LLAMA_SO" "$HL/$LLAMA_VER"
podman cp moe-build:/src-moe/build/bin/libllama-server-impl.so "$HL/"
podman cp moe-build:/src-moe/build/bin/llama-server "$HL/"
COMMON_SO=$(podman exec moe-build bash -c 'ls /src-moe/build/bin/libllama-common.so.0.0.* | sort -V | tail -1')
podman cp moe-build:"$COMMON_SO" "$HL/$(basename "$COMMON_SO")"

echo "=== 4. symlinks ==="
cd "$HL"
rm -f libllama.so libllama.so.0 libllama-common.so libllama-common.so.0
ln -sf "$LLAMA_VER" libllama.so
ln -sf "$LLAMA_VER" libllama.so.0
CVER=$(basename "$COMMON_SO")
ln -sf "$CVER" libllama-common.so
ln -sf "$CVER" libllama-common.so.0

echo "=== 5. verify LRU symbols ==="
strings "$HL/libllama.so.0" | grep -c "EXPERTLRU-DBG" && echo "DBG present"
echo "deploy OK -> $LLAMA_VER"