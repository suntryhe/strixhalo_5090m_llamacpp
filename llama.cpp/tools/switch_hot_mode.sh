#!/bin/bash
# Switch the running server between fixed-hot and LRU-spare hot_cfg.
# Usage:
#   bash tools/switch_hot_mode.sh fixed   -> 纯固定 200 槽 (无 spare)
#   bash tools/switch_hot_mode.sh lru     -> 200 hot + per-layer spare (LRU 缓存环)
#   bash tools/switch_hot_mode.sh 400     -> 固定 400 槽
set -e
SRC=/home/suntryho/opencode/llamacpp-moe
HL=/tmp/opencode/moe-out/hotlibs-prod
MODE="${1:-lru}"

# stop server
podman exec moe-srv bash -c 'kill $(cat /tmp/server.pid) 2>/dev/null' 2>/dev/null || true
podman stop moe-srv 2>/dev/null || true
podman rm -f moe-srv 2>/dev/null || true

case "$MODE" in
  fixed)
    git -C "$SRC" show backup-lru-pre-experiment:tools/PROGRESS/gen_cfg/hot_cfg_200_greedy.json > "$HL/hot_cfg.json"
    echo "-> fixed 200-hot (no spare)"
    ;;
  400)
    git -C "$SRC" show backup-lru-pre-experiment:tools/PROGRESS/gen_cfg/hot_cfg_400_greedy.json > "$HL/hot_cfg.json"
    echo "-> fixed 400-hot (no spare)"
    ;;
  lru|*)
    cp "$SRC/tools/PROGRESS/gen_cfg/hot_cfg_200_spare.json" "$HL/hot_cfg.json" 2>/dev/null \
      || cp /tmp/hot_cfg_200_spare.json "$HL/hot_cfg.json"
    echo "-> 200-hot + per-layer spare (LRU)"
    ;;
esac

# restart container (uses current hot_cfg)
BIN_DIR="$HL" "$SRC/tools/create-moesrv.sh" 2>/dev/null || BIN_DIR="$HL" /tmp/opencode/moe-out/create-moesrv.sh
echo "container restarted, hot_cfg = $MODE"