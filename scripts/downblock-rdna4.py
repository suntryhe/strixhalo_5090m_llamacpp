#!/usr/bin/env python3
"""rdna4 降块移植: 将最新版 mmq-config-rdna4.cuh 中 J>=64 的 256x2x128 大块降为 128x2x64
规则完全对齐生产树 (07-22 + 调优): 所有 J in (64, 80, 96, 112, 128) 一律降半块, 无例外.
128 内块: GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0 / Q8_1 保持原值.
"""
import re, sys

path = sys.argv[1]
with open(path) as f:
    lines = f.readlines()

changed = 0
out = []
for line in lines:
    m = re.match(r'^(\s*CASE\(GGML_TYPE_[A-Z0-9_]+, )256, 2, 128, ( 64| 80| 96|112|128)(,.*)$', line)
    if m:
        # J>=64 大块 → 降为 128x2x64 (对齐生产树)
        new = f"{m.group(1)}128, 2, 64,{m.group(2)}{m.group(3)}\n"
        out.append(new)
        changed += 1
    else:
        out.append(line)

with open(path, 'w') as f:
    f.writelines(out)
print(f"✅ 降块行数: {changed} (J>=64 的 256 大块全部转为 128x2x64)")