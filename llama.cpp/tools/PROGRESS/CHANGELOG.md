# 改动手册: MoE hot-expert 卸载全部改动点 (2026-08-22)

> 重要: /tmp 是 tmpfs, 重启即丢. 本文件放项目目录 tools/PROGRESS/, 与 PROJECT.md 配套.

## 现状总览 (2026-08-22 10:45, 恢复中)

- hot_cfg.json: 40层 x 10 = 400 槽 (blk.3-42), 源码固化
- 生产参数: -c 8192 -b 2047 -ub 2047 -ot 'blk\.([3-9]|[1-2][0-9]|3[0-9]|4[0-2])\.ffn_.*_exps\.weight=ROCm0'
- 生产容器: localhost/llamacpp:moe-hotstats-prof, 脚本 tools/moe_run_server.sh
- 库目录: /tmp/opencode/moe-out/hotlibs/ (bind-mount 到容器 /build-moe/bin)

## 一、ROCm 侧提速改动 (grq. ggml-cuda/mmq.cu) - ROCm 用

### 1. 网格裁剪 (PROJECT.md §39) - ROCm mul_mat_q -3.1%
- 文件: ggml/src/ggml-cuda/mmq.cu + mmq.cuh
- 内容: mmq grid z 从固定 256 -> n_nonempty(实际非空专家数), 加 expert_map 映射
- 开关: GGML_MMQ_NO_TRIM=1 关闭裁剪
- 实测: NOTRIM 2514.9ms vs TRIM 2438.1ms => -3.1%
- 附带: 双 pool 分配 bug 修复

### 2. zero_neg1_slots 优化 (PROJECT.md §40) - ROCm 105x
- 文件: ggml/src/ggml-cuda/mmq.cu
- 内容: cold 链跳过槽位(-1)时清零逻辑, 初版每线程串行清整列(32x 写放大)
        -> 2D grid + float4 合并写
- 实测: 单次 0.781ms -> 0.0074ms (105x), 总 182.7ms -> 2.6ms (-98.6%)

### 3. ROCm aperture 崩溃修复 (PROJECT.md §35) - 必须保留
- 文件: ggml/src/ggml-cuda/mmq.cu (cudaMemsetAsync(ids_src1,0) @ ~209)
- 文件: ggml/src/ggml-cuda/quantize.cu (3处 if(i01<0) return)
- 作用: 防 cold chain -1 槽位时访问非法内存

## 二、N 卡侧提速改动 (CUDA0 用)

### 4. topk 批量优化 (PROJECT.md §43) - CUDA0 +16.3%
- 文件: ggml/src/ggml-cuda/top-k.cu
- 内容: nrows==1 保留 cub::DeviceTopK; nrows>1 走批量分段 argsort(desc)+memcpy2D 取前k
- 原因: 原 for 循环逐行启动 DeviceTopK = 51.7万次 / 1.14s (22.5% CUDA0)
- 实测: 517,188次 -> 282次, 1137ms -> 11.7ms, pps 410.3 -> 477.1
- patch 备份: /tmp/opencode/topk_batch.patch

## 三、hot 机制核心 (src/llama-graph.cpp)

### 5. hot/cold 双链构建 (b6ac594 起的多 commit)
- hot 链在 CUDA0, cold 链在 ROCm0; 每层 hot 专家走 CUDA0, 其余走 ROCm0
- 文件: src/llama-graph.cpp build_moe_ffn 中
- 关键机制:
  - remap/flag 表: [n_expert] I32/F32 静态表, 标记 hot 专家
  - hot_ids = get_rows(remap_3d, selected_experts) -> hot 链用的 id (n_hot 槽映射)
  - cold_ids = get_rows(cold_remap_3d, selected_experts) -> cold 链用 (-1 跳过 hot 槽)
  - cold_boundary (host 缓冲): cur/weights/ids/sel 复制到 CPU host 缓冲, ROCm UMA zero-copy 读
  - gf_cold: GGML_MOE_DUAL=1 时 cold 链进独立图 (已否决竞态, 生产不用)
  - moe_host: cold 聚合结果复制到 host, 主图读回与 hot_sum 合并 (cold_moe_out)
- 文件: src/llama-graph.h (llama_graph_moe_hot 结构: up_exps/gate_exps/down_exps/remap/flag/cold_remap/cold_skip)
- 文件: src/llama-model.cpp (setup_moe_hot_exps: 从 hot_cfg.json 读每层 hot 专家, 拷权重到 CUDA0, 建 remap/flag/cold_remap)
- 文件: src/llama-context.cpp+h (llm_graph_result: cold_host_bufs 生命周期, gf_cold, moe_host 管理等)

### 6. **重要: 冷列压缩已废弃**
- up_exps_cold/gate_exps_cold/down_exps_cold (compact cold 张量) 是早期实验, 已删除
- cold_remap 现在 = -1 for hot experts, **original id for cold** (不是 compact index)
- 模型只生成 cold_remap(原始id) + up_exps/gate_exps/down_exps(完整专家), 不再建 *_cold 紧凑列
- 恢复代码时不要重新引入 *_cold 成员

## 四、本次传递分析 (2026-08-22, 进行中, 尚未 commit)

### 7. 修复#1: 移除冗余 ggml_cont(experts_hot) <<< 已验证有效, 必须保留
- 文件: src/llama-graph.cpp (build_moe_ffn hot 链聚合处 ~2347 行)
- 改动: `ggml_tensor * hot_experts = experts_hot;` (原为 ggml_cont(ctx0, experts_hot))
- 原因: experts_hot 是 ggml_mul 连续输出, view_2d 用显式 stride 切片即可,
        ggml_cont 无连续短路 => 每层每批复制整个 [n_embd, n_expert_used, n_tokens]
        201MB x 120次 = 24GB D2D!
- 实测: DTOD 26.4GB -> 2.2GB (120次201MB 全消除), SHA1 0ca676b1 稳定, pps 480.3
- 备份: /tmp/opencode/libggml-cuda_batched.so 无关; 源码改在 llama-graph.cpp

### 8. 修复#2: remap/flag 表钉扎 CUDA0 <<< 已尝试但崩溃, 已回退
- 尝试: 把 remap_3d/flag_3d/hot_ids_raw 一起 ggml_backend_sched_set_tensor_backend 到 CUDA0
- 结果: SIGSEGV in ggml_cuda_op_repeat (Exited 139) - repeat 的输入是 weights buffer 不在 CUDA0?
- 结论: 不能直接钉扎 repeat 输出; 未提交; 若重试需先查 repeat 输入后端
- 收益预期: 2.1MB x 242次 HTOD = 70.6ms (0.5%) - 小额

### 9. 传递分析核心结论 (给未来自己)
- 两卡不同厂商(NVIDIA+AMD)无 P2P, 跨卡数据必须经 host 中转 (H2D/D2H)
- hot400 prefill 期 memcpy: HTOD 1.26k次/658ms / DTOH 2.07k次/576ms / DTOD 2.2GB(修复#1后)
- cur 激活 33.5MB x 123 次是必传(冷链 ROCm 要读完整激活), hot 不减少这部分
- 真正的"hot 减少传递"要等冷链权重真分拆(修复#3: 按 hot 配置裁剪权重, 未做)

## 五、git 恢复记录 (2026-08-22 10:40)

- 事故: 回滚修复#2 时误用 `git checkout -- src/llama-graph.cpp` => 整个文件回滚到 HEAD(丢未提交 hot 机制)
- 恢复: 从 dangling stash `3495349a` (WIP on moe-hotstats) 取回 llama-graph.cpp
  - `git checkout 3495349a -- src/llama-graph.cpp`
- 恢复后需清理: 删除 up_exps_cold 旧实验块(3行 if + 注释, 已做, 见备份 /tmp/opencode/llama-graph_3495349a.cpp.bak)
- 备份: /tmp/opencode/llama-graph_3495349a.cpp.bak
- **教训: 每改一个改动立刻 git add + commit, 不要堆积工作区改动!**

## 六、验证流程 (每次改动后必做)

1. SHA1 一致性: 短请求 3 次, 期望 0ca676b1 (hot400 基线)
2. prefill pps: 6156-token 请求, 与基线 480.3 (修复#1后) 对比
3. 长 prefill 零崩溃
4. 如需 prof: run_nsys_prof.sh (CUDA0) / prof_cold_8k.sh (ROCm)
## 七、2026-08-22 恢复后验证结果(已 commit a56a572)

- 恢复验证全部通过:
  - SHA1 3x = 0ca676b1 (数学一致)
  - 6156-token prefill pps = 480.5 (修复#1 基线 480.3, 一致)
  - N 卡 nsys: DTOD 2.2GB/3.2ms (修复#1 前 26.4GB/62.9ms), 120次201MB 大拷贝 = 0
  - ROCm rocprof: 7168ms (修复#1 前 hot400 7096ms, nohot 7729ms; 恢复版仍比 nohot 快 -7%)
- 结论: 修复#1(去 ggml_cont)效果 100% 保留, 恢复无回归

## 八、当前传递状况总结 (给未来自己)

| 项 | 修复#1前 hot400 | 修复#1后 hot400 | nohot |
|---|---|---|---|
| DTOD | 26.4GB/62.9ms | **2.2GB/3.2ms** | 2.2GB/3.2ms |
| HTOD | 1263次/658ms | 1263次/658ms | 383次/588ms |
| DTOH | 2071次/576ms | 2071次/576ms | 991次/607ms |

- hot vs nohot 残余差异 = HTOD +880次 / DTOH +1080次 (小张量, 见修复#2 方向)
- cur 激活 33.5MB x 123 次 = 必传, 两版相同, hot 不减少 → 修复#3 方向

## 九、修复#2 评估定论 (2026-08-22)

- 尝试: 把 remap_3d/flag_3d/hot_ids_raw 一起钉扎 CUDA0 => SIGSEGV (ggml_cuda_op_repeat)
- 根因(已确认): remap/cold_remap 是 I32; CUDA 的 ggml_cuda_op_repeat -> bin_bcast
  硬约束 GGML_ASSERT(src1->type == F32 || F16)。I32 repeat 只有 CPU 后端支持。
  钉扎后 repeat 被调度到 CUDA -> assert 失败 -> 崩。
- 钉扎前 repeat 跑 CPU(host) -> 输出 host 张量 -> HTOD 上传 = 那 242 次 2.1MB 传输(70.6ms)的来源
- 定论: 方案#2(钉扎)不可行。要省这 70ms 需在模型加载时把 remap/flag 的
  [1, n_expert, max_ubatch] 表预物化为 CUDA0 weights 张量, 逐批 view 切片
  (中等改动, 有正确性风险). 收益 70ms/0.5%, 暂缓.

## 十、修复#3 方案设计定论 (2026-08-22)

### 核心数据(决定性)
- cur 激活往返(30-40MB): 243次/1143.8ms/8.1GB = 传递总量 93%
- 权重级(1-5MB): 266次/80ms/575MB
- 小张量(<1MB): 2825次/10.6ms/78MB
- 传递总耗时 1234ms (prefill 13124ms 的 9.4%)
- 关键: nohot 与 hot400 的 cur 激活传递完全相同 (243次/8.1GB/1.14s)

### 结论: 修复#3(冷链权重真分拆)方案被数据否决
- 跨卡传递 93% 是 cur 激活往返, 不是专家权重
- 冷专家权重静态驻留 ROCm 显存, 不参与每层传递
- 权重真分拆只能减 ROCm 计算(已减 8%), 不减传递 => 无法实现"hot 让传递大幅减少"

### 新方向(待用户确认后实施)
- A. 整层全 hot (用已有 cold_skip): 全 hot 层不建 cold 链 => 该层完全不传 cur 上 ROCm
     每覆盖一层省约 1.1s/43层 的份额; 受 CUDA0 显存限制 (22.2/24GB 近满)
- B. 硬件限制: 两卡无 P2P 只能走 host, 无法优化
- C. 层间融合减少 cur 往返次数: 图重构大工程
- 建议: 先做 A 的可行性测算(整层 hot 显存需求 vs CUDA0 剩余)

## 十一、等待气泡分析 + 微批流水线方案 (2026-08-22)

### 气泡量化(prof 实证)
- 本地层 blk.0-2 (FFN 权重留 CUDA0): 层周期 43ms, CUDA0 忙 36-40ms (利用率 92%), 几乎无气泡
- 跨卡层 blk.3-42 (FFN 权重在 ROCm0): 层周期 100ms, CUDA0 忙 26ms (利用率 26%), 气泡 44ms/层
- ROCm 一层 FFN = 3 个 mul_mat_q 共 46ms 忙, span 98ms (利用率 55%)
- 气泡总计 43层x3批x44ms ≈ 5.6s = prefill 13124ms 的 43% <- 当前最大未优化空间
- 根因: 层间串行接力 (attention CUDA -> cur 传 -> FFN ROCm -> 结果回传 -> 下一层), 两卡无法跨层并行, CUDA0 等 ROCm 干等

### 零拷贝确认(已核实, 生产已开)
- GGML_HIP_ENABLE_UNIFIED_MEMORY=1 在 run_top12.sh; HIP 后端 cudaMallocManaged
  + hipMemAdviseSetCoarseGrain/PreferredLocation/AccessedBy 全生效
- cold_boundary host 缓冲 = ROCm UMA zero-copy 直读 (APU 优势点)
- 带宽余量巨大: 全程传递 2.9GB/13.1s = 0.2GB/s, PCIe 4.0 x16 上限 25GB/s, 只占 0.8%

### 微批流水线方案(待用户确认后实施)
- 思路: 6156-token 拆 2 微批; ROCm 算微批1 的 FFN(46ms)时 CUDA 同时算微批2 的 attention
  -> 填满 CUDA0 空闲, 每层从 100ms 压向 ~56ms
- 带宽: 总量不变(切分非复制), 瞬时峰值 ~1.5GB/s << 25GB/s 上限, 非瓶颈
- 预估: 13.1s -> ~8s (省 40%+)
- 待办: 具体代码改动方案(改 ubatch 调度)需先出方案给用户再实施

## 十二、微批流水线校准 (2026-08-22)

### 原始方案(已推翻)
- 拆 2 微批 -> CUDA 算 B 的 attn 时 ROCm 算 A 的 FFN -> 预估省 40%

### 勘察发现两个结构性硬约束 -> 方案不成立
1. 层间依赖(残差流): attn(l+1) 输入 = ffn(l) 输出 + 残差
   => ffn(l) 没完成, attn(l+1) 不能开始, 两卡计算严格串行
2. 同序列拆批依赖(因果 KV): B 的 attention 需要 A 的所有 KV
   => 同一序列的微批在 attention 层不独立
=> 单序列 prefill 下, 每层 100ms 是结构性串行
=> 这也解释了 §42 PP(层切分)负收益 293 vs 327

### 精确校准后的每层构成 (blk.3-42 跨卡层)
| 环节 | 耗时 | 说明 |
|------|------|------|
| CUDA attention | 26ms | 必需, 计算 |
| ROCm FFN (3x mul_mat_q) | 38ms | 必需, 计算(dur 12.6+12.8+12.3) |
| 传递 (cur上+结果回) | 10ms | host 中转带宽 |
| 同步/调度 | ~26ms | 纯开销, 可优化 |
| 合计 | ~100ms | |

### 真正可优化 = 26ms/层 同步开销
- 43层 x 3批 x 26ms ≈ 3.4s (占总 26%)
- 若压到 10ms: 每层 84ms, 总 10.8s, 省 ~18%
- 方向: 减少 sched 跨后端同步次数 / 传递与上一层计算重叠
- 注意: 非微批流水线(那需要多序列才有效), 是同步开销优化

### 结论
- 微批流水线(同序列) 不实施
- 新方向: 压每层 ~26ms 同步开销(收益 ~18%, 中等)
- 需用户确认后再动

## 十三、同步开销深度分析 (2026-08-22)

### 5568 次 cudaStreamSynchronize 分类(全 prefill 13124ms)
| 类型 | 次数 | 占比 | 性质 |
|------|------|------|------|
| sync 后<0.1ms 有 kernel | 1725 | 31% | 纯检查(数据已就绪), 理论可消除 |
| sync 后>1ms 无 kernel | 2785 | 50% | 等 ROCm(必要等待) |
| 中间态 | 1058 | 19% | - |

### 关键区分
- "同步开销" 26ms/层 中: 约 31% 是纯 CPU 检查(sync 不等待), ~50% 是 CUDA 等 ROCm 计算(本质是计算等待, 非可消除开销)
- 真正可省的纯检查类: 1725 次, 若消除可省 ~1.1s (8.4%)
- 剩余大部分是结构性串行(依赖链决定), 无法通过同步优化解决

### 诚实结论
- 微批流水线(同序列): 不成立(依赖链硬约束)
- 每层 26ms "同步开销" 只有 ~31% 是纯检查可优化(约1.1s / 8%)
- 大头(50%等ROCm) 是真实计算等待, 消除不了
- 投入产出比低: 需要动 sched 核心同步逻辑, 收益 ~8%, 风险高
- 建议: 微批流水线与同步优化均不推进; 当前优化(修复#1+topk+网格裁剪+zero_neg1)已是高价值项

## 十四、F16 激活支持 (2026-08-22, 最终: 传输减半 50% 验证通过)

### 关键教训: 架构路径错位
- DeepSeek-V4-Flash 走 LLM_ARCH_DEEPSEEK4 -> src/models/deepseek4.cpp
- 最初的 cast 加在 src/models/deepseek.cpp (V2/V3 架构) -> 从未执行, 传输无变化
- 正确位置: deepseek4.cpp 的 ffn_norm 之后、build_moe_ffn 之前

### 实现 (5 处改动, 全部验证通过)
1. src/models/deepseek4.cpp: `cur = ggml_cast(ctx0, cur, GGML_TYPE_F16)` + F16CAST 诊断日志
2. ggml/src/ggml-cuda/ggml-cuda.cu supports_op: 放行 F16 激活 x F32/BF16/量化权重
   (原代码: `if (b->type == F16 && a->type != F16) return false` 拒绝所有 F16 激活)
3. ggml/src/ggml-cuda/ggml-cuda.cu ggml_cuda_mul_mat_id: 断言放宽 F32|F16 + 分发条件放行 F16
4. ggml/src/ggml-cuda/mmq.cu: src1 为 F16 时 host 侧用 to_fp32_cuda 转 F32 再进 DP4A
5. ggml/src/ggml-cuda/mmvq.cu: 同上 (decode 路径)

### 崩溃修复链 (4 处, 每处实测定位)
- 崩溃1: NOBACKEND op=29(MUL_MAT) ffn_moe_logits src0.type=30(BF16) src1.type=1(F16)
  -> supports_op 只放行了 F32+量化, BF16 权重被拒 -> 加 BF16 放行
- 崩溃2: ggml-cuda.cu:1948 GGML_ASSERT(src1->type == F32) in ggml_cuda_mul_mat_id
  -> 入口断言 + 分发条件(1965行)放行 F16
- 崩溃3: common.cuh:1197 GGML_ASSERT(pool != nullptr)
  -> mmq/mmvq 中 src1_f32_alloc 默认构造后 .alloc() 传 null pool -> 改 alloc(ctx.pool(), n)

### 传输验证结论 (nsys A/B, 同 workload 5948+200, F32 基线 vs F16 支持版)
| 指标 | F32 基线 | F16 支持版 | 变化 |
|------|---------|-----------|------|
| D2H 单次最大 | 33.538MB (F32 cur) | 16.769MB (F16 cur) | **-50%** |
| D2H 总量 | 9001.8MB | 6976.8MB | -2025MB |
| 16.77MB(F16)拷贝 | 0 次 | 40 次 (PINNED->DEV, 对应40层) | 新出现 |
| SHA1 同prompt 3x | b3ad8f2c | b3ad8f2c | 完全一致 |
| F16CAST 日志 | - | 430 次 (cur type=1 F16, nbytes=8192) | cast 生效 |

### 结论 (最终版 2026-08-22 晚, 原生 __half 方案)
- F16 激活在 DeepSeek-V4-Flash 上**可行且已部署**: 跨卡传输减半 50%
- 数值一致 (SHA1 b3ad8f2c 3x 相同)
- **最终实现(非本节省略的 host 转 F32)**: quantize_q8_1 模板化原生读 __half
  (quantize.cu: `quantize_q8_1<__half>` + `__half2float`), mmq/mmvq 零 host 转换
- 重启后干净环境重测: F16 nohot decode 26.3 t/s, 与 F32 持平 (掉速是环境噪声)
- 局限: 33.5MB 剩余拷贝为权重传输(UNK->PINNED 41次), 与激活无关, 不减

### 部署状态
- hotlibs 当前为 F16 原生支持版 + cold_ids pin (libllama.so.0.0.26, 2026-08-23 09:10 构建)
- 诊断代码 (SPLITDUMP/MOESPLITS/NOBACKEND) 已全部移除, 生产无噪音日志
- 生产 server 运行中: 端口 5235, decode 22-25 t/s (RAM 压力下), 短输出 SHA1 确定性验证通过
- git: 10 个文件改动 (6 个 F16 实现 + 错误修复/断言清理 + 文档), 已 stage, 未 commit

## 十五、decode 真瓶颈深挖 (2026-08-23, 重启后干净环境重测)

### 背景修正: 重启前"F16掉速"全是环境噪声
- 重启前测到的 hot 8-24 t/s 波动 = 内存耗尽(124GB RAM < 138GB 模型)+ 库混搭错乱
- 重启后干净环境重测: F16 原生 nosync-hot 26.3 t/s, 与 F32 基线持平
- 关键: decode 全程换页增量极小(10 次 decode 仅 pswpin+666) -> 模型常驻, 内存不是 decode 瓶颈

### 决定性对比: hot(40层专家上CUDA) vs nohot(全ROCm)
| 配置 | decode t/s |
|------|-----------|
| hot (hot_cfg 400槽, 40层x10专家上CUDA) | 24.5 |
| nohot (全专家在ROCm) | 26.3 |

**hot 反而更慢!** 每层 ffn 需跨卡合并(mul_mat_id 结果 CUDA+ROCm 各算一部分再合并),
CUDA 8x 算力省下的计算时间被每层一次跨卡合并同步抵消。=> 逐层切专家是负优化。

### nsys 精确账本 (decode 950ms/20tok = 47.5ms/token, 此为 prof 期间含开销)
| 项目 | ms/token | 占比 | 性质 |
|------|---------|------|------|
| CUDA kernel 执行 | 3.3 | 7% | 几乎空闲 |
| 同步等待-中间态(0.1-1ms) | 13.8 | 29% | 等 ROCm 完成 |
| 同步-纯检查(<0.1ms) | 3.7 | 8% | 871次/token, 理论可消 |
| 同步-真等待(>1ms) | 1.4 | 3% | 罕见大等候 |
| memcpy | 0.6 | 1% | 423次/token 小拷贝 |
| ROCm侧kernel+host | 24.7 | 52% | 不在CUDA视野, 主嫌疑 |

### 实时双卡利用率 (decode 期间)
- CUDA 峰值 46%, ROCm 峰值 46%, 且互补(一张忙另一张闲)
- 无任何卡饱和 => 串行往返等待主导, 不是单卡计算算不动

### 最终结论
1. decode 瓶颈 = **每 token 908 次同步 + 423 次拷贝的海量小粒度跨卡往返 + ROCm 侧冷 FFN 串行计算**
2. 不是内存不足(换页极小), 不是 CUDA 算力(仅 7% 忙碌), 不是 ROCm 计算饱和(46%)
3. hot 机制在 decode 是负优化(合并同步 > 上 CUDA 的算力收益), nohot 更快
4. 40-50 t/s 目标需要减少跨卡往返次数(而非字节), 或让 ROCm 单侧管线化

### ROCm 侧直接测量 (rocprofv3 launch, 60token decode @ 25.3t/s = 39.5ms/token)
| 项目 | 值 | 说明 |
|------|-----|------|
| ROCm kernel 总耗时 | 15.8 ms/token | 40% 墙钟 |
| mul_mat_vec_q (Q8_0, type21) | 8.7 ms/token | 冷FFN专家 Q8_0 计算 |
| mul_mat_vec_q (type39) | 4.8 ms/token | 第二类量化专家 |
| mul_mat_vec_q_moe (type21/39) | 0.76 ms/token | MoE 专用 |
| quantize_q8_1<__half> | 0.19 ms/token | F16 原生量化(减半生效) |
| ROCm 忙碌占比(decode窗口) | 37.8% | 未饱和 |

### 完整时间账本汇总 (39.5ms/token @ 25.3t/s)
- ROCm kernel 计算: 15.8ms (40%)  <- 冷FFN专家 mul_mat_vec_q 主导
- CUDA kernel 计算: ~3.3ms (8%)
- 同步等待+传输+host调度: ~20ms (52%)  <- 海量小往返(908同步/423拷贝每token)
- 双卡均在 decode 中不足 50% 忙碌 => 瓶颈 = 串行跨卡往返开销, 不是单卡算不动

### 修改建议 (待用户决策)
1. 减少跨卡往返次数: 每token 908次同步是极端值, 合并小拷贝/减少调度切换点
2. hot 机制 decode 应关闭 (nohot 26.3 > hot 24.5)
3. 40-50 t/s 目标需先解决往返开销, 再考虑 ROCm 专家计算优化

### 决定性因果链 (2026-08-23 补)
- decode 每 token: 908 次同步 x ~21us/次 = 18.9ms/token 同步开销
- 计算量本身: ROCm 15.8ms + CUDA 3.3ms = 19.1ms/token
- **同步开销 18.9ms 几乎等于全部计算 19.1ms!** 消除拉平 -> 理论 ~50 t/s (恰为目标)
- 同步被推到每 op 粒度(908次), 而非仅层边界(理论 ~86次) -> 10x 冗余
- 真正瓶颈 = **同步次数过多**(op 级跨卡分界), 不是计算、不是内存、不是单卡饱和
- 修复方向: 把跨卡同步从每 op 一次提升到每层一次 (908 -> ~86), 即调优 sched 的同步粒度

### 图优化组合 (2026-08-23 用户确认)
- `GGML_CUDA_GRAPH_OPT=1` (CUDA 图开): NVIDIA 侧图捕获, prefill/decode 都受益
- `GGML_HIP_DISABLE_GRAPHS=1` (HIP 图关): ROCm 侧**禁用**图, 用户实测 395-token prefill 和 decode 都变快
  - 代码: common.cuh:1257 is_enabled() 在 HIP 后端独立读该 env (自定义语义, 见 1258-1263 注释)
  - 原因: ROCm 图重建/更新开销 > 重放收益 (与 CUDA 图同变量冲突, 故分开控制)
- 当前生产 server 已带此最优组合 (environ 确认)
- **验证了用户判断: 同步/管理层开销 > 计算本身**, 与十五节结论一致

## 十六、hot/cold split 结构实证与修复 (2026-08-23)

### 决定性修复: CUDA REPEAT 支持 I32 (每层 7 段 -> 3 段)
- 根因(已确认): hot/cold 的 remap 表是 I32; ggml_cuda_op_repeat -> bin_bcast 的
  GGML_ASSERT(src1->type == F32 || F16) 只放行 F32/F16。I32 repeat 只有 CPU 后端支持
  -> remap 表被迫丢 CPU 执行 -> 每层 2 个 CPU 中转段 (SPLIT 2/4)
- 修复两处:
  1. binbcast.cu: ggml_cuda_op_bin_bcast 断言扩展 `|| src1->type == GGML_TYPE_I32`,
     增加 I32 分派分支 (I32 REPEAT = 纯数据复制, 对 -1..255 索引无损)
  2. ggml-cuda.cu: supports_op 的 GGML_OP_REPEAT 放行 I32 (否则钉扎后仍回退 CPU)
- 效果 (SPLITDUMP 实测): 每层 7 段 -> **3 段** (CUDA725 -> ROCm19 -> CUDA296),
  2 次跨卡同步, 与用户理论 "hot cold 应该 2 次跨卡" 完全对齐
- rocprofv3 实证: hot 的 ROCm 冷 FFN kernel **15.76 -> 13.96 ms/token** (省 1.8ms), 
  = 用户理论 "hot 减少 ffn 计算" 的直接证实
- 测速 (干净内存, 200 token x8): hot **23.8 -> 25.0-25.6 t/s** (run2-8, run1 预热 20.8)
- 数值正确性: 短输出 5 token 3x SHA1 一致 (2458c760b6c5) = REPEAT 数据复制无损

### 剩余差距与未决问题
- hot (25.0-25.6) 仍略低于 nohot baseline (25.8-25.9), 差距 ~0.5 t/s
- ROCm 少算的 1.8ms/token 未全部转化为提速 -> 被其余开销抵消 (hot 链 CUDA 侧
  remap 成本 + 同步次数不变), 待后续管线化方向

### SPLITDUMP 诊断(单图模式, GGML_SCHED_DUMP_SPLITS=1)
decode 单 token 图 = **242 splits**, 每层 blk.N 结构:
```
SPLIT 1  CUDA0  695 nodes [hc_init ... ffn_hot_remap]    <- attn + hot 专家链
SPLIT 2  CPU     1 node                                  <- 中转
SPLIT 3  CUDA0   26 nodes [ffn_cold_remap]               <- cold 路由表 prep
SPLIT 4  CPU     1 node                                  <- 中转
SPLIT 5  CUDA0    2 nodes [ffn_moe_cold_ids]             <- cold ids 计算
SPLIT 6  ROCm0   21 nodes [ffn_moe_gate ...]             <- 真正冷专家计算
SPLIT 7  CUDA0  266 nodes [ffn_moe_out ... 下一层]        <- 结果回收
```
- 每层 6 段 x 40 层 = 240 splits, 理论应 ~86 (每层 2 段)
- 真正在 ROCm0 的计算仅 ~21 nodes/层, 其余 ~990 nodes 全在 CUDA0/CPU

### 修复实验 (cold_ids pin 到 ROCm0)
- 改动: build_moe_ffn 中 cold_ids_raw set_tensor_backend 到冷设备
- 效果: n_splits 242 -> **202** (证实 split 结构可压缩)
- **但 decode t/s 不变 (24.2 vs 基线 24.5) = 负结果**
- 结论: **split 数不是 decode 瓶颈**; 冷链 prep 只有 26+2 nodes, 压缩它们无实质收益

### 真实瓶颈定位 (rocprofv3 直接测量, 前序)
- ROCm 冷 FFN kernel: **15.8 ms/token (40%)** <- 铁堆, 与 split 数无关
- CUDA kernel: 3.3ms
- 同步等待: 18.9ms (其中中间态 13.8ms = 等 ROCm 完成, 必要等待)
- 每 token 908 次同步 / 423 次拷贝: 是症状不是病根

### 用户理论验证结论
- "hot cold 应该 2 次跨卡, 超 2 次是设计缺陷": **结构上成立**(每层 2 段是可能的), 
  但 sched 产生 6 段是 ggml 行为, 压缩后无速度收益
- "更多 hot 理论上更快 (ffn 计算变少, CUDA 8x 快)": **方向上成立**, 但受限于:
  - CUDA0 24GB 已用 22GB, 每专家 ~11.7MB, 最多再放 ~170 个
  - hot_cfg 当前 400 槽 (每层 10/256 = 4%), 提升空间被 VRAM 卡死
- **铁的事实: 40% 时间是 ROCm 冷 FFN 计算, 只有减少冷专家数或加快 ROCm kernel 才能破**

### 已排除的死路 (每条都有实测)
1. n_copies=1 时单 events -> 波动 18-24 t/s, 回退
2. GGML_MOE_ASYNC 双sched -> SHA1 不一致 (真竞态), 回退
3. GGML_MOE_DUAL 双sched -> 21-23 t/s 更慢, 回退
4. cold_ids pin -> splits 242->202 但无速度变化, 保留(无副作用)

### 诚实结论
decode 40-50 t/s 目标的主要障碍 = **ROCm0 冷专家 mul_mat_vec_q kernel 速度** (15.8ms/token),
不是传输字节 (F16 已减半), 不是 split 结构 (已验证), 不是内存 (换页极小)。
下一步应聚焦: (a) 增加 hot 覆盖到 VRAM 极限 (b) 优化 ROCm 冷 FFN kernel。

## 二十、200 槽 greedy 配置稳定性验证 (2026-08-24)

### 背景
用户目标从 400 槽改为 200 槽(先验证稳定性,后续加卡可扩到 300/400)。
100/200/300/400 各档配置已按"逐层命中率 + 全局抢槽"规则生成(gen_cfg/)。

### 200 槽 greedy 配置
- 总槽 200,40 层(blk.3-42),每层 1-8 个专家(按采样命中率分配)
- 预期命中率 ~26%(200 槽)vs 400 槽 ~37%

### 实测(干净启动,日志关闭,200 槽 hot_cfg)
- decode(warmup 后): 25.75 / 25.85 / 26.19 t/s
- 与 400 槽(25.8-26.6)持平
- 温度 0 同 prompt 两次输出内容一致(确定性无回归)
- 注意:DeepSeek-V4-Flash 是思维链模型,max_tokens 小时只输出 reasoning_content
  导致 content 为空,这是模型行为非配置 bug(需 max_tokens 足够大才能看到最终回答)

### 结论
200 槽稳定,与 400 槽同速。显存更省(200×6.3MB vs 400×6.3MB ≈ 省 1.3GB),KV 余量更多。
下一步:LRU 专家缓存环(见十九节讨论 + lucebox 分析)——不增加显存的前提下把动态命中率
从 37% 提升到 90%+。

## 二十一、LRU 专家缓存环 v1 (2026-08-24)

### 设计(参考 lucebox Spark 三层)
- 静态热集(200槽 greedy) + LRU spare 环(K/层) + 请求边界频率换载
- 关键洞察: 固定400槽命中率37%天花板 -> 靠 spare 环动态换入提升
- 运行期可改: remap/flag/cold_remap 三张表(get_rows 每次执行读当前内容)
- hot 张量布局: [n_hot pinned][K spare][1 mask], ne[2] = n_hot+K+1

### 实现
- llama-model.cpp: ne2_hot 超配 + remap默认到新mask位 + spare列零初始化
- llama-expert-lru.{h,cpp}: 新增 LRU 管理器(计数阈值+配额+LRU逐出)
- llama-graph.cpp: LRU 时强制建 selected_experts snap
- llama-context.cpp: decode 记账(纯计数) + 每APPLY_EVERY批应用 swap
- env: LLAMA_EXPERT_LRU_SLOTS_PER_LAYER / _APPLY_EVERY / _MAX_SWAP_PER_BATCH

### 实测教训(重要)
1. 每token同步换入: 2.39 t/s (灾难) - 跨卡拷贝阻塞decode
2. 每8token批量全换: 2.29 t/s - 批量换入量太大(几十个专家x19MB同步)
3. 纯统计不换(APPLY=10000): 24.7 t/s - 统计开销仅~1t/s(选中ids读取免费)
4. **配额控制8/批: 23.7-24.0 t/s** - 换入变稀疏, 不阻塞 ← 当前状态

### 等待优化(下一步)
- 命中率验证: LRU换入专家是否后续被命中(hotstats对比冷失误)
- 异步换入: 独立流+event, 换入与下token计算重叠 (23.7 -> 25+)
- 调参: K/APPLY_EVERY/MAX_SWAP 最优组合

### LRU 实测命中统计 (K=2, APPLY_EVERY=8, 配额8) - 决定性负结果
```
EXPERTLRU sel=1488 pinned=19.4% sparehit=0.7% cold=80.2% swaps=984 totalhit=19.9%
```
- **swaps 疯狂增长 (760->984 持续换入) 但 sparehit 只有 0.6%**
- 换入的专家后续几乎从不被再次选中 -> 路由无时间局部性
- 与方向A证伪(43.3%一致率)同根因: DeepSeek-V4-Flash 路由高度分散
- 结论: LRU 反应式换入在此模型上无效 - 换入一次后再次命中概率 <1%
  验证了 lucebox 数据 "predictor prefetch recall 上限 ~53%" 背后的路由分散性
- 速度 22-23 t/s (受 snap 读取 + swap 开销拖累, 比基线 25.8 低 ~10%)
- **变更**: 恢复无 LRU 部署 (纯 200 槽 greedy = 25.8-26.2 t/s 稳定基线)

### 技术资产 (本次实验沉淀)
- llama-expert-lru.{h,cpp}: LRU 管理器(计数阈值+配额+LRU逐出+命中统计)
- tools/deploy_lru.sh: 可靠部署脚本(自动选最新 soname, 解决 .27/.28/.29 递增坑)
- 教训: CMake 每次重编 soname 递增 -> 部署必须按最新 soname 拷贝, 不可写死

### WR-LFU+G (速率门控 LFU + 保护期) 改进 - 防抖生效
参数: K=2, APPLY_EVERY=8, MAX_SWAP=8, GRACE_TICKS=32, WINDOW_TOKENS=2048
```
EXPERTLRU sel=225216 pinned=24.6% sparehit=3.5% cold=71.9% swaps=80 totalhit=28.1%
```
- **swap 从 984 暴降到 80** (-92%) - 保护期防止"换入->马上逐出"死循环
- **sparehit 0.6% -> 3.5%** (5.8x) - 换入的专家真正被后续命中
- 防抖原理: 新换入专家 GRACE_TICKS(32) 内不可逐出; 逐出选 LFU(slot_hits 最少)
- 窗口(2048 token)重置候选证据但保留累计命中(W-TinyLFU 风格)
- 次发现: 之前 soname 每次重编递增(.27->.28->.29->.30), 部署必须按最新拷贝
  (tools/deploy_lru.sh 已固化); slot_tick 残留导致旧二进制统计打不出
- 速度 23.3 t/s (略低基线 25.8 = snap读取+换入开销); pinned 已占 24.6%,
  spare 只补 3.5pp -> #### 该方案验证路径正确但提升有限

### subagent 结论 (eviction policy analysis)
- 反应式换入在无时间局部性下收益低(选中率2.3% x 驻留4-8token = <0.2期望命中)
- 最优 = 速率门控 LFU + 保护期 + 窗口衰减 (WR-LFU+G), 多并发共享专家自动胜出
- 架构级: prefill 驱动的热点规划(静态热集)优于反应式环, 环只是补充
