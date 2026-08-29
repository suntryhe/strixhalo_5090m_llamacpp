# MoE hot-expert 卸载优化项目进度

## §35 ROCm aperture 崩溃修复 (2026-08-20)
- mmq.cu:209 cudaMemsetAsync(ids_src1, 0) + quantize.cu 3处 if(i01<0) return
- 部署后 SHA1 ca540601 稳定, 长 prefill 零崩溃

## §36 概念测试结论 (2026-08-20)
- 双 sched 竞态不可用(短请求 SHA1 3次全不同)
- 热槽重排: 4.3% -> 35.0%(top8_real), top12=43.8%
- top16: OOM (CUDA0 显存边界)

## §37-38 ROCm 真减负审计 (2026-08-20)
- rocprofv3: mul_mat_q -11% (2492->2217ms), cold n_nonempty 216->189
- 结论: 时间∝非空专家数, hot 卸载减专家的传导正确

## §39 网格裁剪 (2026-08-21)
- mmq grid z 256->n_nonempty + expert_map 映射, GGML_MMQ_NO_TRIM 开关
- mul_mat_q -3.1% (NOTRIM 2514.9 vs TRIM 2438.1ms)
- 双 pool 分配 bug 修复

## §40 zero_neg1_slots 优化 (2026-08-21)
- 初版每线程串行清整列 + 32x写放大 -> 2D grid + float4 合并写
- 单次 0.781ms -> 0.0074ms (105x), 总 182.7->2.6ms (-98.6%)
- 端到端 7016: 398-404 -> 414.9 t/s (反超无hot基线402), SHA1 一致
- decode 路径(mm vq)核查: 按槽跳过-1, 无低效, 已随专家减负 -6%

## §41 系统重启恢复 (2026-08-22)
- /tmp 是 tmpfs 重启即丢; 固化 tools/moe_deploy.sh + tools/moe_run_server.sh
- 关键坑: hotlibs 缺主版本 symlink => 链接器回退镜像旧库 SIGSEGV(139)
- nvidia-libs 需 libcuda.so/libcuda.so.1 symlink 否则 CUDA0 不可见
- 现用 hot_cfg.json(40层x10=400槽, 源码固化)

## §42 pipeline_parallel 实测结论 + 双 sched 裁决 (2026-08-22)
- 突破: 小 hot(160槽)让 CUDA0 降到 15.7GB, PP_FORCE=1 首次真正启用(无回退), SHA1=3de1ad3c 三次一致
- 同配置对比(160槽 + -c 4096 -b 1024 + 2k请求): PP版 293.2 pps vs 非PP版 326.9 pps
  => pipeline_parallel 负收益! 小配置下 hot 链短, CUDA0 早做完, 事件流水线开销拖慢
- 生产配置(-c 8192 -b 2047 + 400槽): PP 因 CUDA0 OOM 无法启用(需额外 6.9GB)
- 双 sched(MOE_DUAL=1): 400槽 OOM; 160槽能启动但 SHA1 3次全不同(竞态) => 否决
- 最终裁决: 单图 + 400槽 hot + 无PP(-c 8192 -b 2047)是生产最优

## §43 topk 批量优化 (2026-08-22)
- 背景: N卡 nsys prof 发现 CUDA0 路由 topk 逐行启动 cub::DeviceTopK(51.7万次, 1.14s, 22.5% CUDA0) => 最大单点开销
- 根因: top-k.cu `ggml_cuda_op_top_k` 的 `for(i<nrows) top_k_cub(...)` 每 (token,层) 启动一次
- 修复: nrows==1 保留 DeviceTopK; nrows>1 走批量分段 argsort(desc) + memcpy2D 取前k (patch: /tmp/opencode/topk_batch.patch)
- A/B 验证(同 6156-token prefill, hot400):
  - 原版(逐行 DeviceTopK): 410.3 pps | 批量版: 477.1 pps => +16.3%
  - 短请求 SHA1: 两版均 0ca676b1 逐字节一致(数学无回归; §42 的 3de1ad3c 是 160槽 配置基线, 非 hot400, 曾误判为回归)
- nsys 验证: DeviceTopK 启动次数 517,188 -> 282 (↓1834x), 耗时 1137ms -> 11.7ms (↓97x)
- 稳定性: 3x SHA1 全部 0ca676b1, pps 稳定 ~470-477
- 已固化源码(ggml/src/ggml-cuda/top-k.cu), 生产已跑批量版
