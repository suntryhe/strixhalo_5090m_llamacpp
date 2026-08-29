# LRU 专家缓存环设计 (2026-08-24)

参考: lucebox Spark (moe_hybrid_storage_cache_swap, LRU ring) + LvLLM (被废弃避开)
目标: 不增加显存预算, 把动态命中率从 37% (固定200/400槽) 提升到 90%+

## 背景: 为什么固定槽不够

- DeepSeek-V4-Flash 每层 256 专家, top-6 路由
- 实测(28.4万样本): 全局 top-400 专家只覆盖 37% 路由选择
- 固定槽 = 命中率天花板取决于路由分散度, 无法再提高
- lucebox 的答案: 固定热集 + 1.25% LRU spare 环, 冷失误 39.7%->0.8%

## 现状机制(已探明, 代码级证据)

### hot/cold 分卡架构
- 每层 hot 张量: `ffn_up_exps_hot [n_embd, n_ff, n_hot+1]` (up/gate/down 三个), 最后一列 = 零 mask 列
  - llama-model.cpp:1755 `ne2_hot = n_hot + 1`
- 三个运行期可改的 1D 表(映射逻辑核心):
  - `ffn_hot_remap` (I32): 全局专家id -> hot槽位索引; 冷专家->n_hot(mask列)
    - `hot_ids = ggml_get_rows(remap_3d, selected_experts)` (llama-graph.cpp:2140)
  - `ffn_hot_flag` (F32): 1.0=热 / 0.0=冷; 权重分流 `w_hot = weights*flag, w_cold = weights-w_hot` (2182-2184)
  - `ffn_cold_remap` (I32): 热专家->-1(冷链跳过), 冷专家->原id
    - `cold_ids = ggml_get_rows(cold_remap_3d, selected_experts)` (2193)
- selected_experts = argsort_top_k 每次 compute 运行期生成 (2059)
- RADEON 冷链: cold_up [n_embd, n_ff, 256] 在 ROCm0

### 可行性事实 (explore 结论)
1. ids 是运行期张量(argsort 输出), 不是固定 input
2. remap/flag/cold_remap 在 WEIGHTS buffer, get_rows 每次执行重新读取
   -> **改表内容 = 改路由, 无需重建图**
3. ggml_backend_tensor_set 已在加载时写 hot 张量(有先例, llama-model.cpp:1840-1858)
4. hot 张量布局连续: 写第 k 槽 = offset k*col_size 连续写
5. graph reuse (can_reuse) 只比较形状/输入指针, 不比较权重内容
   -> 两 compute 之间改权重, 下个 compute 自动用新数据
6. 冷链 -1 id: mmid kernel 跳过(id<0 无行), w_cold 乘 0 -> 安全
7. 跨 backend get: cold 权重在 ROCm0 (UMA host 可见), host 暂存 -> CUDA0 set 已有先例

## 设计: 三层合一

### 层次划分 (200槽预算示例)
- 热集 hot_active: 如 170 槽(校准静态)
- LRU spare 环: 如 30 槽(每层 ~0-2, 预算 = expert_budget/8, 上限小)
- 纯冷: 其余全在 ROCm

### 核心数据结构 (per layer)
```cpp
int hot_active;                  // 静态热专家数
int cache_slots;                 // spare 槽数
std::vector<int32_t> spare_global;  // 每个 spare 槽当前放哪个全局专家id
std::vector<uint64_t> spare_lru;    // 每槽 last-use tick
uint64_t lru_clock;
```

### 运行期换入流程 (post-compute, 下个 token 生效)
1. decode 完当前 token, 从 snap 读 selected_experts (hotstats 同样路径)
2. 对每层每个选中专家 e:
   - 若 e 已在 hot_active 或 spare: 更新 spare_lru tick (若在 spare)
   - 若 e 是冷专家 且 有 spare 槽:
     a. 选 LRU 最旧槽 s (或空槽)
     b. 若 s 被占用, 逐出旧专家 e': remap[e']=n_hot, flag[e']=0, cold_remap[e']=e'
     c. 把 e 的权重列从 ROCm 拷到 hot 张量槽 s (异步, 独立流)
     d. remap[e]=s, flag[e]=1.0, cold_remap[e]=-1
     e. spare_global[s]=e, spare_lru[s]=++clock
3. 三表更新走 ggml_backend_tensor_set (小张量 256 ints)
4. 下次 compute 自动用新映射

### 逐出流程
- 反向: remap[e]=n_hot(回 mask), flag[e]=0(回冷), cold_remap[e]=e(冷链恢复)

## 为什么 post-compute 反应式而非预测式

- lucebox 实测: 预测式 pre-gate 识别率上限 ~53%
- 反应式: 算完当前 token 立刻换入选中专家, 下个 token 命中
- 代价: 被路由的冷专家第一次出现算在冷链(慢), 此后都在 hot 链 (快)
- 收益: 会话内高频专家迅速常驻 -> 命中率趋近热集内命中 + 缓存饱和

## 实现步骤
1. [P0] llama-model.cpp: ne2_hot = n_hot + 1 + cache_slots (预留 spare), 初始化 spare 列为0
2. [P0] llama-graph.cpp/h: hot 结构带 cache_slots; 三表更新接口
3. [P0] 新增 lru manager: 读 snap -> 决策 -> 拷权重列 -> 更新三表 (挂 llama-context 解码后)
4. [P0] env: LLAMA_EXPERT_LRU=1 + 槽数
5. [P1] 异步拷贝重叠 decode (独立流)
6. [P1] 预算: spark_budget_split 同款 (expert_budget/8)

## 风险
- 权重列拷贝每次 ~6.3MB/专家 (up+gate+down), 逐出+换入 ~12.6MB
  -> ~1-2ms (PCIe 40GB/s), 可异步重叠
- 数值: 热/冷路径 kernel 不同 (MMVQ vs MMQ) 可能有 ULP 漂移
  -> 已有 hot/cold 同槽布局一致 (46e4704), 风险低
- mlu 细节: flash-decoding/图上 pin (hot_ids_raw 已 pin CUDA0)

## 验收
- 命中率: 打开 LRU 后 hotstats 采样, 冷失误率应显著下降
- 速度: decode 应 > 200槽固定(25.8) 且 <= 400槽(26.6) 同量级, 期望更高
- 确定性: 温度0 同 prompt 输出一致 (不要求与不启用LRU一致)