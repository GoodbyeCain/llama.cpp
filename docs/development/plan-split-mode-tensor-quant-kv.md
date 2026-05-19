# Plan: SPLIT_MODE_TENSOR + Quantized KV Cache

> Status: draft / working plan. Not for upstream. Owner: saycain.

## 1. 背景

`src/llama-context.cpp:3392-3395` 当前对 `LLAMA_SPLIT_MODE_TENSOR + quantized type_k/type_v` 直接拒绝：

```cpp
if (ggml_is_quantized(params.type_k) || ggml_is_quantized(params.type_v)) {
    LLAMA_LOG_ERROR("%s: simultaneous use of SPLIT_MODE_TENSOR and KV cache quantization not implemented\n", __func__);
    return nullptr;
}
```

这是预防性 guard，并非 kernel 缺失：

- `ggml/src/ggml-cuda/set-rows.cu` 已支持 `Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/IQ4_NL` 量化 dst。
- `ggml/src/ggml-cuda/fattn-vec.cuh` / `fattn-tile.cu` 原生支持量化 K/V。
- `ggml-hip` 对同一份 CUDA 源做 HIP 编译，ROCm 自动继承。
- `tests/test-backend-ops.cpp:7671-7684, 8964-8969` 已覆盖量化变体的单 op 正确性。

真正未验证的是 `ggml/src/ggml-backend-meta.cpp` 在 `SPLIT_MODE_TENSOR` 下的张量 split 推断对量化对齐的处理，以及 `K-shift` / `defrag` 跨设备路径的数值一致性。

## 2. 根因分解

| # | 子问题 | 触发点 | 风险 |
|---|---|---|---|
| ① | KV cache 沿 head 切分是否满足量化 block 对齐 | `src/llama-kv-cache.cpp:210-211` | 低 |
| ② | `cpy_k/cpy_v` 走 `ggml_set_rows`，量化 dst 在多设备下的轴推断 | `src/llama-kv-cache.cpp:1229,1264`；`ggml-backend-meta.cpp:704` | 中 |
| ③ | `FLASH_ATTN_EXT` 沿 axis-2 split + 量化 K/V 未验证 | `ggml-backend-meta.cpp:724` | 中 |
| ④ | K-cache rotation (Hadamard) 跨设备 | `src/llama-kv-cache.cpp:284-324,attn_rot_k/v` | 中 |
| ⑤ | K-shift（RoPE 重应用）反量化→RoPE→量化跨设备 | `src/llama-context.cpp:1773-1835` | 高 |
| ⑥ | defrag：cache 内部搬移涉及反量化/再量化 | `src/llama-context.cpp` (defrag) | 高 |

## 3. 布局约束（前置推理）

KV cache tensor 形状 `[n_embd_k_gqa, kv_size, n_stream]`，其中 `n_embd_k_gqa = n_embd_head * n_head_kv`。

- `SPLIT_MODE_TENSOR` 沿 head 切分 → 实质沿 axis-0 切分 `n_head_kv` 维度
- 切分粒度 = `n_embd_head`（典型值 64/128/192/256）
- 量化 block_size = 32（Q4_x/Q5_x/Q8_0/IQ4_NL 全部）
- **`n_embd_head % 32 == 0` 即可保证切分点落在 block 边界**

→ 这是架构级前置条件，应当在 ctx 创建期校验，而非全盘拒绝。

## 4. 分层方案

### Layer 1 — Ctx-层 guard 改写
**位置**：`src/llama-context.cpp:3382-3396`

把"完全禁止"改为"按前置条件放行"。新增本地辅助函数 `validate_split_tensor_kv_quant(model, params)`：

1. 遍历 layers 校验 `n_embd_head_k(il) % ggml_blck_size(type_k) == 0`（V 同理）
2. 通过 `ggml_backend_dev_supports_op` 对 meta device 下属每张卡询问 SET_ROWS / FLASH_ATTN_EXT 是否支持目标量化类型
3. 校验通过后放行；否则给出明确的失败原因

### Layer 2 — Meta backend 轴推断补强
**位置**：`ggml/src/ggml-backend-meta.cpp`

- `handle_set_rows`（line 704）：dst 量化且沿 axis-0 切分时，每段 `ne` 必须按 `ggml_blck_size(tensor->type)` 整除
- `handle_cpy`（line 558）：同样保护（V-cache 非 FA 路径会触发 CPY）
- `handle_flash_attn_ext`（line 724）：若 K/V 量化，head_dim 必须按 block_size 对齐

**重构前置**：将三个 lambda 提取为文件作用域 static 函数，签名 `(tensor, src_ss) -> split_state`，以便单元测试直接调用。

### Layer 3 — n_stream + tensor split 交叉验证

`src/llama-kv-cache.cpp:1225` 的 `ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream)`：

- reshape 前 split state = `{axis_0, ne=[n_embd_gqa/N0, ...], n_segments=N}`（head-split 在 axis-0）
- reshape 后 `ne[0]=n_embd_gqa`、`ne[1]=kv_size*n_stream`
- `handle_reshape`（line 575）会保留 axis-0 → 应该没问题

**单元测试断言** reshape 后 split state 仍是 axis-0、segments 数与设备数一致。

### Layer 4 — FA 路径回归

`handle_flash_attn_ext` 已断言 K/V 沿 axis-2 split，新增：

- `K.type` 量化时 `K.ne[0] % blck_size == 0`
- `V.type` 量化时 `V.ne[0] % blck_size == 0`
- 返回 `{AXIS_1, ...}` 保持不变

### Layer 5 — K-shift 临时降级（第一阶段不解锁）

`src/llama-context.cpp:1773-1835` 的 `llm_graph_input_k_shift` 路径在量化 KV + tensor split 时数值一致性未知。

**做法**：在 ctx 构造时，若组合命中，强制 `attn_rot_disable = true`（已有环境变量 `LLAMA_ATTN_ROT_DISABLE` 走的同一路径），并通过 `llama_context` 内部 flag 暴露给测试观察。

### Layer 6 — defrag 临时降级（第一阶段不解锁）

若组合命中，强制 `cparams.defrag_thold = -1.0f` 并 `LLAMA_LOG_WARN`，等后续 PR 处理。

## 5. 测试方案

### 5.1 测试金字塔

| Tier | 名称 | 硬件需求 | 覆盖 Layer |
|---|---|---|---|
| 1 | Pure-function unit | CPU only | L2/L3/L4 split 推断 |
| 2 | Op-level integration | 单 GPU | L2/L4 数值正确性 |
| 3 | Arch-level integration | 1+ GPU | L1+L2+L3+L4 全栈 |
| 4 | End-to-end PPL | 多 GPU | 端到端数值漂移 |
| 5 | Guard regression | CPU only | L1/L5/L6 状态机 |

### 5.2 Tier 1 — Pure-function unit

**新文件**：`tests/test-backend-meta-split.cpp`

前置依赖：Layer 2 重构（lambda → static function）。

**测试用例**：

```text
SET_ROWS:
  - dst.type in {Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,IQ4_NL}
  - src_ss[0].axis = AXIS_0
  - segments: aligned (32/64/128) → pass
  - segments: misaligned (33/47) → ASSERT trigger (death test)

CPY:
  - 同上

FLASH_ATTN_EXT:
  - K.type/V.type ∈ {F16, Q8_0, Q4_0}
  - head_dim ∈ {32, 64, 128, 192, 256}
  - 量化 + head_dim 对齐 → 返回 AXIS_1
  - 量化 + head_dim 未对齐 → ASSERT trigger

RESHAPE preservation:
  - Input [n_embd_head*n_head_kv, kv_size, n_stream] split axis_0
  - After reshape_2d → 仍是 axis_0
```

**CI 集成**：CPU-only job，每次 push 跑。

### 5.3 Tier 2 — Op-level integration

**改动**：在 `tests/test-backend-ops.cpp` 加 "meta wrapper" 模式

- 现有测试只对每个注册 backend 逐个跑
- 新模式：对每张 GPU 创建 1-device meta backend（singleton meta）作为对照
- 单 GPU 上即可运行——meta 退化为 pass-through，但仍走 split-state 推断路径

**目标用例**：
- 复用现有的 `test_set_rows` 量化变体（line 7677-7684）
- 复用现有的 `test_flash_attn_ext` 量化 K/V 用例（line 8964-8969）
- 阈值：与原 backend 比较 NMSE < 1e-5

**CI 集成**：所有 GPU job（CUDA + HIP）。

### 5.4 Tier 3 — Arch-level integration（最关键）

**改动位置**：`tests/test-llama-archs.cpp:515-516`

现有 `device_config` 结构没有 KV type 维度。新增内层循环：

```cpp
const std::vector<std::pair<ggml_type, ggml_type>> kv_type_pairs = {
    {GGML_TYPE_F16,    GGML_TYPE_F16   },  // baseline (existing)
    {GGML_TYPE_Q8_0,   GGML_TYPE_Q8_0  },
    {GGML_TYPE_Q4_0,   GGML_TYPE_Q4_0  },
    {GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL},
};
```

`get_model_and_ctx` 增加 `(type_k, type_v)` 参数透传到 `llama_context_params`。

**Skip 逻辑**：架构 `n_embd_head_k % blck_size != 0` 时打 `SKIP` 而非 fail（沿用现有 `arch_supported` 模式）。

**阈值**：沿用 `nmse(logits_cpu, logits_dev) < 1e-4`（line 566）。

**CI 集成**：GPU job，已存在的 `Meta` 配置自然继承多 KV-type 维度。多 GPU runner 上 `devices_meta.size() >= 2` 才真正运行 split tensor。

### 5.5 Tier 4 — End-to-end PPL

**新脚本**：`tests/test-split-tensor-quant-kv.sh`

伪代码：
```bash
MODEL=$1
PPL_SINGLE=$(llama-perplexity -m $MODEL -ctk q8_0 -ctv q8_0 -ngl 99 -f wiki.test.raw)
PPL_SPLIT=$(llama-perplexity  -m $MODEL -ctk q8_0 -ctv q8_0 -ngl 99 -sm row -f wiki.test.raw)
assert abs(PPL_SPLIT - PPL_SINGLE) / PPL_SINGLE < 0.005
```

**CI 集成**：多 GPU runner，nightly。

### 5.6 Tier 5 — Guard regression

**新文件**：`tests/test-context-guards.cpp`（或并入 `test-arg-parser`）

测试矩阵：

| 输入 | 期望 |
|---|---|
| `split_mode=TENSOR + type_k=F16 + type_v=F16` | OK |
| `split_mode=TENSOR + type_k=Q8_0 + head_dim=128` | OK |
| `split_mode=TENSOR + type_k=Q8_0 + head_dim=33`（人工 mock） | reject with clear msg |
| `split_mode=TENSOR + type_k=Q8_0` 触发 K-shift | `ctx.attn_rot_disabled == true` |
| `split_mode=TENSOR + type_k=Q8_0 + defrag_thold=0.5` | `cparams.defrag_thold == -1.0` + warn |

**CI 集成**：CPU-only。

### 5.7 可测试性硬性要求

| Layer | 可测试性改动 |
|---|---|
| L1 | `validate_split_tensor_kv_quant` 作为 file-scope static，签名 `(const llama_model&, const llama_context_params&) -> bool`，测试可直接调用 |
| L2 | meta backend 三个 lambda 提取为 file-scope static 函数 |
| L5 | `llama_context` 新增可观察字段（或 getter）暴露 `attn_rot_force_disabled`、`defrag_force_disabled` |

不满足以上 testability 要求的代码改动不接受。

## 6. 实施次序

每步独立可验证、可单独 revert：

| Step | 内容 | 测试 | 风险 |
|---|---|---|---|
| 1 | Layer 2 重构（lambda 提取为 static），不改逻辑 | Tier 1（含 fixture） | 极低 |
| 2 | Layer 2/3/4 新增量化对齐 ASSERT | Tier 1 | 低 |
| 3 | Layer 1 改写 guard + Layer 5/6 冻结 K-shift/defrag | Tier 5 + Tier 3 | 中 |
| 4 | 跑 Tier 3 全套架构验证 | — | 看结果 |
| 5 | 跑 Tier 4 端到端 PPL | — | 看结果 |
| 6 | （后续 PR）解锁 K-shift | 单独验证 | 高 |
| 7 | （后续 PR）解锁 defrag | 单独验证 | 高 |

每步落 commit，commit message 用 English（项目要求），不在 commit 中提及 AI 协作（AGENTS.md 禁止）。

## 7. 回滚策略

- 引入构建期 macro `LLAMA_DISABLE_TENSOR_SPLIT_QUANT_KV` 控制 guard 是否放行；默认未定义即放行。
- Step 1/2 是纯重构与 ASSERT，行为不变，单 commit revert 即可。
- Step 3 之后若出现数值漂移，revert 该 commit 自动回到旧 guard 行为。

## 8. 明确不做

- 不改 CUDA/HIP kernel 实现
- 不引入 backend-specific 分支；所有判断通过 `ggml_backend_dev_supports_op` 通用询问
- 不为 K-quants（Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / IQ1_x / IQ2_x / IQ3_x）特殊照顾——KV cache 本身也不支持
- 不解锁 MLA + tensor split + 量化 KV（has_v=false 路径单独验证）
- 不写 PR description / commit message 由 AI 生成（AGENTS.md:58）

## 9. 验收清单

- [ ] Tier 1 全绿（CPU job）
- [ ] Tier 2 在 CUDA + HIP 各一台机器上全绿
- [ ] Tier 3 在 Meta 配置下 NMSE < 1e-4，所有 `head_dim % 32 == 0` 的架构通过
- [ ] Tier 4 PPL delta < 0.5%
- [ ] Tier 5 全绿
- [ ] 老路径 `SPLIT_MODE_LAYER` 与 `SPLIT_MODE_NONE` 在 Tier 3 上无回归
- [ ] 文档更新：`docs/build.md` 或 `tools/main/README.md` 提及 `-sm row -ctk qN -ctv qN` 组合的支持范围
