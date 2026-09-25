# EnochModel1 优化记录 (记忆化 / 剪枝 / 降本增效)

本文记录这一轮优化的目标、改了什么、以及**可复现的证据**。所有数字都来自
`tools/bench.py` + `tools/compare.py`，原始 JSON 在 `benchmarks/`。

## 0. 目标与约束

| 项 | 内容 |
| --- | --- |
| 目标 | 让纯 NumPy 的字符级 Transformer 更快、更小、更省 CPU，且**行为不变** |
| 硬约束 | 只依赖 NumPy；三个入口 CLI 与既有测试语义不变；旧 checkpoint 仍可加载 |
| 红线 | 采样解码的 RNG 消耗顺序不能变（否则同 seed 输出不再可复现） |
| 基线锚点 | git 提交 `bcee522`（优化前原始状态），对照实验直接跑该提交的 worktree |

## 1. 优化清单

### 1.1 记忆化：增量解码 (KV cache) —— 最大单项收益

优化前 `generate_batch` 每吐一个 token 就把整段序列重新前向一次（`forward` 覆盖
整个 `[B, W]`），复杂度 O(max_new · L²)。

现在：prompt 只做一次**预填充**并写入 `KVCache`（每层 `[B, H, capacity, dh]`），
之后每步只喂一个新 token（`TinyTransformer.decode_step`），复用历史 K/V。

- 结构证据：`tests/test_incremental.py::test_generate_batch_does_not_recompute_prefix`
  断言 `forward` 只被调用 1 次、`decode_step` 调用次数 ≤ `max_new`。
- 行为证据：与"每步重算前缀"的朴素实现逐 token 一致（含采样，即 RNG 顺序一致）。

### 1.2 记忆化：其它复用与去分配

| 项 | 说明 |
| --- | --- |
| causal mask 缓存 | 原来每次前向都 `np.full + tril_indices` 重建；现在按 `(L, dtype)` 全局复用 |
| q/k/v 复用 | 反向原来重算 3 次投影 matmul；现在前向把 `q,k,v` 存进 cache |
| in-place | `softmax_inplace` / 残差融合 `att_out += h` / `mlp_out += h2`，减少临时数组 |
| 单次分词 | `evaluate` / `show_examples` 原来把 prompt 编码两次 |

### 1.3 计算剪枝：去掉"死算"

| 项 | 说明 |
| --- | --- |
| `einsum` → dgemm | 反向 8 处 `np.einsum("bij,bik->jk", ...)`（c_einsum 曾是单项最大开销，占 ~34%）改为 `atb()` 拍平成一个 dgemm，约 3x |
| 去 onehot | `token_loss` 不再构造 `[B, L, V]` one-hot，只在目标位置减一次 |
| log-softmax | 用 `log_softmax` 取代 `log(clip(softmax))`，更稳且省一次 clip |
| 越界写入剪掉 | 解码达到 `max_pos` 上限时停止增长（原实现会直接抛广播错误） |

### 1.4 降本：单线程 BLAS

模型矩阵都是 64×64 量级，OpenBLAS 的线程同步开销远大于并行收益：默认 16 线程时
`fwd+bwd` 反而更慢，而且 CPU 时间被吃掉近 30 倍。

`enochmodel1/__init__.py` 在导入 numpy 之前把 `OMP/OPENBLAS/MKL/NUMEXPR` 线程数设为
`ENOCH_BLAS_THREADS`（默认 1）；需要并行时导出该变量即可。

> 注意：该默认只在 numpy 尚未导入时生效。入口脚本必须先 `import enochmodel1`
> （`tools/bench.py` 已按此顺序写），或直接在外部导出 `OPENBLAS_NUM_THREADS`。

实测（同一份优化后代码，仅线程数不同）：

| 线程 | fwd+bwd | CPU 时间 (全负载) |
| --- | --- | --- |
| 1 | **25.9 ms/步** | **2.2 s** |
| 16 | 40.1 ms/步 | 61.3 s |

### 1.5 存储剪枝：dtype 档 + checkpoint 压缩

- `--dtype float32`：三个入口都支持，参数体积与峰值内存减半，速度再翻倍；
  默认仍是 `float64`（精度优先）。
- `save_checkpoint` 改用 `np.savez_compressed`；并在 config 里记录模型自身的
  `d_mlp` / `attn_dim` / `dtype`，旧 checkpoint（无这些字段）仍按默认值加载。

| 档位 | 参数体积 | checkpoint (压缩) |
| --- | --- | --- |
| float64 | 1,680,584 B | 1,615,841 B |
| float32 | 840,292 B | 786,820 B |

### 1.6 结构化剪枝：`enoch-prune`

真的把参数矩阵切小（不是置零伪剪枝），两步都带重要性排序：

| 目标 | 重要性 | 切法 |
| --- | --- | --- |
| MLP 隐藏单元 | `‖W_1[:,j]‖ · ‖W_2[j,:]‖` | `W_1` 列 / `b_1` / `W_2` 行同时删 |
| 注意力头 | `(‖W_q/W_k/W_v 该头块‖) · (‖W_o 该行块‖)` | q/k/v 的列块与 `W_o` 的行块一起删，`attn_dim` 变窄 |

为了让"剪头"在数学上精确成立，模型新增了 `attn_dim`（默认 = `d_model`）：
保留头之间的输出拼接顺序不变，所以保留部分的参数逐位不变。

实测（`checkpoints/chat` → `checkpoints/chat-pruned`，MLP 剪 50% + 头剪 25%）：

| 指标 | 剪枝前 | 剪枝后 | 恢复训练 300 步后 |
| --- | --- | --- | --- |
| 参数 | 210,073 | 168,857 (80.4%) | 168,857 |
| 参数体积 | 1,680,584 B | 1,350,856 B | — |
| checkpoint | 1,687,842 B | **1,300,158 B** | — |
| 算术准确率 (60 题) | 1.000 | 0.417 | **0.983** |
| 平均回答长度 | 1.28 | 2.13 | 1.28 |

剪枝 + 300 步恢复训练全程 **1.4 s**（同样 300 步在优化前约 13 s）。

## 2. 等价性证据（优化不能改变行为）

用 git 基线提交 `bcee522` 的 `enoch.py` 与新实现并排加载，同一 seed、同一输入：

| 检查项 | 结果 |
| --- | --- |
| 初始化参数 | 逐位相同 |
| `forward` logits | **逐位相同**（max diff 0.0） |
| `token_loss` (含 KL 档) | loss 差 2.2e-16，dlogits 差 3.5e-18 |
| `backward` 梯度 | 最大绝对差 4.9e-17 |
| `generate_batch` 贪心 | token 序列完全一致，logprob 差 0 |
| `generate_batch` 采样 (group 2/4) | token 序列完全一致，logprob 差 ≤1.8e-15 |
| 20 步 `lm_step` 轨迹 | loss 最大差 8.9e-16；训练后参数差 9.8e-15 |
| 梯度数值校验 | `[PASS]`（最大相对误差 8.6e-05） |

float32 档的精度代价（同一 checkpoint，仅 dtype 不同）：

| 检查项 | 结果 |
| --- | --- |
| 算术评估准确率 | 1.000 vs 1.000 |
| 20 题贪心生成 | 20/20 token 序列一致 |
| 20 步 LM loss 漂移 | 9.9e-07（loss 量级 ~1） |
| 训练后参数相对漂移 | ~1.6e-04 |

## 3. 基准（`benchmarks/bench_report.md` 全表）

负载：语料 prompt 解码 48 token / 4 路采样 24 步 / 语料 LM 步 / RL 步(含 rollout) /
单次 fwd+bwd，均取 3 次最优后再取中位数。

| 负载 | 优化前 | 优化后 (fp64) | 优化后 (fp32) | 加速比 (fp64 / fp32) |
| --- | --- | --- | --- | --- |
| 贪心解码 | 1.71 ms/token | 0.16 | 0.14 | 10.95x / 12.56x |
| 4 路采样 | 1.09 ms/token | 0.12 | 0.09 | 9.21x / 11.98x |
| 语料 LM 步 | 43.03 ms | 21.38 | 11.44 | 2.01x / 3.76x |
| RL 步(含 rollout) | 244.49 ms | 97.83 | 49.98 | 2.50x / 4.89x |
| fwd+bwd | 40.56 ms | 25.87 | 11.77 | 1.57x / 3.45x |
| 峰值 RSS | 113.8 MiB | 99.6 | 71.2 | -12.5% / -37.4% |
| 全负载总耗时 | 2.57 s | 1.26 | 0.63 | 2.04x / 4.08x |

端到端 `enoch-pretrain --lm-steps 200 --task-steps 100`：

| | wall | CPU (user+sys) |
| --- | --- | --- |
| 优化前 | 10.87 s | 91.6 s |
| 优化后 | **5.59 s** | **4.64 s** |

## 4. 复现方式

```bash
# 单元测试 (96 项)
.venv/bin/python -m pytest -q

# 基准 (优化前后同一脚本, 归档 JSON)
.venv/bin/python tools/bench.py --tag optimized --out benchmarks/bench_optimized.json
.venv/bin/python tools/bench.py --tag fp32 --dtype float32
.venv/bin/python tools/compare.py benchmarks/bench_baseline.json \
    benchmarks/bench_optimized.json --out benchmarks/bench_report.md

# 剪枝 (不覆盖原 checkpoint)
.venv/bin/python -m enochmodel1.prune --checkpoint checkpoints/chat \
    --out-dir checkpoints/chat-pruned --mlp-prune 0.5 --head-prune 0.25 \
    --recover-steps 300

# 基线 worktree 对照
git worktree add /tmp/enoch_base bcee522
PYTHONPATH=/tmp/enoch_base/src .venv/bin/python /tmp/enoch_base/tools/bench.py
```

## 5. 已知限制与后续

- 剪枝是**一次性结构化剪枝**，没有做迭代式 prune-and-retrain；更激进的压缩
  （多轮剪枝 + 蒸馏）需要更多训练步数。
- float32 只做实测报告，没有做混合精度（累加用 float64）；长训练下的漂移会累积。
- 单线程 BLAS 的默认值依赖导入顺序；已记录规避方式，但没有引入 `threadpoolctl`
  之类的新依赖。
- 未收录：注意力头维度的细粒度剪枝（会破坏 `attn_dim = Σhead_dim` 的整齐结构）、
  词表行剪枝（会改动 token id 映射，属于破坏性变更）。
