# EnochModel1

[![ci](https://github.com/Enoch-199811/EnochModel1/actions/workflows/ci.yml/badge.svg)](https://github.com/Enoch-199811/EnochModel1/actions/workflows/ci.yml)
[![train](https://github.com/Enoch-199811/EnochModel1/actions/workflows/train.yml/badge.svg)](https://github.com/Enoch-199811/EnochModel1/actions/workflows/train.yml)

纯 NumPy 的字符级微型 Transformer, 训练目标是"既答对、又简洁"
(奖励 `reward = speed × score`)。项目用 [uv](https://docs.astral.sh/uv/)
管理, 采用 src 布局, 核心代码集中在 `src/enochmodel1/enoch.py` 一份
共享库里。

## 命令入口

| 命令 | 作用 |
| --- | --- |
| `enoch-train` | speed × score 强化学习演示: 先监督预训练学会答对, 再 RL 压缩输出长度 |
| `enoch-pretrain` | 预训练: 语料无监督下一个字符预测 (学语言) + 算术监督 (学会答对) |
| `enoch-chat` | 日常对话: 手动输入, 可以直接纠正 (`!c`) 或打分 (`!s`) 并在线训练 |
| `enoch-build-corpus` | 生成 10 亿字符的分片语料 `data/corpus/` |
| `enoch-prune` | 结构化剪枝: 真的把 MLP 隐藏单元 / 注意力头切掉, 可选恢复训练 |

## 快速开始

```bash
# 0. 安装依赖并生成可执行命令 (numpy 会装进 .venv)
uv sync

# 1. 梯度自检 (可选, 确认反向传播正确)
uv run enoch-train --check-gradients

# 2. 预训练: 语料 LM + 算术监督
#    (小语料用 data/corpus.txt; 10 亿语料用目录 data/corpus)
uv run enoch-pretrain --corpus data/corpus.txt --lm-steps 1000 \
                      --task-steps 300 --out-dir checkpoints/pretrain

# 3. 日常对话 (接着预训练模型聊, 手动输入也能训练)
uv run enoch-chat --checkpoint checkpoints/pretrain
```

`uv sync` 会按 `pyproject.toml` 创建 `.venv` 并锁定依赖 (`uv.lock`)。
命令行里的相对路径 (语料 / checkpoint / 记忆文件) 统一相对于项目根目录
解析, 所以无论从哪个目录启动 `enoch-*` 命令都能找到 `data/` 和
`checkpoints/`。也可以直接 `uv run python -m enochmodel1.pretrain --help`
查看某个入口的完整参数。

## 云端训练 (借用 GitHub Actions 的算力)

仓库自带一条训练流水线 `.github/workflows/train.yml`: 预训练跑在 GitHub 的
runner 上, 不占本机 (公开仓库的 standard runner 不计费, 单 job 上限 6 小时,
每 job 4 vCPU / 16 GB)。

| 环节 | 做法 |
| --- | --- |
| 触发 | Actions → **train** → Run workflow (可选配置 / 分钟数 / worker 数 / 语料配比) |
| 并行 | 每个模型配置一个 job; job 内部再用 `tools/parallel_pretrain.py` 开 N 路**多进程数据并行**(纯 NumPy 单线程最快, 见优化文档), 每 `sync_steps` 步 barrier 同步并平均权重 |
| 语料 | runner 上现造: `enoch-build-corpus` 按 **对话优先** 配比生成 → `enoch-pool` 按 manifest 偏移拼 token 池 (对话 24M / 数学 12M / 代码 4M) |
| 产物 | 每个 job 上传 checkpoint + `train_report.json`; publish job 汇总对比表 (验证困惑度 / 算术准确率 / tok/s), 并把**最佳模型**归档到 `ci-checkpoints` 分支 |
| 判定 | 汇总表以 `val_perplexity_after` 最低者为最佳, 同分看算术准确率 |
| 算力 | 公开仓库 standard runner 免费; 4 vCPU/job, 单 job 上限 6 小时 |

本机只用来冒烟/验证, 日常训练都跑在云端 (CI 里同一条命令, 便于复现):

```bash
uv run enoch-build-corpus --target-chars 40000000 --dialogue-chars 24000000 \
    --math-chars 12000000 --code-chars 4000000 --shard-chars 8000000 --out-dir data/corpus
uv run enoch-pool --corpus-dir data/corpus --out /tmp/pool.npz
python tools/parallel_pretrain.py --config d128-L3-ctx192 --workers 4 --minutes 60 \
    --pool /tmp/pool.npz --out-dir checkpoints/daily-128
```

`ci.yml` 每次 push/PR 会跑: 96 项单元测试 (py3.12/3.13)、梯度数值校验、ruff,
以及一条"造小语料 → 拼池 → 30 秒并行训练 → 剪枝"的端到端冒烟。

## 性能 (记忆化 / 剪枝 / 降本增效)

这一轮优化做了三件事，且**行为逐位不变**（见 `docs/OPTIMIZATION.md` 的等价性表）：

1. **记忆化**: 解码改为"预填充 + KV cache 增量解码"，不再每吐一个 token 就重算整段
   前缀；causal mask / 前向 q,k,v 也都被复用；
2. **计算剪枝**: 反向的 8 处 `np.einsum` 收缩换成单个 dgemm，`token_loss` 不再构造
   one-hot；矩阵很小时关掉多线程 BLAS（默认单线程，CPU 时间降近 20 倍）；
3. **存储剪枝**: `--dtype float32` 档、`savez_compressed` checkpoint、以及
   `enoch-prune` 真正切小参数矩阵（MLP 隐藏单元 + 注意力头，带重要性排序与恢复训练）。

实测（本机 i5-13500H，`benchmarks/bench_report.md`）：

| 负载 | 优化前 | 优化后 | 加速比 |
| --- | --- | --- | --- |
| 贪心解码 | 1.71 ms/token | 0.16 ms/token | **10.9x** |
| 4 路采样解码 | 1.09 ms/token | 0.12 ms/token | **9.2x** |
| 语料 LM 步 | 43.03 ms | 21.38 ms | 2.0x |
| RL 步 (含 rollout) | 244.49 ms | 97.83 ms | 2.5x |
| `enoch-pretrain` 端到端 (200+100 步) | 10.87 s | 5.59 s | 1.9x (CPU 时间 91.6s → 4.6s) |
| 峰值内存 | 113.8 MiB | 99.6 MiB (`--dtype float32` 71.2 MiB) | -12% / -37% |

```bash
# 复现基准
uv run python tools/bench.py --tag optimized --out benchmarks/bench_optimized.json
uv run python tools/compare.py benchmarks/bench_baseline.json \
    benchmarks/bench_optimized.json --out benchmarks/bench_report.md

# 更省内存/更快 (精度略降, 默认仍是 float64)
uv run enoch-chat --checkpoint checkpoints/pretrain --dtype float32

# 结构化剪枝 (不覆盖原 checkpoint)
uv run enoch-prune --checkpoint checkpoints/chat --out-dir checkpoints/chat-pruned \
                   --mlp-prune 0.5 --head-prune 0.25 --recover-steps 300
```

`checkpoints/chat-pruned/` 是这一轮剪枝的产物（参数 210,073 → 168,857，体积
1.69 MB → 1.30 MB，恢复训练后算术准确率 0.983），可直接用
`enoch-chat --checkpoint checkpoints/chat-pruned` 加载。

需要多线程 BLAS 时导出 `ENOCH_BLAS_THREADS=8`（默认 1：小矩阵上单线程更快也更省 CPU）。

## 测试

```bash
uv sync                  # 安装依赖 (pytest 作为 dev 依赖自动装好)
uv run pytest            # 跑全部单元测试 (96 项)
uv run pytest -v         # 显示每个用例
uv run pytest --cov=enochmodel1 --cov-report=term   # 附带覆盖率报告
```

测试覆盖核心库 (softmax / LayerNorm / 分词器 / Transformer 前向反向 /
损失 / Adam / 动态词表与序列扩展)、**增量解码与全量前向的等价性 (含采样 RNG 顺序)**、
dtype 与 checkpoint 往返、结构化剪枝、三种训练步骤 (预训练 / LM / RL)、
checkpoint 往返、入口模块的辅助函数和命令行参数解析。

## 对话里的训练命令

```
你: 你好
Enoch: 你好呀
你: !c 你好！很高兴见到你     # 纠正 → MLE 在线训练一步
你: !s 8                      # 打分 → RL 在线训练一步 (10=完美)
你: !train                    # 把记忆里纠正过/打过分的对话批量回放训练
你: !save                     # 保存 checkpoint
你: !q                        # 退出 (自动保存记忆到 data/chat_memory.jsonl)
```

直接输入纯算术表达式 (如 `1+2`) 时, 会自动补成训练格式 `1+2=` 并用贪心
解码, 避免模型把 `=` 当成回答的一部分 (例如输出 `=3`) 或直接输出空回复。
注意: 算术能力只在训练分布内有效 (默认 easy 是个位数); 多位数题目模型
答错或答空都属于能力边界, 不是 bug。

### 日常训练模式 (推荐)

在 REPL 里输入 `!d` 进入日常训练循环: 输入语料 → 模型输出 → 给评分,
评分立刻转成一次 RL 训练, 也可以直接 `!c` 纠正走 MLE:

```
你: !d
训练> 你好
Enoch: 你好呀
评分 (0~10, 回车跳过, !c 纠正, !q 退出): 8
已按 8/10 训练 (RL loss=1.0234)
训练> 2+3=
Enoch: 5
评分 (0~10, 回车跳过, !c 纠正, !q 退出): !c 5
已按正确回答训练 (MLE loss=0.3210)
训练> !q
```

退出训练模式时, 本次会话训练的样本会自动保存 checkpoint, 不需要手动
`!save`。记忆仍然存在 `data/chat_memory.jsonl`, 可用 `!train` 随时回放。

模型是字符级的: 遇到没见过的字 (比如中文) 会自动扩充词表并同步扩展
模型参数, 所以直接输入中文即可, 不需要预处理。

## 语料

仓库自带的 `data/corpus.txt` (数万字符) 是快速演示用的小语料。

`enoch-build-corpus` 默认生成 **10 亿字符**的分片语料, 写到
`data/corpus/` (`shards/shard_*.txt` + `manifest.json`):

- 代码 ~40%: Go / Node.js / PHP 真实源码 (从官方 CDN 下载, 只保留源码);
- 数学 ~30%: 合成四则运算 / 多步表达式 / 应用题 / 解方程 (答案正确);
- 日常对话 ~30%: 合成多轮中文对话 (插槽组合 + 追问 + 知识讲解)。

语料流式写入, 按字符数轮转分片 (默认每片 64M 字符), 不会一次性占用
全部内存; `data/corpus/` 和 `data/raw/` 是构建产物, 已加入 `.gitignore`。

```bash
uv run enoch-build-corpus                             # 10 亿字符 (约 1 分钟)
uv run enoch-build-corpus --target-chars 100000000    # 只要 1 亿
uv run enoch-build-corpus --no-download               # 用已有缓存, 不联网
```

预训练时 `--corpus` 可以传单个文件或分片目录, 目录会按分片**流式读取**
(每片加载后释放, 内存只占一个分片):

```bash
uv run enoch-pretrain --corpus data/corpus --lm-steps 10000 \
                      --task-steps 1000
```

想加大训练量, 直接提高 `--lm-steps` 和 `--task-steps` 即可。

## 交替训练 (默认开启)

预训练同时给语料和算术时, `pretrain.py` 默认**交替训练**: 每轮先做一步
语料语言建模, 再做一步算术监督, 而不是先刷完语料再一口气刷算术。原因:
算术监督的 "answer+EOS" 目标如果刷得太集中, 会把 EOS 学成默认输出,
导致模型对话时也立刻闭嘴; 交替训练能同时保住"答对并停止"和"继续对话"
两种能力。如需旧顺序, 加 `--no-interleave`。

## checkpoint

`checkpoints/` 下每个目录包含:

- `model.npz`: 模型参数;
- `config.json`: 训练配置 + `vocab` (词表, 保证恢复时与模型一致)。

旧版不带 `vocab` 的 checkpoint 也能加载, 自动退回默认算术词表。
