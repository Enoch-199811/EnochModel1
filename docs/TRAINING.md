# 云端训练手册与"日常智能"路线

本文记录**实测**出来的云端训练约束、正确的用法、以及达到"日常智能"的分阶段方案。
所有数字都来自 `tools/ci_probe.py`（runner 上实测）与 `benchmarks/`（本机实测）。

## 1. 一句话结论

GitHub Actions 的免费 runner **硬件很好，但 vCPU 有配额节流**：单进程能跑满
（d128 ≈ 9,000–11,700 tok/s），一旦在同一个 job 里开多个进程，整体会被压到
~1/20（4 进程每进程只剩 118 tok/s）。**所以并行度必须放在 job 之间，每个 job 只跑
一个进程**（`workers: 1`）。

## 2. 实测数据（同一份代码、同一配置 d128-L3，batch 8 × lm_len 128）

| 场景 | 本机 i5-13500H | GitHub runner (AMD EPYC 9V74, 4 vCPU / 2 核) |
| --- | --- | --- |
| 单进程 | 70.0 ms/步 → 14,628 tok/s | 87.8–112 ms/步 → **9,128–11,669 tok/s** |
| 2 进程合计 | 21,714 tok/s（×1.48） | **868 tok/s**（每进程 434） |
| 4 进程合计 | 29,652 tok/s（×2.03） | **471 tok/s**（每进程 118，比单进程还慢） |

结论与推论：

- runner 的 CPU 不是瓶颈（EPYC 9V74 单核性能与本机同级），**配额+节流是瓶颈**；
- 旧配置（`workers: 4`）实测只有 545 tok/s，与上表 4 进程的 471 tok/s 完全吻合，
  15 分钟只走了 51 万 token —— 这就是"云端训练慢"的真正原因；
- 修正后单 job 单进程：**约 33–42M token/小时**（d128）。

## 3. 正确的云端用法

```bash
# 单配置：一个 job，一个进程，最长 ~5.5 小时（runner 单 job 硬上限 6h）
gh workflow run train.yml -f config=d128-L3-ctx192 -f minutes=330 -f workers=1

# 多配置横向对比（矩阵：每个配置一个 job，各自独立 runner）
gh workflow run train.yml -f config=all -f minutes=330 -f workers=1

# 含真实中文（维基）：会在 runner 上下载 ~3.5GB dump 再流式解析
gh workflow run train.yml -f config=d128-L3-ctx192 -f minutes=330 -f workers=1 \
    -f dialogue_chars=12000000 -f math_chars=8000000 -f chinese_chars=12000000
```

| 输入 | 默认 | 说明 |
| --- | --- | --- |
| `config` | all | 4 档模型；`all` = 4 个 job 并行 |
| `minutes` | 60 | 每 job 训练分钟数（≤330 安全） |
| `workers` | 1 | **保持 1**：多进程会被配额节流 |
| `sync_steps` | 25 | 每轮平均权重的步数（单进程时只是计时粒度） |
| `data_seed` | 1234 | 同配置多 job 时用不同值：数据顺序不同、模型初值相同 |
| `dialogue/math/code/chinese_chars` | 24M/12M/4M/0 | 语料配比；`chinese>0` 才会下载维基 |

产物：每个 job 上传 `ckpt-<config>`（checkpoint + `train_report.json`）；
`publish` job 汇总对比表并把**最佳模型**推到 `ci-checkpoints` 分支。

> 归档分支是**孤儿分支**（只装模型，不含 `.github/`）：`GITHUB_TOKEN` 属于
> GitHub App，不被允许创建/更新 workflow 文件，从 master 拉分支会被拒。

## 4. 预算 → token 换算（单 job 单进程，实测外推）

| 配置 | 参数 | 单进程速率 | 1 小时 | 5.5 小时 |
| --- | --- | --- | --- | --- |
| d64-L2-ctx128 | 20 万 | ~36,000 tok/s | ~130M | ~715M（容量已到顶） |
| d128-L3-ctx192 | 80 万 | ~9,000–11,700 tok/s | ~33–42M | **~180–230M（≈250 token/参数）** |
| d192-L4-ctx256 | 210 万 | ~3,500–4,500 tok/s | ~13–16M | ~70–90M |
| d256-L6-ctx384 | 400 万 | ~1,500–2,000 tok/s | ~6–7M | ~33–40M |

参考点：Chinchilla 的"参数 : token ≈ 1 : 20"意味着 80 万参数的 d128 只需 ~1,600 万
token 就在数据量上"够了"；**瓶颈因此在容量与数据质量**，不在时长。

## 5. 分阶段路线

| 阶段 | 做法 | 状态 |
| --- | --- | --- |
| S1 选型 | `config=all`、`workers=1`、330 分钟：4 档同数据同预算横向对比 | ✔ 已派发 |
| S2 真实中文 | `chinese_chars=8M~12M` 把维基正文混进池子（合成对话是模板化的，真实中文提升语感） | ✔ 已接入并派发验证 |
| S3 跨 job 数据并行 | 同 `config` 开 N 个 job、不同 `data_seed`（同初值），结束后平均权重（`tools/average_checkpoints.py`）+ 各自评估取最优 | ✔ **实测有效** |
| S4 压缩落盘 | 用 `enoch-prune` 做结构化剪枝（MLP 隐藏单元 + 注意力头），`eval_compare` 同尺评估 | ⏳ 待做 |

### S3 实测（d128，两个 job 各 3 分钟、同初值、不同 data_seed）

| 模型 | 验证困惑度 | 说明 |
| --- | --- | --- |
| job A (data_seed=11) | 62,647 | 单独评估 |
| job B (data_seed=22) | 29,568 | 单独评估 |
| **权重平均** | **20,036** | 比最好的单个模型再降 32% |

> 口径说明：这三个数字在同一段验证文本、同一词表下计算，彼此可比；绝对值和
> 长训任务不可比（这次只有 3 分钟、语料池仅 150 万字符）。结论是**平均这一步
> 本身有效**，值得在长训时按 "同 config 多 job + 平均" 组织。

## 6. 已知限制

- 免费 runner 单 job 上限 6 小时；公开仓库不计费，但并发 job 数受配额限制。
- 同一 job 内不能用多进程（见第 2 节）；同 job 内多线程 BLAS 也无益（小矩阵）。
- artifact 默认保留 30 天；长期要留的模型应归档到 `ci-checkpoints` 分支或 Release。
- 中文维基 dump 下载会占用 ~3.5GB 磁盘与十几分钟，属于"值得但不可频繁"的操作。
