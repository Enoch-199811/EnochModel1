"""EnochModel1: 纯 NumPy 的字符级微型 Transformer。

项目按使用场景提供五个命令行入口 (由 ``uv`` 安装):

* ``enoch-train``          speed × score 强化学习演示;
* ``enoch-pretrain``       预训练: 语料无监督 LM + 算术监督;
* ``enoch-chat``           日常对话, 可纠正 / 打分在线训练;
* ``enoch-prune``          结构化剪枝 (MLP 隐藏单元 / 注意力头)。

核心实现见 :mod:`enochmodel1.enoch`。
"""

from __future__ import annotations

import os as _os

# --- 降本: 小矩阵上关掉多线程 BLAS -----------------------------------------
# d_model=64, 每层矩阵都是 64×64 量级, OpenBLAS 的线程同步开销远大于并行
# 收益 (实测 fwd+bwd 反而更慢, 而 CPU 时间吃掉数倍)。这里在 numpy 被导入
# 之前把线程数设为 1; 需要并行时导出 ENOCH_BLAS_THREADS=8 即可。
_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS")
if not any(v in _os.environ for v in _BLAS_VARS):
    _threads = _os.environ.get("ENOCH_BLAS_THREADS", "1")
    for _v in _BLAS_VARS:
        _os.environ.setdefault(_v, _threads)
del _os, _BLAS_VARS

# 必须在设置好线程环境变量之后再导入 (E402 是有意为之)
from .enoch import (  # noqa: E402
    Adam,
    CharTokenizer,
    KVCache,
    SpeedTracker,
    TinyTransformer,
    build_pretrain_batch,
    build_rl_batch,
    check_gradients,
    evaluate,
    generate_text,
    grow_max_pos,
    grow_vocab,
    lm_step,
    load_checkpoint,
    load_config,
    model_from_config,
    pretrain_step,
    rl_step,
    save_checkpoint,
    show_examples,
    token_loss,
    tokenizer_from_config,
)

__version__ = "0.1.0"

__all__ = [
    "Adam",
    "CharTokenizer",
    "KVCache",
    "SpeedTracker",
    "TinyTransformer",
    "build_pretrain_batch",
    "build_rl_batch",
    "check_gradients",
    "evaluate",
    "generate_text",
    "grow_max_pos",
    "grow_vocab",
    "lm_step",
    "load_checkpoint",
    "load_config",
    "model_from_config",
    "pretrain_step",
    "rl_step",
    "save_checkpoint",
    "show_examples",
    "token_loss",
    "tokenizer_from_config",
]
