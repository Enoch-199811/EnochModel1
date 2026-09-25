"""EnochModel1: 纯 NumPy 的字符级微型 Transformer。

项目按使用场景提供三个命令行入口 (由 ``uv`` 安装):

* ``enoch-train``          speed × score 强化学习演示;
* ``enoch-pretrain``       预训练: 语料无监督 LM + 算术监督;
* ``enoch-chat``           日常对话, 可纠正 / 打分在线训练。

核心实现见 :mod:`enochmodel1.enoch`。
"""

from __future__ import annotations

from .enoch import (
    Adam,
    CharTokenizer,
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
    "pretrain_step",
    "rl_step",
    "save_checkpoint",
    "show_examples",
    "token_loss",
    "tokenizer_from_config",
]
