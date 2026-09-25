"""
EnochModel1: 权重平均 (enochmodel1.merge)
==================================

跨 job 数据并行的收尾动作: 同一 `--seed` (相同初值) 、不同 `--data-seed`
(不同数据顺序) 的多个 job 各自训练后, 把权重平均起来。平均前会校验参数集合与
形状完全一致, 不一致直接报错而不是静默出垃圾。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .enoch import CharTokenizer, TinyTransformer, build_lm_batch, token_loss


def check_compatible(models: list[TinyTransformer]) -> None:
    """校验所有模型的参数集合与形状一致。"""
    base = models[0]
    for i, m in enumerate(models[1:], start=1):
        if list(m.params.keys()) != list(base.params.keys()):
            raise ValueError(f"第 {i} 个模型的参数集合与基准不同")
        for k in base.params:
            if m.params[k].shape != base.params[k].shape:
                raise ValueError(
                    f"第 {i} 个模型的参数 {k} 形状不同: "
                    f"{base.params[k].shape} vs {m.params[k].shape}")


def average_models(models: list[TinyTransformer]) -> TinyTransformer:
    """逐参数平均, 返回新模型 (不改动输入)。

    中间用 float64 累加再落回原 dtype, 避免 float32 下多个大模型相加的累积误差。
    """
    if not models:
        raise ValueError("至少要给一个模型")
    check_compatible(models)
    base = models[0]
    out = TinyTransformer(base.vocab_size, base.d_model, base.n_layers,
                          base.n_heads, base.max_pos, seed=0,
                          d_mlp=base.d_mlp, attn_dim=base.attn_dim,
                          dtype=base.dtype)
    for k in base.params:
        stacked = np.mean([m.params[k].astype(np.float64) for m in models], axis=0)
        out.params[k] = stacked.astype(base.dtype, copy=False)
    return out


def val_perplexity(model: TinyTransformer, tokenizer: CharTokenizer,
                   val_ids: np.ndarray, length: int, batch: int = 8,
                   n_batches: int = 8) -> float:
    """验证集困惑度 = exp(平均下一个字符 CE), 只做前向。"""
    if len(val_ids) < 2:
        return float("nan")
    rng = np.random.default_rng(0)
    length = max(2, min(length, len(val_ids) - 1))
    losses = []
    for _ in range(n_batches):
        X, targets, weights = build_lm_batch(val_ids, length, batch, rng, tokenizer)
        logits, _ = model.forward(X)
        loss, _ = token_loss(logits, targets, weights, beta=0.0)
        losses.append(loss)
    return float(np.exp(np.mean(losses)))


# artifact 目录命名: ckpt-<config>-r<副本号>; 兼容旧的 ckpt-<config>
REPLICA_RE = re.compile(r"^ckpt-(?P<config>.+?)(?:-r(?P<replica>\d+))?$")


def group_replica_dirs(root: str | Path) -> dict[str, list[tuple[int, Path]]]:
    """扫描 artifact 根目录, 按配置分组返回 ``{config: [(副本号, 目录), ...]}``。"""
    root = Path(root)
    groups: dict[str, list[tuple[int, Path]]] = {}
    if not root.is_dir():
        return groups
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d / "model.npz").is_file():
            continue
        m = REPLICA_RE.match(d.name)
        if not m:
            continue
        idx = int(m.group("replica") or 0)
        groups.setdefault(m.group("config"), []).append((idx, d))
    return groups
