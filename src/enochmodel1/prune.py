"""
EnochModel1: 结构化剪枝入口 (enochmodel1.prune)
========================================

把已训练好的 checkpoint 真正剪小 —— 不是加 mask, 而是把参数矩阵切掉:

* ``--mlp-prune r``  按 ``‖W_1[:,j]‖ · ‖W_2[j,:]‖`` 的重要性排序, 每个 MLP
  隐藏单元只保留前 ``(1-r)`` 比例, ``W_1`` 列 / ``b_1`` / ``W_2`` 行同时切掉;
* ``--head-prune r`` 按 ``(‖W_q/W_k/W_v 该头块‖) · (‖W_o 该行块‖)`` 排序,
  去掉最不重要的注意力头: ``W_q/W_k/W_v`` 的列块与 ``W_o`` 的行块一起切掉,
  ``attn_dim`` 相应变窄 (这是精确的: 保留头之间的输出拼接顺序不变)。

剪枝后可选跑一段 ``--recover-steps`` 监督恢复训练, 并给出剪枝前后的
参数量 / 体积 / 算术准确率对照。旧 checkpoint 不会被覆盖 (另存 --out-dir)。

示例::

    uv run enoch-prune --checkpoint checkpoints/chat --out-dir checkpoints/chat-pruned \\
                       --mlp-prune 0.5 --head-prune 0.25 --recover-steps 200
    uv run enoch-prune --checkpoint checkpoints/chat --dry-run   # 只看剪枝代价
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from ._paths import CHECKPOINTS_DIR, resolve
from .enoch import (
    Adam,
    CharTokenizer,
    TinyTransformer,
    evaluate,
    load_checkpoint,
    load_config,
    pretrain_step,
    tokenizer_from_config,
)


def mlp_importance(model: TinyTransformer, layer: int) -> np.ndarray:
    """每个 MLP 隐藏单元的重要性: 输入侧范数 × 输出侧范数。"""
    P = model.params
    return (np.linalg.norm(P[f"W_1{layer}"], axis=0)
            * np.linalg.norm(P[f"W_2{layer}"], axis=1))


def head_importance(model: TinyTransformer, layer: int) -> np.ndarray:
    """每个注意力头的重要性: (q/k/v 该头块的范数和) × (W_o 该行块的范数)。"""
    P = model.params
    dh = model.head_dim
    out = np.empty(model.n_heads)
    for h in range(model.n_heads):
        s = slice(h * dh, (h + 1) * dh)
        nin = sum(float(np.linalg.norm(P[f"W_{n}{layer}"][:, s]))
                  for n in ("q", "k", "v"))
        nout = float(np.linalg.norm(P[f"W_o{layer}"][s, :]))
        out[h] = nin * nout
    return out


def _keep_indices(scores: np.ndarray, ratio: float) -> np.ndarray:
    """保留重要性最高的若干项, 返回升序下标 (至少留 1 个)。"""
    n = len(scores)
    n_drop = round(n * ratio)
    n_drop = max(0, min(n_drop, n - 1))
    keep = np.argsort(scores)[::-1][:n - n_drop]
    return np.sort(keep)


def prune_model(model: TinyTransformer, mlp_ratio: float = 0.0,
                head_ratio: float = 0.0) -> TinyTransformer:
    """返回剪枝后的新模型 (不改动原模型)。

    这是"结构性"剪枝: 参数矩阵真的变小, 因此推理与训练都更快、checkpoint
    更小 —— 不是把权重置零的伪剪枝。
    """
    P = model.params
    dh = model.head_dim

    # --- 1. 决定每层的保留头 / 保留隐藏单元 ---
    if head_ratio > 0.0 and model.n_heads > 1:
        avg = np.mean([head_importance(model, l)
                       for l in range(model.n_layers)], axis=0)
        keep_heads = _keep_indices(avg, head_ratio)
    else:
        keep_heads = np.arange(model.n_heads)
    if mlp_ratio > 0.0 and model.d_mlp > 1:
        avg = np.mean([mlp_importance(model, l)
                       for l in range(model.n_layers)], axis=0)
        keep_mlp = _keep_indices(avg, mlp_ratio)
    else:
        keep_mlp = np.arange(model.d_mlp)

    n_heads = len(keep_heads)
    attn_dim = n_heads * dh
    d_mlp = len(keep_mlp)
    cols = np.concatenate([np.arange(h * dh, (h + 1) * dh) for h in keep_heads])

    # --- 2. 建新模型并搬运保留的参数 ---
    new = TinyTransformer(model.vocab_size, model.d_model, model.n_layers,
                          n_heads, model.max_pos, seed=0, d_mlp=d_mlp,
                          attn_dim=attn_dim, dtype=model.dtype)
    dt = model.dtype
    new.params["W_e"] = P["W_e"].copy()
    new.params["W_p"] = P["W_p"].copy()
    new.params["W_out"] = P["W_out"].copy()
    new.params["b_out"] = P["b_out"].copy()
    new.params["g_lnf"] = P["g_lnf"].copy()
    new.params["b_lnf"] = P["b_lnf"].copy()
    for l in range(model.n_layers):
        new.params[f"W_q{l}"] = P[f"W_q{l}"][:, cols].astype(dt, copy=True)
        new.params[f"W_k{l}"] = P[f"W_k{l}"][:, cols].astype(dt, copy=True)
        new.params[f"W_v{l}"] = P[f"W_v{l}"][:, cols].astype(dt, copy=True)
        new.params[f"W_o{l}"] = P[f"W_o{l}"][cols, :].astype(dt, copy=True)
        new.params[f"W_1{l}"] = P[f"W_1{l}"][:, keep_mlp].astype(dt, copy=True)
        new.params[f"b_1{l}"] = P[f"b_1{l}"][keep_mlp].astype(dt, copy=True)
        new.params[f"W_2{l}"] = P[f"W_2{l}"][keep_mlp, :].astype(dt, copy=True)
        new.params[f"b_2{l}"] = P[f"b_2{l}"].copy()
        for k in ("g_ln1", "b_ln1", "g_ln2", "b_ln2"):
            new.params[f"{k}{l}"] = P[f"{k}{l}"].copy()
    return new


def evaluate_model(model: TinyTransformer, tokenizer: CharTokenizer,
                   args: argparse.Namespace, n: int) -> dict:
    rng = np.random.default_rng(args.seed)
    eval_args = argparse.Namespace(task=args.task, max_new=args.max_new,
                                   max_pos=model.max_pos, score_mode="partial")
    return evaluate(model, tokenizer, rng, eval_args, n=n)


def _model_size(model: TinyTransformer) -> tuple[int, int]:
    return (int(sum(v.size for v in model.params.values())),
            int(sum(v.nbytes for v in model.params.values())))


def _residual(model: TinyTransformer, other: TinyTransformer,
              tokenizer: CharTokenizer) -> float:
    """同一批输入下两个模型 logits 的最大差 (剪枝代价的直接度量)。"""
    rng = np.random.default_rng(0)
    X = rng.integers(0, tokenizer.vocab_size, size=(4, 24)).astype(np.int64)
    a, _ = model.forward(X)
    b, _ = other.forward(X)
    return float(np.abs(a - b).max())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EnochModel1: 结构化剪枝 (MLP 隐藏单元 + 注意力头)")
    p.add_argument("--checkpoint", type=str, default=str(CHECKPOINTS_DIR / "chat"),
                   help="要剪枝的 checkpoint 目录")
    p.add_argument("--out-dir", type=str, default=None,
                   help="剪枝结果输出目录 (默认 <checkpoint>-pruned)")
    p.add_argument("--mlp-prune", type=float, default=0.5,
                   help="MLP 隐藏单元剪枝比例 (0~0.95)")
    p.add_argument("--head-prune", type=float, default=0.0,
                   help="注意力头剪枝比例 (0~0.95)")
    p.add_argument("--recover-steps", type=int, default=0,
                   help="剪枝后监督恢复训练步数 (0=不训练)")
    p.add_argument("--recover-lr", type=float, default=3e-3,
                   help="恢复训练学习率")
    p.add_argument("--task", choices=["easy", "hard"], default="easy",
                   help="评估用的算术任务")
    p.add_argument("--eval-n", type=int, default=40, help="评估题数")
    p.add_argument("--max-new", type=int, default=12, help="评估时最大生成长度")
    p.add_argument("--batch-prompts", type=int, default=8,
                   help="恢复训练每步样本数")
    p.add_argument("--entropy-beta", type=float, default=0.01,
                   help="恢复训练熵正则")
    p.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--dry-run", action="store_true", help="只评估, 不保存")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = resolve(args.checkpoint)
    out_dir = resolve(args.out_dir) or f"{ckpt}-pruned"
    if not (0.0 <= args.mlp_prune < 1.0 and 0.0 <= args.head_prune < 1.0):
        raise SystemExit("剪枝比例必须在 [0, 1) 内")

    cfg = load_config(ckpt)
    tokenizer = tokenizer_from_config(cfg)
    model = TinyTransformer(
        tokenizer.vocab_size,
        cfg.get("d_model", 64) if cfg else 64,
        cfg.get("n_layers", 2) if cfg else 2,
        cfg.get("n_heads", 4) if cfg else 4,
        cfg.get("max_pos", 64) if cfg else 64,
        seed=args.seed,
        d_mlp=cfg.get("d_mlp") if cfg else None,
        attn_dim=cfg.get("attn_dim") if cfg else None,
        dtype=cfg.get("dtype", "float64") if cfg else "float64")
    load_checkpoint(model, tokenizer, ckpt)

    n_param0, n_byte0 = _model_size(model)
    print("=" * 68)
    print("EnochModel1: 结构化剪枝")
    print("=" * 68)
    print(f"源: {ckpt}")
    print(f"  结构: d={model.d_model} layers={model.n_layers} heads={model.n_heads} "
          f"attn_dim={model.attn_dim} mlp={model.d_mlp} dtype={model.dtype.name}")
    print(f"  参数 {n_param0:,} | 参数体积 {n_byte0:,} B")

    before = evaluate_model(model, tokenizer, args, args.eval_n)
    print(f"  剪枝前评估 ({args.eval_n} 题): accuracy={before['accuracy']:.3f} "
          f"mean_len={before['mean_len']:.2f}")

    if args.mlp_prune <= 0.0 and args.head_prune <= 0.0:
        raise SystemExit("没有指定剪枝比例 (--mlp-prune / --head-prune)")

    t0 = time.time()
    pruned = prune_model(model, args.mlp_prune, args.head_prune)
    n_param1, n_byte1 = _model_size(pruned)
    print(f"\n剪枝: MLP 保留 {100 * (1 - args.mlp_prune):.0f}% "
          f"({model.d_mlp}->{pruned.d_mlp}), 头 保留 "
          f"{100 * (1 - args.head_prune):.0f}% ({model.n_heads}->{pruned.n_heads})")
    print(f"  结构: heads={pruned.n_heads} attn_dim={pruned.attn_dim} "
          f"mlp={pruned.d_mlp}")
    print(f"  参数 {n_param0:,} -> {n_param1:,} ({100 * n_param1 / n_param0:.1f}%)"
          f" | 参数体积 {n_byte0:,} -> {n_byte1:,} B")
    print(f"  logits 漂移 (同一批随机输入): {_residual(model, pruned, tokenizer):.4f}")

    after = evaluate_model(pruned, tokenizer, args, args.eval_n)
    print(f"  剪枝后评估: accuracy={after['accuracy']:.3f} "
          f"mean_len={after['mean_len']:.2f}")

    if args.recover_steps > 0:
        opt = Adam(pruned.params)
        rng = np.random.default_rng(args.seed)
        targs = argparse.Namespace(
            task=args.task, batch_prompts=args.batch_prompts, group_size=1,
            entropy_beta=args.entropy_beta, lr=args.recover_lr,
            grad_clip=args.grad_clip, verbose_pretrain=False,
            max_new=args.max_new, max_pos=pruned.max_pos, score_mode="partial")
        last = 0.0
        for step in range(args.recover_steps):
            last = pretrain_step(pruned, opt, tokenizer, rng, targs)
            if (step + 1) % 50 == 0:
                print(f"    恢复训练 {step + 1}/{args.recover_steps} loss={last:.4f}")
        rec = evaluate_model(pruned, tokenizer, args, args.eval_n)
        print(f"  恢复训练 {args.recover_steps} 步后: accuracy={rec['accuracy']:.3f} "
              f"mean_len={rec['mean_len']:.2f} (loss={last:.4f})")

    if args.dry_run:
        print("\n--dry-run: 未写盘")
        return

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "model.npz"), **pruned.params)
    new_cfg = dict(cfg or {})
    new_cfg.update({
        "d_model": pruned.d_model, "n_layers": pruned.n_layers,
        "n_heads": pruned.n_heads, "max_pos": pruned.max_pos,
        "d_mlp": pruned.d_mlp, "attn_dim": pruned.attn_dim,
        "dtype": pruned.dtype.name, "vocab": list(tokenizer.chars),
        "prune_source": ckpt, "prune_mlp_ratio": args.mlp_prune,
        "prune_head_ratio": args.head_prune,
        "prune_recover_steps": args.recover_steps,
    })
    import json

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(new_cfg, f, ensure_ascii=False, indent=2)

    npz = os.path.getsize(os.path.join(out_dir, "model.npz"))
    print(f"\n已写出 {out_dir}")
    print(f"  checkpoint 体积 {npz:,} B | 全流程 {time.time() - t0:.1f}s")
    print("  用 enoch-chat --checkpoint " + out_dir + " 即可加载剪枝后的模型")


if __name__ == "__main__":
    main()
