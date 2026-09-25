"""
EnochModel1: speed × score 强化学习训练 (enochmodel1.main)
================================================

奖励公式
--------
        reward = speed × score

其中 speed = 1 / max(ema_len, speed_floor) 用 EMA 追踪"上一步输出 token 数"。
边际效应: ∂reward/∂n = -score/n², 输出越长压缩越狠, 最终稳定在
"正确且最短"的行为上。

本入口是「预训练 + RL 收缩」的演示; 核心实现见 enochmodel1.enoch。
独立入口:
    uv run enoch-pretrain    # 预训练 (语料无监督 LM + 算术监督)
    uv run enoch-chat        # 日常对话 (手动输入, 可纠正/打分在线训练)

示例
----
    uv run enoch-train --check-gradients        # 梯度校验
    uv run enoch-train                          # 标准训练 (先答对, RL 保持简洁)
    uv run enoch-train --verbose-pretrain       # 演示收缩: 先教模型啰嗦,
                                                # 再看 speed×score 把输出压缩掉
    uv run enoch-train --resume checkpoints     # 从 checkpoint 继续
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ._paths import CHECKPOINTS_DIR, resolve
from .enoch import (
    Adam,
    CharTokenizer,
    SpeedTracker,
    TinyTransformer,
    check_gradients,
    evaluate,
    load_checkpoint,
    load_config,
    pretrain_step,
    rl_step,
    save_checkpoint,
    show_examples,
    tokenizer_from_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EnochModel1: speed × score 强化学习训练 (纯 NumPy)")
    p.add_argument("--steps", type=int, default=300, help="RL 训练步数")
    p.add_argument("--pretrain-steps", type=int, default=300,
                   help="监督预训练步数 (先学会答对)")
    p.add_argument("--task", choices=["easy", "hard"], default="easy",
                   help="任务难度: easy=个位数, hard=多位数")
    p.add_argument("--verbose-pretrain", action="store_true",
                   help="预训练目标设为 answer+answer (先学'啰嗦'), "
                        "便于观察 RL 阶段 speed×score 的收缩效果")
    p.add_argument("--batch-prompts", type=int, default=8,
                   help="每步采样的 prompt 数")
    p.add_argument("--group-size", type=int, default=4,
                   help="每个 prompt 采样几条回答 (组内 baseline 用)")
    p.add_argument("--max-new", type=int, default=24, help="回答最大 token 数")
    p.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    p.add_argument("--d-model", type=int, default=64, help="模型宽度")
    p.add_argument("--n-layers", type=int, default=2, help="Transformer 层数")
    p.add_argument("--n-heads", type=int, default=4, help="注意力头数")
    p.add_argument("--max-pos", type=int, default=64, help="最大序列长度")
    p.add_argument("--lr", type=float, default=3e-3, help="学习率")
    p.add_argument("--rl-lr", type=float, default=1e-3,
                   help="RL 阶段学习率 (通常比预训练小, 更稳)")
    p.add_argument("--entropy-beta", type=float, default=0.02,
                   help="熵正则强度 (防止策略坍塌)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪范数")
    p.add_argument("--speed-ema", type=float, default=0.9,
                   help="speed 中'上一步 token 数'的 EMA 系数")
    p.add_argument("--speed-floor", type=float, default=1.0,
                   help="speed 分母下限, 防止长度 0 导致奖励爆炸")
    p.add_argument("--score-mode", choices=["exact", "partial"], default="partial",
                   help="score 计算方式 (partial 更稠密, 训练更稳)")
    p.add_argument("--kl-beta", type=float, default=None,
                   help="RL 阶段对预训练策略的 KL 正则强度 (防坍塌); "
                        "默认: verbose-pretrain 时 0, 否则 0.2")
    p.add_argument("--log-every", type=int, default=10, help="日志频率 (步)")
    p.add_argument("--eval-every", type=int, default=25, help="评估频率 (步)")
    p.add_argument("--eval-n", type=int, default=30, help="评估样本数")
    p.add_argument("--out-dir", type=str, default="checkpoints",
                   help="checkpoint 输出目录")
    p.add_argument("--resume", type=str, default=None,
                   help="从该目录恢复模型参数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--check-gradients", action="store_true",
                   help="只做反向传播数值校验后退出")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir = resolve(args.out_dir) or str(CHECKPOINTS_DIR)
    args.resume = resolve(args.resume)
    if args.check_gradients:
        check_gradients(args.seed)
        return

    print("=" * 64)
    print("EnochModel1: speed × score 强化学习")
    print("=" * 64)
    print(f"奖励公式: reward = speed × score, speed = 1/max(ema_len, {args.speed_floor})")
    print("边际效应: ∂reward/∂n = -score/n², 输出越长压缩越狠")
    print(f"任务: {args.task} (verbose_pretrain={args.verbose_pretrain})")
    if not args.verbose_pretrain:
        print("提示: 加 --verbose-pretrain 可直观观察 speed×score 的收缩效果")
    if args.kl_beta is None:
        args.kl_beta = 0.0 if args.verbose_pretrain else 0.2
    print(f"配置: {vars(args)}")
    print()

    rng = np.random.default_rng(args.seed)
    tokenizer = CharTokenizer()
    if args.resume:
        cfg = load_config(args.resume)
        if cfg:
            tokenizer = tokenizer_from_config(cfg)
            for key in ("d_model", "n_layers", "n_heads", "max_pos"):
                if key in cfg:
                    setattr(args, key, cfg[key])
    model = TinyTransformer(tokenizer.vocab_size, args.d_model, args.n_layers,
                            args.n_heads, args.max_pos, seed=args.seed)
    if args.resume:
        load_checkpoint(model, tokenizer, args.resume)
        print(f"已从 {args.resume} 恢复模型 (词表 {tokenizer.vocab_size})\n")
    opt = Adam(model.params)

    t0 = time.time()

    # ---- 阶段 1: 监督预训练 (MLE) ----
    eval0 = None
    if args.pretrain_steps > 0:
        print("[阶段 1] 监督预训练 (学会答对)...")
        for step in range(args.pretrain_steps):
            loss = pretrain_step(model, opt, tokenizer, rng, args)
            if (step + 1) % max(1, args.pretrain_steps // 5) == 0 or step == 0:
                print(f"  pretrain step {step + 1}/{args.pretrain_steps}: loss={loss:.4f}")
        eval0 = evaluate(model, tokenizer, rng, args, n=args.eval_n)
        print(f"  预训练后: accuracy={eval0['accuracy']:.3f}, "
              f"mean_len={eval0['mean_len']:.2f}"
              f"  (这个长度是 speed 收缩的起点)\n")

    # ---- 阶段 2: speed × score 强化学习 ----
    ref_model: TinyTransformer | None = None
    if args.kl_beta > 0.0:
        ref_model = TinyTransformer(tokenizer.vocab_size, args.d_model,
                                    args.n_layers, args.n_heads, args.max_pos,
                                    seed=args.seed)
        ref_model.params = {k: v.copy() for k, v in model.params.items()}
    speed = SpeedTracker(ema=args.speed_ema, floor=args.speed_floor)
    best_acc = -1.0
    best_len = float("inf")
    print("[阶段 2] RL: reward = speed × score ...")
    for step in range(args.steps):
        stats = rl_step(model, opt, tokenizer, rng, speed, ref_model, args)
        if (step + 1) % args.log_every == 0:
            print(f"  rl step {step + 1}/{args.steps}: "
                  f"loss={stats['loss']:.4f} score={stats['mean_score']:.3f} "
                  f"len={stats['mean_len']:.2f} speed={stats['speed']:.3f} "
                  f"(ema_len={speed.ema_len:.2f}) trunc={stats['trunc_rate']:.2f}")
        if (step + 1) % args.eval_every == 0:
            ev = evaluate(model, tokenizer, rng, args, n=args.eval_n)
            better = (ev["accuracy"] > best_acc) or (
                ev["accuracy"] == best_acc and ev["mean_len"] < best_len)
            if better:
                best_acc = ev["accuracy"]
                best_len = ev["mean_len"]
                save_checkpoint(model, tokenizer, args, args.out_dir)
            print(f"  [eval] accuracy={ev['accuracy']:.3f} mean_len={ev['mean_len']:.2f}"
                  f"  (best acc={best_acc:.3f}, len={best_len:.2f})")
            show_examples(model, tokenizer, rng, args, n=3)
            print()

    # ---- 最终评估 ----
    ev = evaluate(model, tokenizer, rng, args, n=args.eval_n * 2)
    print("=" * 64)
    print("训练完成")
    print(f"  最终: accuracy={ev['accuracy']:.3f}, mean_len={ev['mean_len']:.2f}")
    print(f"  最佳: accuracy={best_acc:.3f}, mean_len={best_len:.2f}")
    if eval0 is not None:
        print(f"  收缩: mean_len {eval0['mean_len']:.2f} -> {ev['mean_len']:.2f} "
              f"({(1 - ev['mean_len'] / max(eval0['mean_len'], 1e-9)) * 100:.0f}%)")
    print(f"  耗时: {time.time() - t0:.1f}s")
    print("=" * 64)
    show_examples(model, tokenizer, rng, args, n=5)


if __name__ == "__main__":
    main()
