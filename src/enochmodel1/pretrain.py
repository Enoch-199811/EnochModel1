"""
EnochModel1: 预训练入口 (enochmodel1.pretrain)
=====================================

两个可叠加的训练来源:

1. 语料无监督语言建模 (--corpus)
   给定一个 UTF-8 文本文件, 随机切窗口预测下一个字符。这是真正意义上的
   "预训练": 先让模型学会语言本身 (包括中文、数字、标点)。

2. 算术监督训练 (--task-steps)
   用四则运算 (prompt=算式, 标准答案) 做 MLE, 学会"答对"。

示例
----
    uv run enoch-pretrain --corpus data/corpus.txt --lm-steps 2000
    uv run enoch-pretrain --corpus data/corpus --lm-steps 10000
                          # 分片目录: 流式读取, 不占满内存
    uv run enoch-pretrain --task easy --task-steps 500
    uv run enoch-pretrain --corpus data/corpus.txt --lm-steps 2000 \\
                          --task-steps 300 --out-dir checkpoints/pretrain
    uv run enoch-pretrain --resume checkpoints/pretrain --lm-steps 1000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ._paths import CHECKPOINTS_DIR, resolve
from .enoch import (
    Adam,
    CharTokenizer,
    evaluate,
    generate_text,
    grow_vocab,
    lm_step,
    load_checkpoint,
    load_config,
    model_from_config,
    pretrain_step,
    save_checkpoint,
    show_examples,
    tokenizer_from_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EnochModel1: 预训练 (语料 LM + 算术监督, 纯 NumPy)")
    p.add_argument("--corpus", type=str, default=None,
                   help="语料文本文件 (UTF-8), 做无监督下一个字符预测")
    p.add_argument("--lm-steps", type=int, default=1000,
                   help="语料语言模型训练步数")
    p.add_argument("--lm-len", type=int, default=48,
                   help="语料窗口长度 (每步从语料随机切一段)")
    p.add_argument("--task", choices=["easy", "hard"], default="easy",
                   help="算术任务难度 (用于监督训练)")
    p.add_argument("--task-steps", type=int, default=0,
                   help="算术监督训练步数 (0=跳过)")
    p.add_argument("--interleave", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="语料 LM 与算术监督同时存在时交替训练 (默认开启), "
                        "避免算术阶段把 EOS 当成默认输出而盖掉语言能力")
    p.add_argument("--verbose-pretrain", action="store_true",
                   help="算术目标设为 answer+answer (先学'啰嗦')")
    p.add_argument("--batch-prompts", type=int, default=8,
                   help="每步样本数")
    p.add_argument("--group-size", type=int, default=1,
                   help="每个算术题复制几份进 batch (默认 1)")
    p.add_argument("--max-new", type=int, default=24,
                   help="评估/示例时回答最大 token 数")
    p.add_argument("--d-model", type=int, default=64, help="模型宽度")
    p.add_argument("--n-layers", type=int, default=2, help="Transformer 层数")
    p.add_argument("--n-heads", type=int, default=4, help="注意力头数")
    p.add_argument("--max-pos", type=int, default=128, help="最大序列长度")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                   help="参数 dtype: float64 最精确 (默认), float32 省一半内存/更快")
    p.add_argument("--lr", type=float, default=3e-3, help="学习率")
    p.add_argument("--entropy-beta", type=float, default=0.01,
                   help="熵正则强度")
    p.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪范数")
    p.add_argument("--log-every", type=int, default=10, help="日志频率 (步)")
    p.add_argument("--eval-every", type=int, default=100,
                   help="评估/保存频率 (步)")
    p.add_argument("--eval-n", type=int, default=20, help="评估样本数")
    p.add_argument("--out-dir", type=str, default="checkpoints/pretrain",
                   help="checkpoint 输出目录")
    p.add_argument("--resume", type=str, default=None,
                   help="从该目录恢复模型参数继续训练")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    return p.parse_args()


def load_corpus(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def corpus_shards(path: str | None) -> list[str]:
    """把 --corpus 解析为语料分片列表。

    单个文件 → 单元素列表; 目录 → 目录下所有 *.txt (含 shards/ 子目录,
    按文件名排序), 用于流式读取 10 亿级语料。
    """
    if not path:
        return []
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.txt"))
        if not files and (p / "shards").is_dir():
            files = sorted((p / "shards").glob("*.txt"))
        if not files:
            raise SystemExit(f"目录 {path} 里没有找到语料分片 (*.txt)")
        return [str(f) for f in files]
    return [path]


def encode_ids_np(tokenizer: CharTokenizer, text: str) -> np.ndarray:
    """把文本编码成 np.int32 数组 (流式分片用, 避免超大 Python list)。"""
    return np.fromiter((tokenizer.stoi[c] for c in text), dtype=np.int32,
                       count=len(text))


def allocate_shard_steps(shard_paths: list[str],
                         lm_steps: int) -> list[tuple[str, int]]:
    """把 lm_steps 均匀分配到各分片; 步数少于分片数时只用前几片。"""
    n = len(shard_paths)
    if lm_steps <= n:
        return [(path, 1) for path in shard_paths[:lm_steps]]
    base, rem = divmod(lm_steps, n)
    return [(path, base + (1 if i < rem else 0))
            for i, path in enumerate(shard_paths)]


def main() -> None:
    args = parse_args()
    args.corpus = resolve(args.corpus)
    args.out_dir = resolve(args.out_dir) or str(CHECKPOINTS_DIR / "pretrain")
    args.resume = resolve(args.resume)
    if not args.corpus and args.task_steps <= 0:
        raise SystemExit(
            "请至少指定 --corpus (语料 LM) 或 --task-steps (算术监督), 例如:\n"
            "  uv run enoch-pretrain --corpus data/corpus.txt\n"
            "  uv run enoch-pretrain --task easy --task-steps 300")

    print("=" * 64)
    print("EnochModel1: 预训练")
    print("=" * 64)
    print(f"语料 LM: {args.corpus or '无'} ({args.lm_steps} 步) | "
          f"算术监督: task={args.task}, {args.task_steps} 步")
    print(f"模型: d={args.d_model}, layers={args.n_layers}, "
          f"heads={args.n_heads}, max_pos={args.max_pos}")

    rng = np.random.default_rng(args.seed)
    tokenizer = CharTokenizer()

    shard_paths = corpus_shards(args.corpus)
    corpus = None
    corpus_ids = None
    if shard_paths:
        corpus = load_corpus(shard_paths[0])
        tokenizer.add(corpus)
        corpus_ids = encode_ids_np(tokenizer, corpus)
        print(f"语料: {len(shard_paths)} 片, 首片 {len(corpus):,} 字符, "
              f"词表 {tokenizer.vocab_size}")

    # 恢复时用 checkpoint 里的维度与词表, 避免参数形状不匹配
    cfg = load_config(args.resume) if args.resume else None
    if cfg:
        tokenizer = tokenizer_from_config(cfg)
        if corpus:
            tokenizer.add(corpus)
            corpus_ids = encode_ids_np(tokenizer, corpus)
        for key in ("d_model", "n_layers", "n_heads", "max_pos", "d_mlp",
                    "dtype"):
            if cfg.get(key) is not None:
                setattr(args, key, cfg[key])
        print(f"恢复配置: {args.resume} (词表 {tokenizer.vocab_size})")

    model = model_from_config(
        tokenizer, cfg,
        fallback={"d_model": args.d_model, "n_layers": args.n_layers,
                  "n_heads": args.n_heads, "max_pos": args.max_pos,
                  "dtype": args.dtype},
        seed=args.seed)
    if args.resume:
        load_checkpoint(model, tokenizer, args.resume)
        print(f"已从 {args.resume} 恢复模型参数\n")
    opt = Adam(model.params)

    t0 = time.time()
    do_lm = bool(shard_paths)
    do_task = args.task_steps > 0
    interleave = args.interleave and do_lm and do_task
    alloc = allocate_shard_steps(shard_paths, args.lm_steps)

    # ---- 训练: 语料 LM 与算术监督 (可交替) ----
    if interleave:
        total = max(args.lm_steps, args.task_steps)
        print(f"\n[交替训练] 每轮: LM 一步 + 算术一步, 共 {total} 轮 "
              f"(LM {args.lm_steps} 步, 算术 {args.task_steps} 步, "
              f"{len(shard_paths)} 片语料)...")
        lm_done = task_done = 0
        for si, (shard_path, steps) in enumerate(alloc):
            if lm_done >= args.lm_steps:
                break
            corpus = load_corpus(shard_path)
            n_new = grow_vocab(model, opt, tokenizer, corpus)
            corpus_ids = encode_ids_np(tokenizer, corpus)
            extra = f", 新增词表 {n_new}" if n_new else ""
            print(f"  [分片 {si + 1}/{len(alloc)}] {Path(shard_path).name} "
                  f"({len(corpus):,} 字符{extra})")
            for _ in range(steps):
                lm_loss = lm_step(model, opt, tokenizer, corpus_ids, rng, args)
                lm_done += 1
                task_loss = None
                if task_done < args.task_steps:
                    task_loss = pretrain_step(model, opt, tokenizer, rng, args)
                    task_done += 1
                if lm_done % args.log_every == 0 or lm_done == args.lm_steps:
                    parts = [f"lm={lm_loss:.4f}"]
                    if task_loss is not None:
                        parts.append(f"task={task_loss:.4f}")
                    print(f"  step {lm_done}/{total}: " + " ".join(parts))
                if lm_done % args.eval_every == 0 or lm_done == args.lm_steps:
                    seed = corpus[max(0, len(corpus) // 2):][:24].strip()
                    sample = generate_text(model, tokenizer, seed,
                                           min(32, args.max_new), 0.6, rng)
                    print(f"  [生成] {seed}... -> {sample[:40]!r}")
                    if do_task:
                        ev = evaluate(model, tokenizer, rng, args, n=args.eval_n)
                        print(f"  [eval] accuracy={ev['accuracy']:.3f} "
                              f"mean_len={ev['mean_len']:.2f}")
                        show_examples(model, tokenizer, rng, args, n=2)
                    save_checkpoint(model, tokenizer, args, args.out_dir)
            del corpus_ids
        # 任务尾段: task_steps > lm_steps 时剩余的算术监督
        while task_done < args.task_steps:
            task_loss = pretrain_step(model, opt, tokenizer, rng, args)
            task_done += 1
            if task_done % args.log_every == 0 or task_done == args.task_steps:
                print(f"  task step {task_done}/{args.task_steps}: "
                      f"loss={task_loss:.4f}")
            if task_done % args.eval_every == 0 or task_done == args.task_steps:
                ev = evaluate(model, tokenizer, rng, args, n=args.eval_n)
                print(f"  [eval] accuracy={ev['accuracy']:.3f} "
                      f"mean_len={ev['mean_len']:.2f}")
                show_examples(model, tokenizer, rng, args, n=2)
                save_checkpoint(model, tokenizer, args, args.out_dir)
    else:
        # ---- 阶段 1: 语料无监督 LM ----
        if do_lm:
            print(f"\n[阶段 1] 语料语言建模 (预测下一个字符, {args.lm_steps} 步, "
                  f"{len(shard_paths)} 片语料)...")
            lm_done = 0
            for si, (shard_path, steps) in enumerate(alloc):
                if lm_done >= args.lm_steps:
                    break
                corpus = load_corpus(shard_path)
                n_new = grow_vocab(model, opt, tokenizer, corpus)
                corpus_ids = encode_ids_np(tokenizer, corpus)
                extra = f", 新增词表 {n_new}" if n_new else ""
                print(f"  [分片 {si + 1}/{len(alloc)}] {Path(shard_path).name} "
                      f"({len(corpus):,} 字符{extra})")
                for _ in range(steps):
                    loss = lm_step(model, opt, tokenizer, corpus_ids, rng, args)
                    lm_done += 1
                    if lm_done % args.log_every == 0:
                        print(f"  lm step {lm_done}/{args.lm_steps}: "
                              f"loss={loss:.4f}")
                    if lm_done % args.eval_every == 0 or lm_done == args.lm_steps:
                        seed = corpus[max(0, len(corpus) // 2):][:24].strip()
                        sample = generate_text(model, tokenizer, seed,
                                               min(32, args.max_new), 0.6, rng)
                        print(f"  [生成] {seed}... -> {sample[:40]!r}")
                        save_checkpoint(model, tokenizer, args, args.out_dir)
                del corpus_ids

        # ---- 阶段 2: 算术监督训练 ----
        if do_task:
            print(f"\n[阶段 2] 算术监督训练 (task={args.task}, {args.task_steps} 步)...")
            for step in range(args.task_steps):
                loss = pretrain_step(model, opt, tokenizer, rng, args)
                if (step + 1) % args.log_every == 0:
                    print(f"  task step {step + 1}/{args.task_steps}: loss={loss:.4f}")
                if (step + 1) % args.eval_every == 0 or step + 1 == args.task_steps:
                    ev = evaluate(model, tokenizer, rng, args, n=args.eval_n)
                    print(f"  [eval] accuracy={ev['accuracy']:.3f} "
                          f"mean_len={ev['mean_len']:.2f}")
                    show_examples(model, tokenizer, rng, args, n=2)
                    save_checkpoint(model, tokenizer, args, args.out_dir)

    # ---- 收尾 ----
    save_checkpoint(model, tokenizer, args, args.out_dir)
    print("\n" + "=" * 64)
    print(f"预训练完成, checkpoint 已保存到 {args.out_dir}")
    print(f"  词表大小: {tokenizer.vocab_size}")
    print(f"  总步数: {(args.lm_steps if do_lm else 0) + args.task_steps}, "
          f"耗时: {time.time() - t0:.1f}s")
    print("=" * 64)
    if corpus is not None:
        seed = corpus[:24].strip()
        sample = generate_text(model, tokenizer, seed, min(32, args.max_new), 0.6, rng)
        print(f"续写示例: {seed!r} -> {sample!r}")
    if args.task_steps > 0:
        show_examples(model, tokenizer, rng, args, n=3)


if __name__ == "__main__":
    main()
