"""同步数据并行预训练 (多进程 + 定期权重平均)。

纯 NumPy 单线程在这个模型尺寸上最快 (见 docs/OPTIMIZATION.md), 所以"借用
多核"的正确姿势不是开 BLAS 线程, 而是**多进程数据并行**: N 个 worker 从同一
初始权重出发, 各自持有参数副本与 Adam 状态, 在共享语料池上采样; 每
``--sync-steps`` 步 barrier 同步一次并平均权重 (Local SGD / 同步数据并行),
然后带着平均后的权重继续。共享语料池用 fork 的写时复制传递, 不复制一份。

用法::

    uv run enoch-pool --out /tmp/pool.npz                       # 拼语料池
    python tools/parallel_pretrain.py --config d128-L3-ctx192 \
        --workers 4 --minutes 10 --pool /tmp/pool.npz \
        --out-dir checkpoints/daily-128                          # 本地/CI 通用
"""

from __future__ import annotations

# 允许直接用 python tools/xxx.py 运行 (不必先 uv sync / 激活 venv)
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

# 必须先导入 enochmodel1 (它在 numpy 之前把 BLAS 线程设为 1), 否则多进程
# worker 会各自开出多线程 BLAS, 在物理核上互相抢 (实测慢 7 倍)。
from enochmodel1._paths import PROJECT_ROOT
from enochmodel1.enoch import (
    Adam,
    CharTokenizer,
    TinyTransformer,
    build_lm_batch,
    evaluate,
    generate_text,
    lm_step,
    save_checkpoint,
    token_loss,
)

CONFIGS: dict[str, dict] = {
    "d64-L2-ctx128": {"d_model": 64, "n_layers": 2, "n_heads": 4,
                      "max_pos": 128, "lm_len": 96},
    "d128-L3-ctx192": {"d_model": 128, "n_layers": 3, "n_heads": 8,
                       "max_pos": 192, "lm_len": 128},
    "d192-L4-ctx256": {"d_model": 192, "n_layers": 4, "n_heads": 8,
                       "max_pos": 256, "lm_len": 160},
    "d256-L6-ctx384": {"d_model": 256, "n_layers": 6, "n_heads": 8,
                       "max_pos": 384, "lm_len": 192},
}

SAMPLES = ("你好", "今天天气怎么样？", "1+2=", "你是谁？", "吃饭了吗？")


def load_pool(path: str) -> tuple[np.ndarray, CharTokenizer]:
    data = np.load(path, allow_pickle=False)
    vocab = [str(c) for c in data["vocab"]]
    return data["ids"].astype(np.int32), CharTokenizer(vocab=vocab)


def _flats(model: TinyTransformer) -> list[np.ndarray]:
    return [model.params[k].ravel() for k in model.params]


def pack(model: TinyTransformer, buf) -> None:
    """参数写进共享缓冲 (float32, 与模型 dtype 一致)。"""
    arr = np.frombuffer(buf, dtype=np.float32)
    o = 0
    for v in _flats(model):
        arr[o:o + v.size] = v
        o += v.size


def unpack(model: TinyTransformer, buf) -> None:
    """从共享缓冲读回参数 (原地写, 不改 dict 结构)。"""
    arr = np.frombuffer(buf, dtype=np.float32)
    o = 0
    for v in _flats(model):
        v[...] = arr[o:o + v.size]
        o += v.size


def worker_entry(idx: int, buf, avg, barrier, stop, ids, model, tokenizer,
                 step_args, sync_steps) -> None:
    """worker: 训练 sync_steps 步 -> 交出权重 -> 取回平均权重 -> 继续。"""
    opt = Adam(model.params)
    flat = np.frombuffer(buf, dtype=np.float32)
    flat_avg = np.frombuffer(avg, dtype=np.float32)
    rng = np.random.default_rng(1234 + idx)
    while True:
        for _ in range(sync_steps):
            lm_step(model, opt, tokenizer, ids, rng, step_args)
        pack(model, flat)
        try:
            barrier.wait(timeout=1800)
            barrier.wait(timeout=1800)
        except Exception:            # noqa: BLE001  (主进程异常时别挂死)
            return
        if stop.value:
            return
        unpack(model, flat_avg)


def val_perplexity(model: TinyTransformer, tokenizer: CharTokenizer,
                   val_ids: np.ndarray, length: int, batch: int,
                   n_batches: int = 8) -> float:
    """验证集困惑度 = exp(平均 CE), 只做前向。"""
    rng = np.random.default_rng(0)
    losses = []
    length = max(2, min(length, len(val_ids) - 1))
    for _ in range(n_batches):
        X, targets, weights = build_lm_batch(val_ids, length, batch, rng, tokenizer)
        logits, _ = model.forward(X)
        loss, _ = token_loss(logits, targets, weights, beta=0.0)
        losses.append(loss)
    return float(np.exp(np.mean(losses)))


def main() -> None:
    ap = argparse.ArgumentParser(description="同步数据并行预训练 (纯 NumPy)")
    ap.add_argument("--config", choices=sorted(CONFIGS), default="d128-L3-ctx192")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--rounds", type=int, default=0, help=">0 时忽略 --minutes")
    ap.add_argument("--sync-steps", type=int, default=20, help="每轮各 worker 步数")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lm-len", type=int, default=0, help="0 = 用配置默认值")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--entropy-beta", type=float, default=0.0)
    ap.add_argument("--pool", type=str, default="/tmp/enoch_pool.npz")
    ap.add_argument("--val-chars", type=int, default=200_000)
    ap.add_argument("--out-dir", type=str, default="checkpoints/daily")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-n", type=int, default=40, help="算术评估题数")
    ap.add_argument("--tag", type=str, default="", help="报告标签")
    args = ap.parse_args()

    cfg = dict(CONFIGS[args.config])
    lm_len = args.lm_len or cfg["lm_len"]
    ids, tokenizer = load_pool(args.pool)
    val_chars = min(args.val_chars, ids.size // 10)
    train_ids, val_ids = ids[:-val_chars], ids[-val_chars:]

    model = TinyTransformer(tokenizer.vocab_size, cfg["d_model"], cfg["n_layers"],
                            cfg["n_heads"], cfg["max_pos"], seed=args.seed,
                            dtype="float32")
    n_param = int(sum(v.size for v in model.params.values()))
    step_args = argparse.Namespace(lm_len=lm_len, batch_prompts=args.batch,
                                   entropy_beta=args.entropy_beta, lr=args.lr,
                                   grad_clip=1.0, task="easy", score_mode="partial",
                                   max_new=12, max_pos=model.max_pos)
    before_ppl = val_perplexity(model, tokenizer, val_ids, lm_len, args.batch)

    print(f"[并行预训练] {args.config} 参数={n_param:,} workers={args.workers} "
          f"batch={args.batch} lm_len={lm_len} 训练池={train_ids.size:,} "
          f"验证池={val_chars:,}", flush=True)
    print(f"[初始] 验证困惑度 {before_ppl:.1f} (随机模型基线)", flush=True)

    ctx = mp.get_context("fork")
    bufs = [ctx.Array("f", n_param, lock=False) for _ in range(args.workers)]
    avg = ctx.Array("f", n_param, lock=False)
    barrier = ctx.Barrier(args.workers + 1)
    stop = ctx.Value("i", 0, lock=False)
    procs = [
        ctx.Process(target=worker_entry,
                    args=(i, bufs[i], avg, barrier, stop, train_ids, model,
                          tokenizer, step_args, args.sync_steps), daemon=True)
        for i in range(args.workers)
    ]
    for p in procs:
        p.start()

    rounds = 0
    t0 = time.time()
    stop_flag = False
    try:
        while True:
            try:
                barrier.wait(timeout=1800)
            except Exception:        # noqa: BLE001
                print("[警告] worker 同步超时/异常, 提前结束", flush=True)
                stop_flag = True
                try:
                    barrier.wait(timeout=30)
                except Exception:    # noqa: BLE001,S110  (主进程已在收尾)
                    pass
                break
            acc = None
            for b in bufs:
                w = np.frombuffer(b, dtype=np.float32)
                acc = w.copy() if acc is None else acc + w
            np.frombuffer(avg, dtype=np.float32)[:] = acc / len(bufs)
            rounds += 1
            el = time.time() - t0
            tok = rounds * args.sync_steps * args.workers * args.batch * lm_len
            if rounds == 1 or rounds % 5 == 0:
                print(f"  轮 {rounds:>4} | {el:6.1f}s | 已见 {tok / 1e6:6.2f}M tokens "
                      f"| {tok / el:8,.0f} tok/s", flush=True)
            done = (rounds >= args.rounds) if args.rounds else (time.time() >=
                                                              t0 + args.minutes * 60)
            if done:
                stop_flag = True
                stop.value = 1
            try:
                barrier.wait(timeout=1800)
            except Exception:        # noqa: BLE001
                break
            if done:
                break
    finally:
        stop.value = 1
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

    unpack(model, avg)
    train_secs = max(time.time() - t0, 1e-9)
    tokens = rounds * args.sync_steps * args.workers * args.batch * lm_len
    after_ppl = val_perplexity(model, tokenizer, val_ids, lm_len, args.batch)
    ev = evaluate(model, tokenizer, np.random.default_rng(0), step_args,
                  n=args.eval_n)
    rng = np.random.default_rng(7)
    samples = {p: generate_text(model, tokenizer, p, 32, 0.8, rng) for p in SAMPLES}

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    ckpt_args = argparse.Namespace(**{**vars(args), "d_model": model.d_model,
                                      "n_layers": model.n_layers,
                                      "n_heads": model.n_heads,
                                      "max_pos": model.max_pos})
    save_checkpoint(model, tokenizer, ckpt_args, str(out_dir))

    report = {
        "tag": args.tag or args.config,
        "config": args.config, "dims": cfg,
        "workers": args.workers, "sync_steps": args.sync_steps, "rounds": rounds,
        "batch": args.batch, "lm_len": lm_len, "lr": args.lr,
        "params": n_param, "dtype": str(model.dtype),
        "tokens_seen": int(tokens), "tokens_per_param": round(tokens / n_param, 2),
        "train_seconds": round(train_secs, 1),
        "tokens_per_s": round(tokens / train_secs, 1),
        "val_perplexity_before": round(before_ppl, 3),
        "val_perplexity_after": round(after_ppl, 3),
        "arith_accuracy": round(ev["accuracy"], 4),
        "arith_mean_len": round(ev["mean_len"], 3),
        "train_chars": int(train_ids.size), "val_chars": int(val_chars),
        "vocab_size": tokenizer.vocab_size, "pool": args.pool,
        "samples": samples, "checkpoint": str(out_dir),
        "interrupted": bool(stop_flag and rounds == 0),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
