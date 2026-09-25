"""把多个同结构 checkpoint 的权重平均 (tools/average_checkpoints.py)。

用途：**跨 job 数据并行**。同一 `config`、同一 `--seed`（=> 相同初值）、不同
`--data-seed`（=> 不同数据顺序）的 N 个 CI job 各自训练完之后，把权重平均常常
比任何单个模型更稳（Local SGD / 联邦平均的经典做法）。

本工具不盲信平均：会**逐个评估输入模型与平均模型**（验证困惑度 + 算术准确率），
并把对比表打出来，方便"取最优"。

用法::

    python tools/average_checkpoints.py --pool /tmp/pool.npz \\
        --out checkpoints/avg-128 \\
        checkpoints/jobA checkpoints/jobB checkpoints/jobC
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 先导入本包 (它在 numpy 之前把 BLAS 线程设为 1), 再导入 numpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from enochmodel1._paths import PROJECT_ROOT  # noqa: E402
from enochmodel1.enoch import (  # noqa: E402
    CharTokenizer,
    TinyTransformer,
    build_lm_batch,
    evaluate,
    load_checkpoint,
    load_config,
    model_from_config,
    save_checkpoint,
    token_loss,
    tokenizer_from_config,
)
from enochmodel1.merge import average_models  # noqa: E402


def load_pool(path: str) -> tuple[np.ndarray, CharTokenizer]:
    data = np.load(path, allow_pickle=False)
    return data["ids"].astype(np.int32), CharTokenizer(
        vocab=[str(c) for c in data["vocab"]])


def val_perplexity(model: TinyTransformer, tokenizer: CharTokenizer,
                   val_ids: np.ndarray, length: int, batch: int,
                   n_batches: int = 8) -> float:
    rng = np.random.default_rng(0)
    length = max(2, min(length, len(val_ids) - 1))
    losses = []
    for _ in range(n_batches):
        X, targets, weights = build_lm_batch(val_ids, length, batch, rng, tokenizer)
        logits, _ = model.forward(X)
        loss, _ = token_loss(logits, targets, weights, beta=0.0)
        losses.append(loss)
    return float(np.exp(np.mean(losses)))


def load_model(ckpt: Path, fallback_tokenizer: CharTokenizer | None = None):
    cfg = load_config(str(ckpt))
    tokenizer = tokenizer_from_config(cfg) if cfg else fallback_tokenizer
    model = model_from_config(tokenizer, cfg)
    load_checkpoint(model, tokenizer, str(ckpt))
    return model, tokenizer, cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="平均多个同结构 checkpoint")
    ap.add_argument("checkpoints", nargs="+", help="checkpoint 目录 (≥2 个)")
    ap.add_argument("--pool", type=str, default="/tmp/enoch_pool.npz")
    ap.add_argument("--val-chars", type=int, default=200_000)
    ap.add_argument("--out", type=str, default=None,
                    help="平均后的 checkpoint 输出目录 (默认不保存)")
    ap.add_argument("--eval-n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--json", type=str, default=None, help="结果 JSON 路径")
    args = ap.parse_args()

    if len(args.checkpoints) < 2:
        raise SystemExit("至少给两个 checkpoint")

    ids, pool_tok = load_pool(args.pool)
    val_chars = min(args.val_chars, ids.size // 10)
    val_ids = ids[-val_chars:]

    loaded = []
    rows = []
    for spec in args.checkpoints:
        ckpt = Path(spec)
        if not ckpt.is_absolute():
            ckpt = PROJECT_ROOT / ckpt
        model, tokenizer, _cfg = load_model(ckpt)
        loaded.append(model)
        rows.append({"name": ckpt.name, "model": model, "checkpoint": str(ckpt)})

    averaged = average_models(loaded)
    rows.append({"name": "average", "model": averaged, "checkpoint": "(内存)"})

    for row in rows:
        model = row["model"]
        ppl = val_perplexity(model, pool_tok, val_ids,
                             min(128, model.max_pos), 8)
        ev_args = argparse.Namespace(task="easy", max_new=args.max_new,
                                     max_pos=model.max_pos, score_mode="partial")
        ev = evaluate(model, pool_tok, np.random.default_rng(0), ev_args,
                      n=args.eval_n)
        row.update({"params": int(sum(v.size for v in model.params.values())),
                    "val_perplexity": round(ppl, 2),
                    "arith_accuracy": round(ev["accuracy"], 3)})

    print("| 模型 | 参数 | 验证困惑度 | 算术准确率 |")
    print("| --- | --- | --- | --- |")
    for row in rows:
        print(f"| {row['name']} | {row['params']:,} | {row['val_perplexity']} | "
              f"{row['arith_accuracy']} |")

    best = min(rows, key=lambda r: r["val_perplexity"])
    print(f"\n最优: {best['name']} (ppl {best['val_perplexity']})")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        ckpt_args = argparse.Namespace(d_model=averaged.d_model)
        save_checkpoint(averaged, pool_tok, ckpt_args, str(out))
        print(f"平均模型已保存到 {out}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            [{k: v for k, v in r.items() if k != "model"} for r in rows],
            ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
