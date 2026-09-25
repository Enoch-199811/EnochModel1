"""把多个 checkpoint 放在同一把尺子上比较 (tools/eval_compare.py)。

为什么需要它: 词表是**每个 checkpoint 自己一份** (字符级 + 动态扩表), 所以
token id 不能跨模型比较。这里先把语料池的验证集**解码回文本**, 再用每个模型
自己的词表重新编码, 于是困惑度/准确率是同一段文本上的可比值。

指标:
* 验证困惑度 (下一个字符预测, 越低越好) —— 语言建模能力;
* 算术准确率 (easy 任务, 贪心解码) —— "会算"的能力;
* 词表覆盖率 —— 该模型能表示验证文本的多少字符;
* 对话样例 —— 人眼看"日常智能"到了哪一步。

用法::

    python tools/eval_compare.py --pool /tmp/enoch_pool.npz \
        --checkpoints checkpoints/chat:旧模型 checkpoints/daily-128:d128 \
        --eval-n 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enochmodel1._paths import PROJECT_ROOT  # noqa: E402
from enochmodel1.enoch import (  # noqa: E402
    build_lm_batch,
    evaluate,
    generate_text,
    load_checkpoint,
    load_config,
    model_from_config,
    token_loss,
    tokenizer_from_config,
)

PROMPTS = ("你好", "今天天气怎么样？", "1+2=", "你是谁？", "吃饭了吗？",
           "我喜欢", "3+5=", "晚安")


def val_perplexity(model, tokenizer, ids: np.ndarray, length: int, batch: int,
                   n_batches: int = 8) -> float:
    rng = np.random.default_rng(0)
    losses = []
    length = max(2, min(length, len(ids) - 1))
    for _ in range(n_batches):
        X, targets, weights = build_lm_batch(ids, length, batch, rng, tokenizer)
        logits, _ = model.forward(X)
        loss, _ = token_loss(logits, targets, weights, beta=0.0)
        losses.append(loss)
    return float(np.exp(np.mean(losses)))


def val_text(pool: str, val_chars: int) -> str:
    data = np.load(pool, allow_pickle=False)
    ids = data["ids"]
    vocab = [str(c) for c in data["vocab"]]
    tail = ids[-min(val_chars, ids.size // 10):]
    return "".join(vocab[int(i)] for i in tail)


def main() -> None:
    ap = argparse.ArgumentParser(description="多 checkpoint 同尺对比")
    ap.add_argument("--pool", type=str, default="/tmp/enoch_pool.npz")
    ap.add_argument("--val-chars", type=int, default=200_000)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="格式 路径[:标签], 多个用空格分隔")
    ap.add_argument("--eval-n", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--out", type=str, default=None, help="结果 JSON")
    args = ap.parse_args()

    text = val_text(args.pool, args.val_chars)
    rows = []
    for spec in args.checkpoints:
        path, _, tag = spec.partition(":")
        ckpt = Path(path)
        if not ckpt.is_absolute():
            ckpt = PROJECT_ROOT / ckpt
        cfg = load_config(str(ckpt))
        tokenizer = tokenizer_from_config(cfg)
        model = model_from_config(tokenizer, cfg)
        load_checkpoint(model, tokenizer, str(ckpt))

        usable = "".join(c for c in text if c in tokenizer.stoi)
        ids = np.array([tokenizer.stoi[c] for c in usable], dtype=np.int32)
        ppl = val_perplexity(model, tokenizer, ids,
                             min(128, model.max_pos), 8)
        ev_args = argparse.Namespace(task="easy", max_new=args.max_new,
                                     max_pos=model.max_pos, score_mode="partial")
        ev = evaluate(model, tokenizer, np.random.default_rng(0), ev_args,
                      n=args.eval_n)
        rng = np.random.default_rng(7)
        # 词表是每个 checkpoint 自己一份, prompt 里可能有该模型没见过的字
        samples = {p: generate_text(model, tokenizer, p, 24, 0.8, rng)
                   for p in PROMPTS if all(c in tokenizer.stoi for c in p)}
        rows.append({
            "tag": tag or ckpt.name,
            "path": str(ckpt),
            "params": int(sum(v.size for v in model.params.values())),
            "dtype": str(model.dtype),
            "d_model": model.d_model, "n_layers": model.n_layers,
            "max_pos": model.max_pos, "vocab_size": tokenizer.vocab_size,
            "coverage": round(len(usable) / max(len(text), 1), 4),
            "val_perplexity": round(ppl, 2),
            "arith_accuracy": round(ev["accuracy"], 3),
            "arith_mean_len": round(ev["mean_len"], 2),
            "samples": samples,
        })

    print("| 模型 | 参数 | 结构 | 词表 | 覆盖率 | 验证困惑度 | 算术准确率 |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['tag']} | {r['params']:,} | d{r['d_model']}/L{r['n_layers']}"
              f"/ctx{r['max_pos']} {r['dtype']} | {r['vocab_size']} | "
              f"{r['coverage']:.0%} | **{r['val_perplexity']}** | "
              f"{r['arith_accuracy']} |")
    print()
    for r in rows:
        print(f"### {r['tag']} 对话样例")
        for p, out in r["samples"].items():
            print(f"- `{p}` → `{out}`")
        print()

    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
