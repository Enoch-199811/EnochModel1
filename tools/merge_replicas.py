"""把同一配置的多个 replica 做权重平均 (tools/merge_replicas.py)。

CI 里"借用算力"的正确形态：同一 `config` 开 N 个 job（各自独立 runner、相同模型
初值、不同 `data_seed`），训练完由本工具在 publish 阶段平均成一个模型 —— 相当于把
N 台 runner 的算力合并，而不是在单台 runner 上开多进程（后者会被 vCPU 配额节流）。

扫描规则（artifact 解压后的目录名）：

* ``ckpt-<config>-r<k>``  → 属于配置 ``<config>`` 的第 k 个 replica；
* ``ckpt-<config>``       → 视为单副本（旧命名，兼容）。

对副本数 ≥ 2 的配置：平均权重、在验证集上评估「各副本 vs 平均」，并把平均后的
模型写成 ``<root>/avg-<config>/``（含 ``train_report.json``，因此会被
``tools/collect_ci_reports.py`` 一起纳入排名）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 先导入本包 (它在 numpy 之前把 BLAS 线程设为 1), 再导入 numpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from enochmodel1.enoch import (  # noqa: E402
    CharTokenizer,
    evaluate,
    load_checkpoint,
    load_config,
    model_from_config,
    save_checkpoint,
    tokenizer_from_config,
)
from enochmodel1.merge import (  # noqa: E402
    average_models,
    group_replica_dirs,
    val_perplexity,
)


def load_pool(path: str) -> tuple[np.ndarray, CharTokenizer]:
    data = np.load(path, allow_pickle=False)
    return (data["ids"].astype(np.int32),
            CharTokenizer(vocab=[str(c) for c in data["vocab"]]))


def main() -> None:
    ap = argparse.ArgumentParser(description="平均同配置的多个 replica")
    ap.add_argument("--root", type=str, default="artifacts",
                    help="artifact 解压根目录")
    ap.add_argument("--pool", type=str, default="/tmp/pool.npz")
    ap.add_argument("--val-chars", type=int, default=200_000)
    ap.add_argument("--eval-n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--min-replicas", type=int, default=2,
                    help="副本数达到多少才做平均")
    args = ap.parse_args()

    root = Path(args.root)
    groups = group_replica_dirs(root)
    pool_ids, pool_tok = load_pool(args.pool)
    val_chars = min(args.val_chars, pool_ids.size // 10)
    val_text = "".join(pool_tok.itos[int(i)] for i in pool_ids[-val_chars:])

    summary = []
    for config, members in sorted(groups.items()):
        if len(members) < args.min_replicas:
            continue
        models, tokenizers, names = [], [], []
        for _idx, d in sorted(members):
            cfg = load_config(str(d))
            tok = tokenizer_from_config(cfg)
            model = model_from_config(tok, cfg)
            load_checkpoint(model, tok, str(d))
            models.append(model)
            tokenizers.append(tok)
            names.append(d.name)

        tok = tokenizers[0]
        usable = "".join(c for c in val_text if c in tok.stoi)
        ids = np.array([tok.stoi[c] for c in usable], dtype=np.int32)
        length = min(128, models[0].max_pos)
        ev_args = argparse.Namespace(task="easy", max_new=args.max_new,
                                     max_pos=models[0].max_pos,
                                     score_mode="partial")
        rows = []
        for name, model in zip(names, models):
            rows.append({
                "name": name, "model": model,
                "val_perplexity": round(val_perplexity(model, tok, ids, length), 3),
                "arith_accuracy": round(evaluate(
                    model, tok, np.random.default_rng(0), ev_args,
                    n=args.eval_n)["accuracy"], 4),
            })

        averaged = average_models(models)
        avg_ppl = round(val_perplexity(averaged, tok, ids, length), 3)
        avg_acc = round(evaluate(averaged, tok, np.random.default_rng(0),
                                 ev_args, n=args.eval_n)["accuracy"], 4)
        rows.append({"name": f"avg-{config}", "model": averaged,
                     "val_perplexity": avg_ppl, "arith_accuracy": avg_acc})

        out = root / f"avg-{config}"
        save_checkpoint(averaged, tok, argparse.Namespace(d_model=averaged.d_model),
                        str(out))
        (out / "train_report.json").write_text(json.dumps({
            "tag": f"avg-{config}", "config": config,
            "replicas": [r["name"] for r in rows[:-1]],
            "params": int(sum(v.size for v in averaged.params.values())),
            "dtype": str(averaged.dtype), "vocab_size": tok.vocab_size,
            "val_perplexity_after": avg_ppl, "arith_accuracy": avg_acc,
            "replica_perplexities": {r["name"]: r["val_perplexity"]
                                     for r in rows[:-1]},
            "checkpoint": str(out),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n### {config}: {len(members)} 个副本")
        print("| 模型 | 验证困惑度 | 算术准确率 |")
        print("| --- | --- | --- |")
        for r in rows:
            print(f"| {r['name']} | {r['val_perplexity']} | {r['arith_accuracy']} |")
        best_single = min(r["val_perplexity"] for r in rows[:-1])
        gain = (1 - avg_ppl / best_single) * 100 if best_single else 0.0
        print(f"平均 vs 最好的单副本: {'↓' if gain > 0 else '↑'}"
              f"{abs(gain):.1f}%  (已写入 {out})")
        summary.append({"config": config, "replicas": len(members),
                        "avg_perplexity": avg_ppl, "best_single": best_single,
                        "gain_pct": round(gain, 1)})

    if not summary:
        print("没有副本数达标的配置, 无需平均")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
