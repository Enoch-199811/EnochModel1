"""EnochModel1 能力评测 (tools/eval_intelligence.py)。

用一批**可复核的探针**测模型的实际能力，而不是只看困惑度：

1. 算术（分布内）：个位数 ± ×（训练分布内的"会不会算"）；
2. 算术（分布外）：2~3 位数 ± ×（能力边界，答错属预期）；
3. 少样本算术：``1+1=2 2+2=4 3+3=`` 看能否延续模式（最小的上下文学习探针）；
4. 日常对话（模板内）：8 个日常 prompt → 非空率 / EOS 率 / 回声率 / 重复率 + 样例；
5. 生成行为：贪心与采样的长度分布、是否立刻 EOS；
6. 语言建模：分布内（对话池验证集）与分布外（技术文档中文）的困惑度；
7. 坍塌检查：相邻重复 token 比例、去重字符比例。

输出：每个模型一行指标 + JSON + 逐条样例，便于人工复核。

用法::

    python tools/eval_intelligence.py --models \\
        旧模型:checkpoints/chat d128-5.5h:/tmp/M/cfg-d128-L3-ctx192 \\
        d256-5.5h:/tmp/M/cfg-d256-L6-ctx384 --json /tmp/intel.json
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
    TinyTransformer,
    load_checkpoint,
    load_config,
    make_prompt,
    model_from_config,
    tokenizer_from_config,
)
from enochmodel1.merge import val_perplexity  # noqa: E402

# 解码约束（CLI 注入；默认 0 = 关闭，与历史结果可比）
DECODE: dict = {"top_k": 0, "no_repeat_ngram": 0}

CHAT_PROMPTS = ("你好", "今天天气怎么样？", "吃饭了吗？", "你是谁？",
                "我喜欢听音乐", "晚安", "谢谢你", "早上好")
COMPLETIONS = ("今天", "我想", "因为")
FEWSHOT = (
    ("1+1=2 2+2=4 3+3=", "6"),
    ("2+3=5 3+4=7 4+5=", "9"),
    ("1+2=3 2+3=5 3+4=", "7"),
    ("5-2=3 6-3=3 7-4=", "3"),
    ("2*2=4 2*3=6 2*4=", "8"),
)


def attractor_stats(responses: list[str], n: int = 10) -> dict:
    """模式坍塌检测：不同 prompt 是否被吸到同一段文本上。

    只看"相邻 token 重复率"抓不到这个问题 —— 真实失效长这样：
    不管问什么都续出同一句 ``计算：96 * 66 - 65 = 62``（语料模板太重复导致的吸引子）。
    * ``distinct_rate``      —— 回答去重率（越低越坍塌）；
    * ``shared_ngram_rate``  —— 有多少条回答含有"在别的回答里也出现过"的 n-gram。
    """
    responses = [r for r in responses if r]
    if not responses:
        return {"distinct_rate": 0.0, "shared_ngram_rate": 0.0}
    grams = [set(x[i:i + n] for i in range(max(len(x) - n + 1, 0)))
             for x in responses]
    shared = 0
    for i, g in enumerate(grams):
        others: set[str] = set()
        for j, gj in enumerate(grams):
            if i != j:
                others |= gj
        shared += int(bool(g & others))
    return {"distinct_rate": round(len(set(responses)) / len(responses), 3),
            "shared_ngram_rate": round(shared / len(responses), 3)}


def attractor_probe(model, tok, max_new: int, temperature: float) -> dict:
    """吸引子检测：8 个日常 prompt + 8 个算式 prompt 的**全部**回答。"""
    responses = []
    for prompt in CHAT_PROMPTS:
        responses.append(generate(model, tok, prompt, max_new, temperature)[0])
    rng = np.random.default_rng(3)
    for _ in range(8):
        a, b = int(rng.integers(0, 500)), int(rng.integers(0, 500))
        responses.append(generate(model, tok, f"计算：{a} + {b} =", max_new, 0.0)[0])
    return attractor_stats(responses)


def load_model(ckpt: Path, dtype: str | None = None):
    cfg = load_config(str(ckpt))
    tok = tokenizer_from_config(cfg)
    model = model_from_config(tok, cfg) if cfg else TinyTransformer(tok.vocab_size)
    if dtype:
        model = TinyTransformer(model.vocab_size, model.d_model, model.n_layers,
                                model.n_heads, model.max_pos, seed=0,
                                d_mlp=model.d_mlp, attn_dim=model.attn_dim,
                                dtype=dtype)
    load_checkpoint(model, tok, str(ckpt))
    return model, tok


def generate(model: TinyTransformer, tok: CharTokenizer, prompt: str,
             max_new: int, temperature: float, seed: int = 0):
    """返回 (文本, 是否吐了 EOS, 长度)。用 generate_batch 以便观察 EOS 行为。"""
    ids = [tok.stoi[c] for c in prompt if c in tok.stoi]
    if not ids:
        return "", False, 0
    seqs, _ = model.generate_batch(ids, 1, max_new, temperature,
                                   np.random.default_rng(seed), tok.eos_id,
                                   model.max_pos,
                                   top_k=int(DECODE["top_k"]),
                                   no_repeat_ngram=int(DECODE["no_repeat_ngram"]))
    resp = seqs[0][len(ids):]
    eos = tok.eos_id in resp
    text = tok.decode([t for t in resp if t != tok.eos_id])
    return text, eos, len([t for t in resp if t != tok.eos_id])


def edit_ratio(a: str, b: str) -> float:
    """1 - 归一化编辑距离 (部分正确率)。"""
    if not a and not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b), 1)


def arith_battery(model, tok, task: str, n: int, max_new: int):
    """算术评测。

    注意要分三档看，否则会误判：
    * ``exact``  —— 输出恰好等于答案（要求模型**会停**，即答完吐 EOS）；
    * ``prefix`` —— 输出以答案开头（纯 LM 流式训练出来的模型通常只会做到这一步，
      因为它没被训练过"答完就停"）；
    * ``first``  —— 第一个字符就对（连答案首位都没命中就是完全没学会）。
    """
    rng = np.random.default_rng(0)
    exact = prefix = first = 0
    partial = 0.0
    lengths, eos_hits, samples = [], 0, []
    for _ in range(n):
        prompt, answer = make_prompt(rng, task)
        text, eos, ln = generate(model, tok, prompt, max_new, 0.0)
        ans = text.strip()
        exact += int(ans == answer)
        prefix += int(ans.startswith(answer))
        first += int(bool(ans) and ans[0] == answer[0])
        partial += edit_ratio(answer, ans)
        lengths.append(ln)
        eos_hits += int(eos)
        samples.append((prompt, ans, answer))
    return {"accuracy": exact / n, "prefix": prefix / n, "first": first / n,
            "partial": partial / n, "mean_len": float(np.mean(lengths)),
            "eos_rate": eos_hits / n, "samples": samples[:5]}


def corpus_arith_battery(model, tok, n: int, max_new: int, seed: int = 0):
    """按语料真实格式考算术: ``计算：{a} {op} {b} = {ans}``。

    训练语料里的数学是带前缀、带空格的（"计算：655 - 342 = 313"），用裸格式
    "5-2=" 去考等于换了分布，会把"没学会"和"没见过这种写法"混在一起。
    """
    rng = np.random.default_rng(seed)
    ops = ["+", "-", "*"]
    stats = {}
    for fmt in ("corpus", "bare"):
        exact = prefix = first = 0
        samples = []
        for _ in range(n):
            op = ops[int(rng.integers(0, 3))]
            a, b = int(rng.integers(0, 500)), int(rng.integers(0, 500))
            if op == "-":
                a, b = max(a, b), min(a, b)
            ans = {"+": a + b, "-": a - b, "*": a * b}[op]
            prompt = (f"计算：{a} {op} {b} =" if fmt == "corpus"
                      else f"{a}{op}{b}=")
            expected = f" {ans}" if fmt == "corpus" else str(ans)
            text, _eos, _ln = generate(model, tok, prompt, max_new, 0.0)
            got = text.split("\n")[0].rstrip()
            exact += int(got.strip() == expected.strip())
            prefix += int(got.lstrip().startswith(expected.strip()))
            first += int(bool(got.strip())
                         and got.strip()[0] == expected.strip()[0])
            samples.append((prompt, got[:18], expected.strip()))
        stats[fmt] = {"exact": exact / n, "prefix": prefix / n,
                      "first": first / n, "samples": samples[:4]}
    return stats


def word_problem_battery(model, tok, n: int, max_new: int):
    """应用题格式: ``…现在一共有多少个？答案：{ans} 个。``"""
    rng = np.random.default_rng(1)
    correct = 0
    samples = []
    for _ in range(n):
        a, b = int(rng.integers(0, 300)), int(rng.integers(0, 300))
        prompt = f"小明有 {a} 个苹果，又买了 {b} 个，现在一共有多少个？答案："
        answer = str(a + b)
        text, _eos, _ln = generate(model, tok, prompt, max_new, 0.0)
        correct += int(text.lstrip().startswith(answer))
        samples.append((prompt, text[:20], answer))
    return {"prefix": correct / n, "samples": samples[:4]}


def fewshot_battery(model, tok, max_new: int):
    correct, samples = 0, []
    for prompt, answer in FEWSHOT:
        text, _eos, _ln = generate(model, tok, prompt, max_new, 0.0)
        got = text.strip()
        correct += int(got == answer)
        samples.append((prompt, got, answer))
    return {"accuracy": correct / len(FEWSHOT), "samples": samples}


def chat_battery(model, tok, max_new: int, temperature: float):
    rows = []
    for prompt in CHAT_PROMPTS:
        text, eos, ln = generate(model, tok, prompt, max_new, temperature)
        cleaned = text.strip()
        echo = int(bool(cleaned) and (prompt in cleaned
                                      or cleaned[:2] == prompt[:2]))
        tokens = [tok.stoi[c] for c in text if c in tok.stoi]
        repeats = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
        rows.append({"prompt": prompt, "response": cleaned, "eos": eos,
                     "len": ln, "echo": echo,
                     "repeat_rate": repeats / max(len(tokens) - 1, 1),
                     "uniq_ratio": len(set(text)) / max(len(text), 1)})
    nonempty = sum(1 for r in rows if r["response"])
    return {
        "nonempty_rate": nonempty / len(rows),
        "eos_rate": sum(r["eos"] for r in rows) / len(rows),
        "echo_rate": sum(r["echo"] for r in rows) / len(rows),
        "repeat_rate": float(np.mean([r["repeat_rate"] for r in rows])),
        "uniq_char_ratio": float(np.mean([r["uniq_ratio"] for r in rows])),
        "mean_len": float(np.mean([r["len"] for r in rows])),
        "rows": rows,
    }


def tokenize_for(tok: CharTokenizer, text: str) -> tuple[np.ndarray, float]:
    usable = "".join(c for c in text if c in tok.stoi)
    ids = np.array([tok.stoi[c] for c in usable], dtype=np.int32)
    return ids, len(usable) / max(len(text), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="EnochModel1 能力评测")
    ap.add_argument("--models", nargs="+", required=True,
                    help="格式 标签:目录（目录可以是本地路径或 /tmp/M/...）")
    ap.add_argument("--pool", type=str, default="/tmp/enoch_pool.npz",
                    help="分布内验证文本来源（token 池）")
    ap.add_argument("--ood", type=str, nargs="*",
                    default=["docs/TRAINING.md", "docs/OPTIMIZATION.md"],
                    help="分布外文本文件")
    ap.add_argument("--val-chars", type=int, default=100_000)
    ap.add_argument("--arith-n", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--top-k", type=int, default=0,
                    help="采样时只保留概率最高的 k 个 token（0=关闭）")
    ap.add_argument("--no-repeat-ngram", type=int, default=0,
                    help="禁止重复出现过的 n-gram（0/1=关闭；实测可打破模板吸引子）")
    args = ap.parse_args()
    DECODE["top_k"] = args.top_k
    DECODE["no_repeat_ngram"] = args.no_repeat_ngram

    # 分布内 / 分布外文本（解码回字符，之后按每个模型自己的词表重编码）
    pool = np.load(args.pool, allow_pickle=False)
    pool_vocab = [str(c) for c in pool["vocab"]]
    in_text = "".join(pool_vocab[int(i)]
                      for i in pool["ids"][-args.val_chars:])
    ood_text = "\n".join(Path(p).read_text(encoding="utf-8")
                         for p in args.ood if Path(p).is_file())[:args.val_chars]

    results = []
    for spec in args.models:
        label, _, path = spec.partition(":")
        ckpt = Path(path)
        model, tok = load_model(ckpt)
        in_ids, in_cov = tokenize_for(tok, in_text)
        ood_ids, ood_cov = tokenize_for(tok, ood_text)
        length = min(128, model.max_pos)

        row = {
            "label": label, "path": str(ckpt),
            "params": int(sum(v.size for v in model.params.values())),
            "dtype": str(model.dtype), "vocab": tok.vocab_size,
            "d_model": model.d_model, "n_layers": model.n_layers,
            "max_pos": model.max_pos,
            "ppl_in": round(val_perplexity(model, tok, in_ids, length), 2)
            if in_ids.size > 1 else None,
            "ppl_ood": round(val_perplexity(model, tok, ood_ids, length), 2)
            if ood_ids.size > 1 else None,
            "in_coverage": round(in_cov, 3), "ood_coverage": round(ood_cov, 3),
            "easy": arith_battery(model, tok, "easy", args.arith_n, args.max_new),
            "hard": arith_battery(model, tok, "hard", args.arith_n // 2,
                                  args.max_new),
            "fewshot": fewshot_battery(model, tok, args.max_new),
            "corpus_arith": corpus_arith_battery(model, tok, args.arith_n // 2,
                                                 args.max_new),
            "word_problem": word_problem_battery(model, tok, args.arith_n // 4,
                                                 args.max_new),
            "attractor": attractor_probe(model, tok, args.max_new,
                                         args.temperature),
            "chat": chat_battery(model, tok, args.max_new, args.temperature),
        }
        completions = {}
        for c in COMPLETIONS:
            completions[c] = generate(model, tok, c, args.max_new, 0.0)[0]
        row["completions"] = completions
        results.append(row)
        print(f"[{label}] 完成", flush=True)

    print("\n## 能力总览\n")
    print("| 模型 | 参数 | 分布内PPL | 分布外PPL | 语料格式算术 首字/前缀/精确 | "
          "应用题前缀 | 少样本 | 对话非空 | 对话EOS | 回答去重率 | 跨回答n-gram重叠 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        c = r["corpus_arith"]["corpus"]
        print(f"| {r['label']} | {r['params']:,} | {r['ppl_in']} | {r['ppl_ood']} | "
              f"{c['first']:.2f} / {c['prefix']:.2f} / {c['exact']:.2f} | "
              f"{r['word_problem']['prefix']:.2f} | "
              f"{r['fewshot']['accuracy']:.2f} | {r['chat']['nonempty_rate']:.2f} | "
              f"{r['chat']['eos_rate']:.2f} | {r['attractor']['distinct_rate']:.2f} | "
              f"{r['attractor']['shared_ngram_rate']:.2f} |")

    for r in results:
        print(f"\n### {r['label']} 样例")
        print("- 算术(easy): " + "; ".join(
            f"`{p}`→`{g}`(标准 `{a}`)" for p, g, a in r["easy"]["samples"][:4]))
        print("- 算术(hard): " + "; ".join(
            f"`{p}`→`{g}`(标准 `{a}`)" for p, g, a in r["hard"]["samples"][:3]))
        print("- 语料格式算术: " + "; ".join(
            f"`{p}`→`{g}`(标准 `{a}`)"
            for p, g, a in r["corpus_arith"]["corpus"]["samples"]))
        print("- 应用题: " + "; ".join(
            f"→`{g}`(标准 `{a}`)" for _p, g, a in r["word_problem"]["samples"]))
        print("- 少样本: " + "; ".join(
            f"`{p}`→`{g}`(标准 `{a}`)" for p, g, a in r["fewshot"]["samples"][:3]))
        for row in r["chat"]["rows"]:
            print(f"- 对话 `{row['prompt']}` → `{row['response']}`")
        print("- 续写: " + "; ".join(f"`{k}`→`{v}`"
                                    for k, v in r["completions"].items()))

    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"\n已写入 {args.json}")


if __name__ == "__main__":
    main()
