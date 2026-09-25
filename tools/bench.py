"""EnochModel1 基准脚本 (优化前后共用, 只依赖公开 API)。

测量的负载就是三个入口真实跑的负载:

* ``decode_greedy``  贪心解码 48 token (chat / 算术问答);
* ``decode_group4``  temperature=0.7 采样 + group_size=4 (RL rollout);
* ``lm_step``        语料无监督一步 (enoch-pretrain 的热路径);
* ``rl_step``        一个 REINFORCE 步 (enoch-train 的热路径, 含 rollout);
* ``fwd_bwd``        单次 forward+backward (含梯度回传的纯计算部分)。

并记录峰值 RSS / 参数体积 / checkpoint 体积, 用于"降本"量化。

用法::

    .venv/bin/python tools/bench.py --tag baseline --out benchmarks/bench_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# 注意: 必须先导入 enochmodel1 (它会设置 BLAS 线程环境变量), 再导入 numpy,
# 否则 numpy 会以默认线程数提前初始化 BLAS。
from enochmodel1._paths import CHECKPOINTS_DIR, DATA_DIR, PROJECT_ROOT
from enochmodel1.enoch import (
    Adam,
    SpeedTracker,
    TinyTransformer,
    build_lm_batch,
    lm_step,
    load_checkpoint,
    load_config,
    rl_step,
    token_loss,
    tokenizer_from_config,
)

DECODE_PROMPT = "1+2="
LM_BATCH, LM_LEN, LM_STEPS = 8, 48, 20
RL_BATCH, RL_GROUP, RL_MAX_NEW, RL_STEPS = 4, 4, 24, 3
FWD_BWD_STEPS = 20

def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def blas_name() -> str:
    try:
        cfg = np.__config__.CONFIG  # type: ignore[attr-defined]
        deps = cfg.get("Build Dependencies", {}).get("blas", {})
        return str(deps.get("name", "unknown"))
    except Exception:  # noqa: BLE001  (纯报告用, 拿不到就返回 unknown)
        return "unknown"


class Timer:
    """对无参函数计时, 返回秒数列表。"""

    def __init__(self, repeats: int) -> None:
        self.repeats = repeats
        self.times: list[float] = []

    def run(self, fn) -> float:
        best = float("inf")
        for _ in range(self.repeats):
            t = time.perf_counter()
            fn()
            dt = time.perf_counter() - t
            self.times.append(dt)
            best = min(best, dt)
        return best

    @property
    def median(self) -> float:
        return statistics.median(self.times) if self.times else float("nan")


def build_args() -> argparse.Namespace:
    """给 lm_step / rl_step 用的参数命名空间 (与 CLI 默认值一致)。"""
    import argparse as _ap

    return _ap.Namespace(
        task="easy", lm_len=LM_LEN, batch_prompts=LM_BATCH, group_size=RL_GROUP,
        max_new=RL_MAX_NEW, temperature=0.8, max_pos=128, score_mode="partial",
        verbose_pretrain=False, entropy_beta=0.01, kl_beta=0.2,
        lr=3e-3, rl_lr=1e-3, grad_clip=1.0, eval_n=20,
    )


def load_model(ckpt: Path, dtype: str = "float64"):
    cfg = load_config(str(ckpt))
    tokenizer = tokenizer_from_config(cfg)
    if cfg:
        d_model = cfg.get("d_model", 64)
        n_layers = cfg.get("n_layers", 2)
        n_heads = cfg.get("n_heads", 4)
        max_pos = cfg.get("max_pos", 128)
        d_mlp = cfg.get("d_mlp")
    else:  # pragma: no cover
        d_model, n_layers, n_heads, max_pos, d_mlp = 64, 2, 4, 128, None
    model = TinyTransformer(tokenizer.vocab_size, d_model, n_layers, n_heads,
                            max_pos, seed=0, d_mlp=d_mlp, dtype=dtype)
    load_checkpoint(model, tokenizer, str(ckpt))
    return model, tokenizer


def corpus_ids_for(tokenizer, path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    ids = [tokenizer.stoi[c] for c in text if c in tokenizer.stoi]
    assert ids, f"{path} 里没有词表内的字符"
    return np.asarray(ids, dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description="EnochModel1 基准")
    ap.add_argument("--tag", type=str, default="run", help="结果标签")
    ap.add_argument("--out", type=str, default=None, help="结果 JSON 路径")
    ap.add_argument("--repeats", type=int, default=3, help="重复次数 (取最优)")
    ap.add_argument("--checkpoint", type=str,
                    default=str(CHECKPOINTS_DIR / "chat"))
    ap.add_argument("--corpus", type=str, default=str(DATA_DIR / "corpus.txt"))
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                    help="模型参数 dtype (float32 省内存/更快, 精度略降)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    model, tokenizer = load_model(ckpt, args.dtype)
    gargs = build_args()
    corpus_ids = corpus_ids_for(tokenizer, Path(args.corpus))
    rng = np.random.default_rng(0)
    opt = Adam(model.params)
    eos_id, max_pos = tokenizer.eos_id, model.max_pos
    prompt_ids = tokenizer.encode(DECODE_PROMPT)
    # 解码负载用一段语料 prompt 更接近真实续写: 避免模型一句就吐 EOS,
    # 导致"48 token 解码"实际只跑了一两步而虚高。
    corpus_text = Path(args.corpus).read_text(encoding="utf-8")
    seed_text = "".join(c for c in corpus_text[:600]
                        if c in tokenizer.stoi)[:32]
    long_prompt = tokenizer.encode(seed_text) or prompt_ids

    def count_new(seqs: list[list[int]], plen: int) -> int:
        return sum(len(s) - plen for s in seqs)

    result: dict = {
        "tag": args.tag,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": blas_name(),
        "cpu_threads": os.cpu_count(),
        "checkpoint": str(ckpt),
        "model": {"d_model": model.d_model, "n_layers": model.n_layers,
                  "n_heads": model.n_heads, "max_pos": model.max_pos,
                  "vocab_size": int(model.vocab_size),
                  "params": int(sum(v.size for v in model.params.values())),
                  "dtype": str(next(iter(model.params.values())).dtype)},
        "workloads": {},
    }
    param_bytes = int(sum(v.nbytes for v in model.params.values()))

    rss_before = peak_rss_mib()

    # --- 1. 贪心解码 48 token (语料 prompt) ---
    def decode_greedy() -> None:
        model.generate_batch(long_prompt, 1, 48, 0.0, np.random.default_rng(0),
                             eos_id, max_pos)

    s0, _ = model.generate_batch(long_prompt, 1, 48, 0.0,
                                 np.random.default_rng(0), eos_id, max_pos)
    n_greedy = count_new(s0, len(long_prompt))
    t = Timer(args.repeats)
    t.run(decode_greedy)
    result["workloads"]["decode_greedy_48"] = {
        "median_s": t.median, "best_s": min(t.times),
        "tokens_generated": n_greedy,
        "tokens_per_s": n_greedy / t.median if t.median else None,
        "ms_per_token": 1000 * t.median / n_greedy if n_greedy else None}

    # --- 2. 采样 + group_size=4 (rollout 形态) ---
    def decode_group4() -> None:
        model.generate_batch(long_prompt, 4, RL_MAX_NEW, 0.7,
                             np.random.default_rng(0), eos_id, max_pos)

    s0, _ = model.generate_batch(long_prompt, 4, RL_MAX_NEW, 0.7,
                                 np.random.default_rng(0), eos_id, max_pos)
    n_group = count_new(s0, len(long_prompt))
    t = Timer(args.repeats)
    t.run(decode_group4)
    result["workloads"]["decode_group4_24"] = {
        "median_s": t.median, "best_s": min(t.times),
        "tokens_generated": n_group,
        "tokens_per_s": n_group / t.median if t.median else None,
        "ms_per_token": 1000 * t.median / n_group if n_group else None}

    # --- 3. 语料 LM 步 ---
    def lm_block() -> None:
        for _ in range(LM_STEPS):
            lm_step(model, opt, tokenizer, corpus_ids, rng, gargs)

    t = Timer(args.repeats)
    t.run(lm_block)
    result["workloads"]["lm_step"] = {
        "median_s": t.median, "best_s": min(t.times),
        "steps_per_s": LM_STEPS / t.median if t.median else None,
        "ms_per_step": 1000 * t.median / LM_STEPS}

    # --- 4. RL 步 (含 rollout) ---
    def rl_block() -> None:
        speed = SpeedTracker(gargs.speed_ema if hasattr(gargs, "speed_ema") else 0.9,
                             1.0)
        for _ in range(RL_STEPS):
            rl_step(model, opt, tokenizer, rng, speed, None, gargs)

    t = Timer(args.repeats)
    t.run(rl_block)
    result["workloads"]["rl_step"] = {
        "median_s": t.median, "best_s": min(t.times),
        "steps_per_s": RL_STEPS / t.median if t.median else None,
        "ms_per_step": 1000 * t.median / RL_STEPS}

    # --- 5. 纯 forward+backward ---
    gg = np.random.default_rng(1)
    X, targets, weights = build_lm_batch(
        list(corpus_ids[:20000]), 64, 8, gg, tokenizer)

    def fwd_bwd() -> None:
        for _ in range(FWD_BWD_STEPS):
            logits, cache = model.forward(X)
            model._set_cache(cache)
            _, dlogits = token_loss(logits, targets, weights, beta=0.01)
            model.backward(X, dlogits, tokenizer.pad_id)

    t = Timer(args.repeats)
    t.run(fwd_bwd)
    result["workloads"]["fwd_bwd"] = {
        "median_s": t.median, "best_s": min(t.times),
        "ms_per_step": 1000 * t.median / FWD_BWD_STEPS}

    # --- 体积 / 内存 ---
    result["memory"] = {
        "peak_rss_mib": peak_rss_mib(),
        "rss_before_mib": rss_before,
        "param_bytes": param_bytes,
        "ckpt_npz_bytes": int((ckpt / "model.npz").stat().st_size)
        if (ckpt / "model.npz").is_file() else None,
    }
    result["total_s"] = sum(w["median_s"] for w in result["workloads"].values())

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"结果已写入 {out}")
    print(text)


if __name__ == "__main__":
    main()
    sys.exit(0)
