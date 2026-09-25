"""CI 环境探测 (tools/ci_probe.py)。

在 GitHub runner 上打印 CPU/BLAS/线程配置, 并实测**单进程**与**多进程**下
TinyTransformer 一步训练 (forward+backward) 的耗时, 用来判断"云端训练为什么慢"。

不依赖语料池: 自己合成一段 token 流, 只测计算本身。

用法::

    uv run python tools/ci_probe.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

# 先导入本包 (它会在 numpy 之前把 BLAS 线程设为 1), 再导入 numpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from enochmodel1.enoch import (  # noqa: E402
    Adam,
    CharTokenizer,
    TinyTransformer,
    lm_step,
)

BATCH, LM_LEN, STEPS = 8, 128, 5
CONFIGS = {
    "d64-L2": {"d_model": 64, "n_layers": 2, "n_heads": 4, "max_pos": 128},
    "d128-L3": {"d_model": 128, "n_layers": 3, "n_heads": 8, "max_pos": 192},
}


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def make_ids(tokenizer: CharTokenizer, n: int = 400_000) -> np.ndarray:
    text = "你好，今天天气不错。1+2=3 4+5=9 吃饭了吗？" * (n // 20 + 1)
    return np.array([tokenizer.stoi[c] for c in text if c in tokenizer.stoi],
                    dtype=np.int32)[:n]


def run_steps(cfg: dict, ids: np.ndarray, tokenizer: CharTokenizer,
              steps: int) -> float:
    import argparse

    model = TinyTransformer(tokenizer.vocab_size, dtype="float32", **cfg)
    opt = Adam(model.params)
    args = argparse.Namespace(lm_len=LM_LEN, batch_prompts=BATCH,
                              entropy_beta=0.0, lr=3e-3, grad_clip=1.0)
    rng = np.random.default_rng(0)
    lm_step(model, opt, tokenizer, ids, rng, args)      # 预热
    t0 = time.perf_counter()
    for _ in range(steps):
        lm_step(model, opt, tokenizer, ids, rng, args)
    return time.perf_counter() - t0


def _worker(cfg: dict, ids: np.ndarray, steps: int, out: list) -> None:
    tokenizer = CharTokenizer()
    tokenizer.add("你好，今天天气不错。1+2=3 4+5=9 吃饭了吗？")
    out.append(run_steps(cfg, ids, tokenizer, steps))


def main() -> None:
    tokenizer = CharTokenizer()
    tokenizer.add("你好，今天天气不错。1+2=3 4+5=9 吃饭了吗？")
    ids = make_ids(tokenizer)
    info = {
        "cpu": cpu_model(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "os_cpu_count": os.cpu_count(),
        "mem_total_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "env": {k: os.environ.get(k) for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "ENOCH_BLAS_THREADS")},
        "batch": BATCH, "lm_len": LM_LEN, "steps": STEPS,
    }
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)

    for name, cfg in CONFIGS.items():
        dt = run_steps(cfg, ids, tokenizer, STEPS)
        tok_per_s = STEPS * BATCH * LM_LEN / dt
        print(f"[单进程] {name}: {dt / STEPS * 1000:.1f} ms/步 | "
              f"{tok_per_s:,.0f} tok/s", flush=True)

    # 4 进程 (模拟 CI 里的 4 路数据并行)
    ctx = mp.get_context("fork")
    cfg = CONFIGS["d128-L3"]
    for procs in (2, 4):
        mgr_out: list = []
        t0 = time.perf_counter()
        ps = [ctx.Process(target=_worker, args=(cfg, ids, STEPS, mgr_out))
              for _ in range(procs)]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        dt = time.perf_counter() - t0
        tok_per_s = procs * STEPS * BATCH * LM_LEN / dt
        print(f"[{procs} 进程] d128-L3 合计: {tok_per_s:,.0f} tok/s "
              f"(每进程 {tok_per_s / procs:,.0f})", flush=True)


if __name__ == "__main__":
    main()
