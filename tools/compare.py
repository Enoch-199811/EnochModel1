"""把若干份 bench JSON 汇总成对比表 (Markdown)。

用法::

    .venv/bin/python tools/compare.py benchmarks/bench_baseline.json \
        benchmarks/bench_optimized.json --out benchmarks/bench_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 工作负载 -> (展示单位, 数值键, 越大越好?)
WORKLOADS = [
    ("decode_greedy_48", "ms/token(贪心解码)", "ms_per_token", False),
    ("decode_group4_24", "ms/token(4路采样)", "ms_per_token", False),
    ("lm_step", "ms/步(语料LM)", "ms_per_step", False),
    ("rl_step", "ms/步(RL含rollout)", "ms_per_step", False),
    ("fwd_bwd", "ms/步(fwd+bwd)", "ms_per_step", False),
]


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="EnochModel1 基准对比")
    ap.add_argument("paths", nargs="+", help="bench JSON 路径, 第一份为基线")
    ap.add_argument("--out", type=str, default=None, help="Markdown 输出路径")
    args = ap.parse_args()

    runs = [load(p) for p in args.paths]
    lines: list[str] = []
    lines.append("# EnochModel1 基准对比\n")
    lines.append("环境: " + " | ".join(
        f"{r['tag']}: python {r['python']}, numpy {r['numpy']}, blas {r['blas']}, "
        f"模型 dtype {r['model']['dtype']}, 参数 {r['model']['params']:,}"
        for r in runs) + "\n")

    head = "| 负载 | " + " | ".join(r["tag"] for r in runs) + " | 加速比 |"
    sep = "| --- | " + " | ".join("---" for _ in runs) + " | --- |"
    lines += [head, sep]
    for key, label, field, _bigger in WORKLOADS:
        cells = []
        vals = []
        for r in runs:
            w = r["workloads"][key]
            v = w.get(field)
            vals.append(v)
            cells.append("n/a" if v is None else f"{v:.2f}")
        speed = "n/a"
        if vals[0] and all(vals[1:]):
            speed = " / ".join(f"**{vals[0] / v:.2f}x**" for v in vals[1:])
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {speed} |")

    lines.append("")
    lines.append("| 资源 | " + " | ".join(r["tag"] for r in runs) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in runs) + " |")
    for key, label, field in [
        ("peak_rss_mib", "峰值 RSS (MiB)", "peak_rss_mib"),
        ("param_bytes", "参数体积 (B)", "param_bytes"),
        ("ckpt_npz_bytes", "checkpoint 体积 (B)", "ckpt_npz_bytes"),
    ]:
        cells = [f"{r['memory'][field]:,}" if isinstance(r["memory"][field], int)
                 else f"{r['memory'][field]:.1f}" for r in runs]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    cells = [f"{r['total_s']:.2f}s" for r in runs]
    lines.append("| 全负载总耗时 | " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
