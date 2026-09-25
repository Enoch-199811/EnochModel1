"""汇总 GitHub Actions 训练作业的报告 (tools/collect_ci_reports.py)。

train 工作流每个配置产出 ``train_report.json`` (由 tools/parallel_pretrain.py 写),
这里把它们读成一张对比表, 选出**验证困惑度最低 (同分看算术准确率更高)**的那个,
并把最佳模型目录路径打到 ``--best-file`` 供后续归档步骤使用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_reports(root: Path) -> list[dict]:
    reports = []
    for path in sorted(root.rglob("train_report.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        data["_dir"] = str(path.parent)
        reports.append(data)
    return reports


def rank_key(r: dict) -> tuple[float, float]:
    """越小越好: (验证困惑度, -算术准确率)。"""
    ppl = r.get("val_perplexity_after")
    acc = r.get("arith_accuracy") or 0.0
    return (float(ppl) if ppl is not None else float("inf"), -float(acc))


def markdown(reports: list[dict], best: dict | None) -> str:
    lines = ["## 训练结果汇总", ""]
    lines.append("| 配置 | 参数 | workers | 轮数 | 已见 tokens | tokens/参数 | "
                 "tok/s | 验证困惑度(前→后) | 算术准确率 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(reports, key=rank_key):
        lines.append(
            f"| `{r.get('tag') or r.get('config')}` | {r.get('params', 0):,} | "
            f"{r.get('workers')} | {r.get('rounds')} | "
            f"{(r.get('tokens_seen') or 0) / 1e6:.2f}M | "
            f"{r.get('tokens_per_param')} | {r.get('tokens_per_s'):,} | "
            f"{r.get('val_perplexity_before')} → **{r.get('val_perplexity_after')}** | "
            f"{r.get('arith_accuracy')} |")
    if best:
        lines += ["", f"**最佳**: `{best.get('tag')}` "
                      f"(ppl {best.get('val_perplexity_after')}, "
                      f"acc {best.get('arith_accuracy')})", ""]
        samples = best.get("samples") or {}
        if samples:
            lines.append("对话样例:")
            lines.append("")
            for p, out in samples.items():
                lines.append(f"- `{p}` → `{out}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总 CI 训练报告")
    ap.add_argument("--root", type=str, default="artifacts")
    ap.add_argument("--out", type=str, default="reports/latest.json")
    ap.add_argument("--summary", type=str, default=None,
                    help="markdown 追加到的文件 (GITHUB_STEP_SUMMARY)")
    ap.add_argument("--best-file", type=str, default=None,
                    help="把最佳模型目录写到这个文件")
    args = ap.parse_args()

    reports = load_reports(Path(args.root))
    best = min(reports, key=rank_key) if reports else None
    md = markdown(reports, best)
    print(md)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"reports": reports,
                               "best": best.get("tag") if best else None},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(md)
    if args.best_file:
        Path(args.best_file).write_text(best["_dir"] if best else "",
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
