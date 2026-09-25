"""汇总 GitHub Actions 训练作业的报告 (tools/collect_ci_reports.py)。

train 工作流每个配置产出 ``train_report.json`` (由 tools/parallel_pretrain.py 写),
这里把它们读成一张对比表, 选出**验证困惑度最低 (同分看算术准确率更高)**的那个,
并把最佳模型目录路径打到 ``--best-file`` 供后续归档步骤使用。
"""

from __future__ import annotations

# 允许直接用 python tools/xxx.py 运行 (不必先 uv sync / 激活 venv)
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import sys
from pathlib import Path

# 先导入本包 (它在 numpy 之前把 BLAS 线程设为 1), 再导入 numpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enochmodel1.reports import (  # noqa: E402
    load_reports,
    markdown,
    pick_best,
)


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
    best = pick_best(reports)
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
