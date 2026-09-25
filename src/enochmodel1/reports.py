"""
EnochModel1: CI 训练报告汇总 (enochmodel1.reports)
============================================

train 工作流每个 job 产出 ``train_report.json``（单副本模型有 tokens_per_s，
多副本平均出来的 ``avg-*`` 没有该字段），这里把它们统一成一张对比表并排名：

* 排名键 = (验证困惑度, -算术准确率)，越小越好；
* 所有可能缺失的字段都渲染成 ``n/a`` 而不是抛异常（曾因 None 传进 ``:,`` 直接崩）。
"""

from __future__ import annotations

import json
from pathlib import Path


def load_reports(root: str | Path) -> list[dict]:
    """递归扫描 root 下所有 train_report.json。"""
    reports: list[dict] = []
    root = Path(root)
    if not root.is_dir():
        return reports
    for path in sorted(root.rglob("train_report.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        data["_dir"] = str(path.parent)
        reports.append(data)
    return reports


def rank_key(r: dict) -> tuple[float, float]:
    """越小越好: (验证困惑度, -算术准确率)。缺字段排到最后。"""
    ppl = r.get("val_perplexity_after")
    acc = r.get("arith_accuracy") or 0.0
    try:
        ppl_v = float(ppl)
    except (TypeError, ValueError):
        ppl_v = float("inf")
    try:
        acc_v = float(acc)
    except (TypeError, ValueError):
        acc_v = 0.0
    return (ppl_v, -acc_v)


def pick_best(reports: list[dict]) -> dict | None:
    return min(reports, key=rank_key) if reports else None


def _num(value, spec: str = ",") -> str:
    """None/NaN -> n/a, 其余按 spec 格式化。"""
    if value is None:
        return "n/a"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return "n/a"


def markdown(reports: list[dict], best: dict | None = None,
             samples: bool = True) -> str:
    """渲染汇总表 (None 安全)。"""
    if best is None:
        best = pick_best(reports)
    lines = ["## 训练结果汇总", ""]
    lines.append("| 模型 | 参数 | 副本 | 轮数 | 已见 tokens | tokens/参数 | "
                 "tok/s | 验证困惑度(前→后) | 算术准确率 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(reports, key=rank_key):
        replicas = r.get("replicas")
        replica_txt = (str(len(replicas)) if isinstance(replicas, list)
                       else _num(r.get("workers"), "d"))
        lines.append(
            f"| `{r.get('tag') or r.get('config') or r.get('_dir')}` "
            f"| {_num(r.get('params'))} | {replica_txt} | {_num(r.get('rounds'), 'd')} "
            f"| {_num(r.get('tokens_seen'))} | {_num(r.get('tokens_per_param'), '.2f')} "
            f"| {_num(r.get('tokens_per_s'))} "
            f"| {_num(r.get('val_perplexity_before'), '.1f')} → "
            f"**{_num(r.get('val_perplexity_after'), '.1f')}** "
            f"| {_num(r.get('arith_accuracy'), '.3f')} |")
    if best:
        lines += ["", (f"**最佳**: `{best.get('tag')}` "
                       f"(ppl {_num(best.get('val_perplexity_after'), '.1f')}, "
                       f"acc {_num(best.get('arith_accuracy'), '.3f')})"), ""]
        if samples and best.get("samples"):
            lines.append("对话样例:")
            lines.append("")
            for p, out in best["samples"].items():
                lines.append(f"- `{p}` → `{out}`")
    return "\n".join(lines) + "\n"
