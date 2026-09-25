"""CI 报告汇总 (enochmodel1.reports) 测试。

曾经的真实事故: 多副本平均出来的报告没有 tokens_per_s, 直接 format 抛
TypeError 让整个 publish job 挂掉 —— 这里把 None 安全钉死。
"""

from __future__ import annotations

import json

from enochmodel1.reports import load_reports, markdown, pick_best, rank_key


def test_rank_key_prefers_lower_perplexity() -> None:
    a = {"val_perplexity_after": 10.0, "arith_accuracy": 0.1}
    b = {"val_perplexity_after": 5.0, "arith_accuracy": 0.0}
    assert rank_key(b) < rank_key(a)
    # 困惑度相同看准确率
    c = {"val_perplexity_after": 5.0, "arith_accuracy": 0.9}
    assert rank_key(c) < rank_key(b)


def test_rank_key_handles_missing_fields() -> None:
    assert rank_key({})[0] == float("inf")
    assert rank_key({"val_perplexity_after": None})[0] == float("inf")


def test_markdown_is_none_safe() -> None:
    """avg-* 报告缺 tokens_per_s / rounds 时不能崩。"""
    reports = [
        {"tag": "d64-r0", "params": 200_000, "workers": 1, "rounds": 192,
         "tokens_seen": 1_000_000, "tokens_per_param": 5.0,
         "tokens_per_s": 10_900, "val_perplexity_before": 626.1,
         "val_perplexity_after": 2.68, "arith_accuracy": 0.0,
         "samples": {"1+2=": "3"}},
        {"tag": "avg-d64", "params": 200_000, "replicas": ["a", "b"],
         "val_perplexity_after": 7.5},          # 只有困惑度
    ]
    md = markdown(reports, samples=False)
    assert "avg-d64" in md and "n/a" in md
    assert "| 模型 |" in md
    assert pick_best(reports)["tag"] == "d64-r0"     # 2.68 < 7.5


def test_load_reports_and_missing_root(tmp_path) -> None:
    d = tmp_path / "ckpt-x-r0"
    d.mkdir()
    (d / "train_report.json").write_text(json.dumps({"tag": "x"}),
                                         encoding="utf-8")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "train_report.json").write_text("{not json",
                                                           encoding="utf-8")
    reports = load_reports(tmp_path)
    assert len(reports) == 1 and reports[0]["tag"] == "x"
    assert reports[0]["_dir"].endswith("ckpt-x-r0")
    assert load_reports(tmp_path / "nope") == []
