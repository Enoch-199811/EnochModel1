"""大规模语料构建器测试 (分片 / 数学正确性 / 对话 / wiki 清洗)。"""

from __future__ import annotations

import argparse
import random
import re

from enochmodel1 import build_corpus


def test_shard_writer_rotates(tmp_path) -> None:
    writer = build_corpus.ShardWriter(tmp_path, shard_chars=100)
    for i in range(50):
        writer.write(f"line {i}\n")
    info = writer.finish()
    assert info["total_chars"] == sum(f["chars"] for f in info["files"])
    assert info["num_shards"] >= 3
    assert (tmp_path / "shards").is_dir()
    shards = sorted((tmp_path / "shards").glob("*.txt"))
    assert len(shards) == info["num_shards"]
    # 每个分片不超过 shard_chars + 单次写入长度
    for s in shards:
        assert len(s.read_text(encoding="utf-8")) <= 100 + 20


def test_generate_math_answers_are_correct(tmp_path) -> None:
    writer = build_corpus.ShardWriter(tmp_path, shard_chars=10_000)
    written = build_corpus.generate_math(writer, 5000, random.Random(0))
    writer.finish()
    text = "".join(
        p.read_text(encoding="utf-8")
        for p in sorted((tmp_path / "shards").glob("*.txt"))
    )
    assert written >= 5000
    # 检查所有 "计算：a op b = ans" 行
    pat = re.compile(r"^计算：(\d+) ([+\-*]) (\d+) = (-?\d+)$")
    checked = 0
    for line in text.splitlines():
        m = pat.match(line)
        if not m:
            continue
        a, op, b, ans = m.groups()
        expected = {
            "+": int(a) + int(b),
            "-": int(a) - int(b),
            "*": int(a) * int(b),
        }[op]
        assert int(ans) == expected
        checked += 1
    assert checked > 0
    # 多步表达式与应用题也出现了
    assert "答案：" in text


def test_generate_dialogue(tmp_path) -> None:
    writer = build_corpus.ShardWriter(tmp_path, shard_chars=10_000)
    written = build_corpus.generate_dialogue(writer, 3000, random.Random(1))
    writer.finish()
    text = "".join(
        p.read_text(encoding="utf-8")
        for p in sorted((tmp_path / "shards").glob("*.txt"))
    )
    assert written >= 3000
    assert "你: " in text and "小诺: " in text
    assert "你好" in text


def test_strip_wiki() -> None:
    raw = (
        "{{Infobox}}'''标题'''\n"
        "这是一段[[链接|显示文字]]和<ref>注释</ref>。\n"
        "== 小节 ==\n"
        "<!-- 注释 -->正文内容。"
    )
    cleaned = build_corpus.strip_wiki(raw)
    assert "显示文字" in cleaned
    assert "注释" not in cleaned
    assert "{{" not in cleaned
    assert "正文内容" in cleaned


def test_category_budgets_scaling() -> None:
    args = argparse.Namespace(
        target_chars=1_000,
        code_chars=None,
        chinese_chars=None,
        math_chars=None,
        dialogue_chars=None,
    )
    b = build_corpus.category_budgets(args)
    assert b == {"code": 400, "math": 300, "dialogue": 300, "chinese": 0}

    args.code_chars = 700
    b2 = build_corpus.category_budgets(args)
    assert b2["code"] == 700
    assert b2["dialogue"] == 300


def test_parse_args_defaults() -> None:
    args = build_corpus.parse_args([])
    assert args.target_chars == build_corpus.DEFAULT_TARGET_CHARS
    assert args.shard_chars == build_corpus.DEFAULT_SHARD_CHARS
    assert args.code_chars is None
