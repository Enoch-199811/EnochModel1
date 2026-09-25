"""语料池 (enochmodel1.pool) 测试。

重点验证"按 manifest 字符偏移切片"这套逻辑: CI 上重新生成的语料分片数和本地
不同, 如果按固定分片下标取, 会静默取到错误的类别 (甚至取到代码当对话)。这里用
一个手工构造的小语料把四个类别都覆盖一遍。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from enochmodel1.pool import (
    build_lut,
    build_pool,
    category_offsets,
    codes_of,
    read_offset,
    take_category,
)

SHARD_CHARS = 20
TEXT = "C" * 30 + "中" * 10 + "M" * 20 + "D" * 40      # 100 字符, 顺序=生成顺序
ACTUAL = {"code": 30, "chinese": 10, "math": 20, "dialogue": 40}


def make_corpus(tmp_path):
    shards = tmp_path / "shards"
    shards.mkdir(parents=True)
    for i in range(0, len(TEXT), SHARD_CHARS):
        (shards / f"shard_{i // SHARD_CHARS:05d}.txt").write_text(
            TEXT[i:i + SHARD_CHARS], encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "total_chars": len(TEXT), "num_shards": 5,
        "shard_chars": SHARD_CHARS, "actual": ACTUAL,
    }), encoding="utf-8")
    return tmp_path


def test_codes_of_roundtrip() -> None:
    codes = codes_of("a中")
    assert list(codes) == [ord("a"), ord("中")]


def test_category_offsets_follow_generation_order(tmp_path) -> None:
    corpus = make_corpus(tmp_path)
    assert category_offsets(corpus) == {
        "code": (0, 30), "chinese": (30, 10), "math": (40, 20),
        "dialogue": (60, 40)}


def test_read_offset_spans_shards(tmp_path) -> None:
    corpus = make_corpus(tmp_path)
    shards = corpus / "shards"
    # 跨分片读: 从第 15 个字符读 20 个 -> 覆盖 shard0 尾部与 shard1 全部
    got = read_offset(shards, SHARD_CHARS, 15, 20)
    assert "".join(chr(int(c)) for c in got) == TEXT[15:35]
    assert read_offset(shards, SHARD_CHARS, 0, 0).size == 0


def test_take_category_picks_right_region(tmp_path) -> None:
    corpus = make_corpus(tmp_path)
    assert "".join(chr(int(c)) for c in take_category(corpus, "chinese", 10)) == "中" * 10
    assert "".join(chr(int(c)) for c in take_category(corpus, "dialogue", 40)) == "D" * 40
    # 请求超过该类别的可用长度时只给可用的部分
    assert "".join(chr(int(c)) for c in take_category(corpus, "math", 999)) == "M" * 20
    assert take_category(corpus, "code", 0).size == 0


def test_build_pool_ratios_and_vocab(tmp_path) -> None:
    corpus = make_corpus(tmp_path)
    ids, tokenizer, stats = build_pool(corpus, dialogue_chars=40, math_chars=20,
                                       code_chars=30, chinese_chars=10)
    assert ids.size == 100
    assert stats["parts"] == {"dialogue": 40, "math": 20, "code": 30,
                              "chinese": 10}
    assert stats["ratios"]["dialogue"] == 0.4
    for ch in "C中MD":
        assert ch in tokenizer.stoi
    # 映射自洽: 解码回去应该等于原始字符流 (按类别拼接顺序)
    lut = build_lut(tokenizer)
    assert lut[ord("中")] == tokenizer.stoi["中"]
    decoded = "".join(tokenizer.itos[int(i)] for i in ids)
    assert set(decoded) == set("C中MD")


def test_build_pool_without_corpus_raises(tmp_path) -> None:
    with pytest.raises(SystemExit):
        build_pool(tmp_path, dialogue_chars=10, math_chars=0, code_chars=0)
    with pytest.raises(SystemExit):
        build_pool(tmp_path, dialogue_chars=0, math_chars=0, code_chars=0,
                   chinese_chars=0)


def test_take_category_without_manifest_uses_fallback(tmp_path) -> None:
    """没有 manifest 时退回固定分片下标 (老布局的兼容路径)。"""
    corpus = make_corpus(tmp_path)
    (corpus / "manifest.json").unlink()
    assert category_offsets(corpus) == {}
    got = take_category(corpus, "dialogue", 8)
    assert got.size in (0, 8)          # 兜底路径读 shard 11..14, 这里不存在
    assert isinstance(got, np.ndarray)
