"""入口模块的纯函数测试: chat 辅助 / 语料读取 / 语料生成。"""

from __future__ import annotations

import json
import sys

from enochmodel1 import build_corpus, chat, pretrain


def test_is_arithmetic() -> None:
    assert chat.is_arithmetic("1+2")
    assert chat.is_arithmetic("1+2=")
    assert chat.is_arithmetic(" 9 - 3 ")
    assert chat.is_arithmetic("2*8=")
    assert not chat.is_arithmetic("你好")
    assert not chat.is_arithmetic("abc")
    assert not chat.is_arithmetic("")


def test_normalize_arith() -> None:
    assert chat.normalize_arith("1+2") == "1+2="
    assert chat.normalize_arith(" 3 * 4 ") == "3 * 4="
    assert chat.normalize_arith("1+2=") == "1+2="
    assert chat.normalize_arith("你好") == "你好"
    assert chat.normalize_arith("  ") == ""


def test_parse_score() -> None:
    assert chat.parse_score("8") == 8.0
    assert chat.parse_score(" 10 ") == 10.0
    assert chat.parse_score("0") == 0.0
    assert chat.parse_score("11") is None
    assert chat.parse_score("-1") is None
    assert chat.parse_score("abc") is None
    assert chat.parse_score("") is None


def test_memory_roundtrip_and_corrupt_line(tmp_path) -> None:
    path = str(tmp_path / "mem.jsonl")
    entries = [{"prompt": "1+2", "response": "3", "expected": None,
                "score": 8, "ts": 1.0}]
    chat.save_memory(path, entries)
    assert chat.load_memory(path) == entries
    with open(path, "a", encoding="utf-8") as f:
        f.write("{bad json}\n")
    loaded = chat.load_memory(path)
    assert len(loaded) == 1
    assert loaded[0] == entries[0]


def test_load_corpus(tmp_path) -> None:
    p = tmp_path / "c.txt"
    p.write_text("你好世界", encoding="utf-8")
    assert pretrain.load_corpus(str(p)) == "你好世界"


def test_build_corpus_generates_mixed_content(tmp_path, monkeypatch) -> None:
    out = tmp_path / "corpus"
    monkeypatch.setattr(sys, "argv", [
        "enoch-build-corpus",
        "--target-chars", "8000",
        "--code-chars", "0",
        "--chinese-chars", "0",
        "--math-chars", "4000",
        "--dialogue-chars", "4000",
        "--shard-chars", "3000",
        "--out-dir", str(out),
    ])
    build_corpus.main()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_chars"] >= 8000
    assert manifest["num_shards"] >= 2
    assert manifest["actual"]["math"] >= 4000
    assert manifest["actual"]["dialogue"] >= 4000
    text = "".join(
        p.read_text(encoding="utf-8")
        for p in sorted((out / "shards").glob("*.txt"))
    )
    assert "你好" in text
    assert any(ch.isdigit() for ch in text)
    assert any("\u4e00" <= ch <= "\u9fff" for ch in text)
