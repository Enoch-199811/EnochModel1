"""预训练流式分片语料测试。"""

from __future__ import annotations

import sys

import pytest

from enochmodel1 import pretrain
from enochmodel1.enoch import CharTokenizer


def _write_shards(tmp_path, texts: list[str]) -> str:
    d = tmp_path / "corpus"
    shards = d / "shards"
    shards.mkdir(parents=True)
    for i, text in enumerate(texts):
        (shards / f"shard_{i:05d}.txt").write_text(text, encoding="utf-8")
    return str(d)


def test_corpus_shards_file_and_dir(tmp_path) -> None:
    f = tmp_path / "single.txt"
    f.write_text("abc", encoding="utf-8")
    assert pretrain.corpus_shards(str(f)) == [str(f)]

    d = _write_shards(tmp_path, ["hello", "world", "你好"])
    shards = pretrain.corpus_shards(d)
    assert len(shards) == 3
    assert shards == sorted(shards)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        pretrain.corpus_shards(str(empty))

    assert pretrain.corpus_shards(None) == []


def test_allocate_shard_steps() -> None:
    paths = ["a", "b", "c"]
    assert pretrain.allocate_shard_steps(paths, 10) == [("a", 4), ("b", 3), ("c", 3)]
    assert pretrain.allocate_shard_steps(paths, 3) == [("a", 1), ("b", 1), ("c", 1)]
    assert pretrain.allocate_shard_steps(["a", "b", "c", "d", "e"], 2) == [
        ("a", 1), ("b", 1)]


def test_encode_ids_np() -> None:
    tok = CharTokenizer()
    text = "你好，世界！"
    tok.add(text)
    ids = pretrain.encode_ids_np(tok, text)
    assert list(ids) == tok.encode(text)
    assert ids.dtype.name == "int32"


def _run_pretrain(tmp_path, monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["enoch-pretrain"] + argv)
    pretrain.main()


def test_pretrain_interleave_sharded(tmp_path, monkeypatch) -> None:
    corpus_dir = _write_shards(tmp_path, ["0123456789+-*= " * 20, "你好世界。今天天气很好。"])
    out = tmp_path / "ckpt"
    _run_pretrain(tmp_path, monkeypatch, [
        "--corpus", corpus_dir,
        "--lm-steps", "2",
        "--task-steps", "2",
        "--out-dir", str(out),
        "--log-every", "1",
        "--eval-every", "99",
    ])
    assert (out / "model.npz").exists()
    assert (out / "config.json").exists()
    cfg = pretrain.load_config(str(out))
    assert "你" in cfg["vocab"]


def test_pretrain_lm_only_sharded(tmp_path, monkeypatch) -> None:
    corpus_dir = _write_shards(
        tmp_path, ["0123456789+-*= " * 20, "abcdefghijklmnopqrstuvwxyz" * 10])
    out = tmp_path / "ckpt2"
    _run_pretrain(tmp_path, monkeypatch, [
        "--corpus", corpus_dir,
        "--lm-steps", "2",
        "--no-interleave",
        "--task-steps", "0",
        "--out-dir", str(out),
        "--log-every", "1",
        "--eval-every", "99",
    ])
    assert (out / "model.npz").exists()
