"""checkpoint 保存 / 加载 / 词表恢复 / 动态扩展同步测试。"""

from __future__ import annotations

import argparse

import numpy as np

from enochmodel1.enoch import (
    Adam,
    CharTokenizer,
    TinyTransformer,
    grow_max_pos,
    grow_vocab,
    load_checkpoint,
    load_config,
    save_checkpoint,
    tokenizer_from_config,
)


def make_args(**over) -> argparse.Namespace:
    args = argparse.Namespace(d_model=16, n_layers=1, n_heads=4, max_pos=64, seed=0)
    for k, v in over.items():
        setattr(args, k, v)
    return args


def test_save_load_roundtrip(tmp_path) -> None:
    tok = CharTokenizer()
    model = TinyTransformer(tok.vocab_size, 16, 1, 4, 64, seed=0)
    save_checkpoint(model, tok, make_args(), str(tmp_path))
    assert (tmp_path / "model.npz").exists()
    assert (tmp_path / "config.json").exists()

    cfg = load_config(str(tmp_path))
    assert cfg["vocab"] == list(tok.chars)
    assert cfg["d_model"] == 16

    tok2 = tokenizer_from_config(cfg)
    assert tok2.chars == tok.chars

    model2 = TinyTransformer(tok2.vocab_size, 16, 1, 4, 64, seed=1)
    load_checkpoint(model2, tok2, str(tmp_path))
    for k in model.params:
        np.testing.assert_array_equal(model2.params[k], model.params[k])


def test_load_config_missing_returns_none(tmp_path) -> None:
    assert load_config(str(tmp_path)) is None


def test_tokenizer_from_config_fallback() -> None:
    tok = tokenizer_from_config(None)
    assert tok.stoi["0"] == 0
    assert tok.eos_id == tok.vocab_size - 1


def test_load_checkpoint_smaller_ckpt_prefix_copy(tmp_path) -> None:
    tok = CharTokenizer()
    model = TinyTransformer(tok.vocab_size, 16, 1, 4, 64, seed=0)
    save_checkpoint(model, tok, make_args(), str(tmp_path))

    # 恢复前词表已扩大 (新增中文), 优化器同步扩展
    opt = Adam(model.params)
    assert grow_vocab(model, opt, tok, "中文") == 2

    model2 = TinyTransformer(tok.vocab_size, 16, 1, 4, 64, seed=5)
    new_emb_before = model2.params["W_e"][-2:].copy()
    load_checkpoint(model2, tok, str(tmp_path))
    # 前缀来自 checkpoint, 新增部分保持 model2 原值
    np.testing.assert_array_equal(model2.params["W_e"][:-2], model.params["W_e"][:-2])
    np.testing.assert_array_equal(model2.params["W_e"][-2:], new_emb_before)


def test_grow_vocab_syncs_optimizer() -> None:
    tok = CharTokenizer()
    model = TinyTransformer(tok.vocab_size, 16, 1, 4, 64, seed=0)
    opt = Adam(model.params)
    assert grow_vocab(model, opt, tok, "中文") == 2
    assert model.params["W_e"].shape[0] == tok.vocab_size
    assert opt.m["W_e"].shape == model.params["W_e"].shape
    assert opt.v["W_e"].shape == model.params["W_e"].shape
    assert opt.m["W_out"].shape == model.params["W_out"].shape
    assert opt.m["b_out"].shape == model.params["b_out"].shape
    np.testing.assert_array_equal(model.params["W_e"][-2:], 0.0)


def test_grow_max_pos_syncs_optimizer() -> None:
    tok = CharTokenizer()
    model = TinyTransformer(tok.vocab_size, 16, 1, 4, 32, seed=0)
    opt = Adam(model.params)
    grow_max_pos(model, opt, 64)
    assert model.max_pos == 64
    assert model.params["W_p"].shape == (64, 16)
    assert opt.m["W_p"].shape == (64, 16)
    assert opt.v["W_p"].shape == (64, 16)
