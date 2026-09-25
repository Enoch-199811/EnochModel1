"""TinyTransformer 前向 / 反向 / 采样 / 动态扩展测试。"""

from __future__ import annotations

import numpy as np
import pytest

from enochmodel1.enoch import CharTokenizer, TinyTransformer, generate_text, token_loss


def make_model(**kw) -> TinyTransformer:
    kw.setdefault("vocab_size", 15)
    kw.setdefault("d_model", 16)
    kw.setdefault("n_layers", 2)
    kw.setdefault("n_heads", 4)
    kw.setdefault("max_pos", 32)
    return TinyTransformer(**kw)


def test_forward_shape() -> None:
    model = make_model()
    X = np.zeros((3, 10), dtype=np.int64)
    logits, cache = model.forward(X)
    assert logits.shape == (3, 10, 15)
    assert "X" in cache


def test_backward_shapes_and_keys() -> None:
    model = make_model()
    rng = np.random.default_rng(0)
    X = rng.integers(0, 15, size=(2, 8)).astype(np.int64)
    targets = rng.integers(0, 15, size=(2, 8)).astype(np.int64)
    weights = np.ones((2, 8))
    logits, cache = model.forward(X)
    model._set_cache(cache)
    _loss, dlogits = token_loss(logits, targets, weights)
    grads = model.backward(X, dlogits, model.params["W_e"].shape[0] - 1)
    assert set(grads.keys()) == set(model.params.keys())
    for k in grads:
        assert grads[k].shape == model.params[k].shape


def test_backward_requires_forward() -> None:
    model = make_model()
    X = np.zeros((1, 4), dtype=np.int64)
    dlogits = np.zeros((1, 4, 15))
    with pytest.raises(RuntimeError):
        model.backward(X, dlogits, 0)


def test_seed_reproducibility() -> None:
    m1 = make_model(seed=42)
    m2 = make_model(seed=42)
    for k in m1.params:
        np.testing.assert_array_equal(m1.params[k], m2.params[k])


def test_greedy_generation_deterministic_and_bounded() -> None:
    model = make_model(seed=0)
    tok = CharTokenizer()
    prompt = [tok.stoi[c] for c in "1+2="]
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(999)
    seqs1, _ = model.generate_batch(prompt, 1, 12, 0.0, rng1, tok.eos_id, model.max_pos)
    seqs2, _ = model.generate_batch(prompt, 1, 12, 0.0, rng2, tok.eos_id, model.max_pos)
    assert seqs1 == seqs2
    assert len(seqs1[0]) <= len(prompt) + 12
    assert seqs1[0][-1] == tok.eos_id or len(seqs1[0]) == len(prompt) + 12


def test_generate_text_returns_no_eos() -> None:
    model = make_model(seed=0)
    tok = CharTokenizer()
    rng = np.random.default_rng(0)
    out = generate_text(model, tok, "1+2=", 16, 0.0, rng)
    assert len(out) <= 16
    assert tok.eos_id not in tok.encode(out)


def test_add_vocab_grows_params() -> None:
    model = make_model()
    v0 = model.vocab_size
    model.add_vocab(3)
    assert model.vocab_size == v0 + 3
    assert model.params["W_e"].shape == (v0 + 3, model.d_model)
    assert model.params["W_out"].shape == (model.d_model, v0 + 3)
    assert model.params["b_out"].shape == (v0 + 3,)
    np.testing.assert_array_equal(model.params["W_e"][-3:], 0.0)


def test_extend_pos() -> None:
    model = make_model(max_pos=16)
    model.extend_pos(40)
    assert model.max_pos == 40
    assert model.params["W_p"].shape == (40, model.d_model)
    np.testing.assert_array_equal(model.params["W_p"][16:], 0.0)
    model.extend_pos(20)  # 不缩小
    assert model.max_pos == 40


def test_d_model_not_divisible_raises() -> None:
    with pytest.raises(AssertionError):
        TinyTransformer(vocab_size=15, d_model=10, n_layers=1, n_heads=4, max_pos=16)
