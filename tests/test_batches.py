"""批次构造: 预训练 / RL / LM 窗口。"""

from __future__ import annotations

import numpy as np

from enochmodel1.enoch import (
    CharTokenizer,
    _fill_batch,
    build_lm_batch,
    build_pretrain_batch,
    build_rl_batch,
)


def test_build_pretrain_batch_positions() -> None:
    tok = CharTokenizer()
    X, targets, weights = build_pretrain_batch([("1+2=", "3")], tok)
    assert X.shape == (1, 6)
    np.testing.assert_array_equal(X[0], tok.encode("1+2=3") + [tok.eos_id])
    # "=" 之后预测 "3", "3" 之后预测 eos
    assert targets[0, 3] == tok.stoi["3"]
    assert targets[0, 4] == tok.eos_id
    np.testing.assert_array_equal(weights[0], [0, 0, 0, 1, 1, 0])


def test_build_pretrain_batch_verbose() -> None:
    tok = CharTokenizer()
    X, _targets, _weights = build_pretrain_batch([("1+2=", "3")], tok, verbose=True)
    np.testing.assert_array_equal(X[0], tok.encode("1+2=33") + [tok.eos_id])


def test_build_rl_batch_advantage_weights() -> None:
    tok = CharTokenizer()
    samples = [
        {"prompt_ids": tok.encode("1+2="), "resp_ids": tok.encode("3") + [tok.eos_id], "adv": 0.5},
        {"prompt_ids": tok.encode("2+3="), "resp_ids": tok.encode("5") + [tok.eos_id], "adv": -0.3},
    ]
    X, targets, weights = build_rl_batch(samples, tok)
    assert X.shape == (2, 6)
    for i, adv in enumerate([0.5, -0.3]):
        np.testing.assert_array_equal(weights[i, 3:5], adv)
        np.testing.assert_array_equal(weights[i, :3], 0.0)
        np.testing.assert_array_equal(weights[i, 5:], 0.0)


def test_build_lm_batch_windows() -> None:
    rng = np.random.default_rng(0)
    tok = CharTokenizer()
    corpus_ids = tok.encode("0123456789")
    X, targets, weights = build_lm_batch(corpus_ids, length=5, batch=4, rng=rng, tokenizer=tok)
    assert X.shape == (4, 5)
    assert targets.shape == (4, 5)
    for i in range(4):
        np.testing.assert_array_equal(weights[i, :5], 1.0)
        np.testing.assert_array_equal(targets[i, :4], X[i, 1:5])


def test_fill_batch_padding() -> None:
    tok = CharTokenizer()
    samples = [
        (tok.encode("1+2="), tok.encode("3") + [tok.eos_id], 1.0),
        (tok.encode("1+2="), tok.encode("45") + [tok.eos_id], 1.0),
    ]
    X, _targets, weights = _fill_batch(samples, tok)
    assert X.shape == (2, 7)  # 最长 4 + 3
    assert X[0, -1] == tok.pad_id
    assert weights[0, -1] == 0.0
