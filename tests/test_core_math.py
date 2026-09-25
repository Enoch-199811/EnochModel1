"""基础数学工具: softmax / onehot / gelu / layer_norm / 编辑距离 / 评分。"""

from __future__ import annotations

import numpy as np
import pytest

from enochmodel1.enoch import (
    gelu,
    gelu_grad,
    layer_norm,
    layer_norm_backward,
    levenshtein,
    onehot,
    score_response,
    softmax,
)


def test_softmax_sums_to_one() -> None:
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    p = softmax(x, axis=-1)
    assert p.shape == x.shape
    np.testing.assert_allclose(p.sum(axis=-1), 1.0, rtol=1e-12)


def test_softmax_numerically_stable() -> None:
    x = np.array([[1e3, 1e3 - 1, -1e3]])
    p = softmax(x)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(), 1.0)


def test_onehot() -> None:
    targets = np.array([[0, 1, 2], [2, 0, 1]])
    oh = onehot(targets, vocab_size=3)
    assert oh.shape == (2, 3, 3)
    assert oh[0, 0, 0] == 1.0 and oh[1, 2, 1] == 1.0
    np.testing.assert_allclose(oh.sum(axis=-1), 1.0)


def test_gelu_grad_matches_numerical() -> None:
    x = np.linspace(-3.0, 3.0, 17)
    eps = 1e-5
    num = (gelu(x + eps) - gelu(x - eps)) / (2 * eps)
    np.testing.assert_allclose(gelu_grad(x), num, atol=1e-4)


def test_layer_norm_normalizes() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 5, 8))
    gamma = rng.normal(size=(8,))
    beta = rng.normal(size=(8,))
    y, (xhat, mu, _var) = layer_norm(x, gamma, beta)
    assert y.shape == x.shape
    np.testing.assert_allclose(mu, x.mean(axis=-1, keepdims=True))
    np.testing.assert_allclose(xhat.mean(axis=-1), 0.0, atol=1e-10)
    # 分母带 eps, 方差略小于 1
    np.testing.assert_allclose(xhat.var(axis=-1), 1.0, atol=1e-3)


def test_layer_norm_backward_matches_numerical() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 4, 6))
    gamma = rng.normal(size=(6,))
    beta = rng.normal(size=(6,))
    dy = rng.normal(size=(2, 4, 6))
    _, (xhat, _mu, var) = layer_norm(x, gamma, beta)
    dx, dgamma, dbeta = layer_norm_backward(dy, xhat, gamma, var)

    def loss(xv, gv, bv) -> float:
        yv, _ = layer_norm(xv, gv, bv)
        return float((yv * dy).sum())

    eps = 1e-5
    dx_num = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                xp, xm = x.copy(), x.copy()
                xp[i, j, k] += eps
                xm[i, j, k] -= eps
                dx_num[i, j, k] = (loss(xp, gamma, beta) - loss(xm, gamma, beta)) / (2 * eps)
    np.testing.assert_allclose(dx, dx_num, atol=1e-4)

    dgamma_num = np.zeros_like(gamma)
    dbeta_num = np.zeros_like(beta)
    for k in range(gamma.shape[0]):
        gp, gm = gamma.copy(), gamma.copy()
        gp[k] += eps
        gm[k] -= eps
        dgamma_num[k] = (loss(x, gp, beta) - loss(x, gm, beta)) / (2 * eps)
        bp, bm = beta.copy(), beta.copy()
        bp[k] += eps
        bm[k] -= eps
        dbeta_num[k] = (loss(x, gamma, bp) - loss(x, gamma, bm)) / (2 * eps)
    np.testing.assert_allclose(dgamma, dgamma_num, atol=1e-4)
    np.testing.assert_allclose(dbeta, dbeta_num, atol=1e-4)


def test_levenshtein() -> None:
    assert levenshtein("", "") == 0
    assert levenshtein("", "abc") == 3
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("123", "123") == 0


def test_score_response_exact() -> None:
    assert score_response("3", "3", "exact") == 1.0
    assert score_response("3", "4", "exact") == 0.0


def test_score_response_partial() -> None:
    assert score_response("3", "3", "partial") == 1.0
    assert score_response("13", "1", "partial") == 0.5
    assert score_response("48", "", "partial") == 0.0


def test_score_response_unknown_mode() -> None:
    with pytest.raises(ValueError):
        score_response("1", "1", "bogus")
