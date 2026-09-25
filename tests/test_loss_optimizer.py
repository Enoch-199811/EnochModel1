"""token_loss / Adam / SpeedTracker 测试。"""

from __future__ import annotations

import numpy as np
import pytest

from enochmodel1.enoch import Adam, SpeedTracker, token_loss


def test_token_loss_perfect_prediction() -> None:
    V = 5
    logits = np.zeros((2, 3, V))
    targets = np.array([[0, 1, 2], [3, 4, 0]])
    for b in range(2):
        for l in range(3):
            logits[b, l, targets[b, l]] = 20.0
    weights = np.ones((2, 3))
    loss, _ = token_loss(logits, targets, weights, beta=0.0)
    assert loss < 1e-6


def test_token_loss_zero_weights_ignored() -> None:
    V = 4
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2, 4, V))
    targets = rng.integers(0, V, size=(2, 4))
    weights = np.zeros((2, 4))
    loss, dlogits = token_loss(logits, targets, weights, beta=0.1)
    assert loss == 0.0
    np.testing.assert_allclose(dlogits, 0.0)


def test_token_loss_kl_zero_when_ref_identical() -> None:
    V = 6
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(1, 3, V))
    targets = rng.integers(0, V, size=(1, 3))
    weights = np.ones((1, 3))
    l0, _ = token_loss(logits, targets, weights, beta=0.0, kl_beta=0.5, ref_logits=logits)
    l1, _ = token_loss(logits, targets, weights, beta=0.0)
    np.testing.assert_allclose(l0, l1)


def test_token_loss_dlogits_matches_numerical() -> None:
    V = 4
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(2, 3, V))
    targets = rng.integers(0, V, size=(2, 3))
    weights = np.ones((2, 3))

    def f(z) -> float:
        loss, _ = token_loss(z, targets, weights, beta=0.0)
        return loss

    _loss, dlogits = token_loss(logits, targets, weights, beta=0.0)
    eps = 1e-5
    num = np.zeros_like(logits)
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            for k in range(logits.shape[2]):
                zp, zm = logits.copy(), logits.copy()
                zp[i, j, k] += eps
                zm[i, j, k] -= eps
                num[i, j, k] = (f(zp) - f(zm)) / (2 * eps)
    np.testing.assert_allclose(num, dlogits, atol=1e-4)


def test_adam_step_updates_params() -> None:
    rng = np.random.default_rng(0)
    params = {"w": rng.normal(size=(3, 3))}
    opt = Adam(params)
    grads = {"w": rng.normal(size=(3, 3))}
    before = params["w"].copy()
    opt.step(grads, lr=0.1, grad_clip=10.0)
    assert opt.t == 1
    assert params["w"].shape == before.shape
    assert not np.allclose(params["w"], before)
    assert set(opt.m.keys()) == {"w"}
    assert set(opt.v.keys()) == {"w"}


def test_adam_zero_grad_keeps_params() -> None:
    params = {"w": np.array([1.0, 2.0])}
    opt = Adam(params)
    opt.step({"w": np.zeros(2)}, lr=0.1, grad_clip=1.0)
    np.testing.assert_allclose(params["w"], [1.0, 2.0])


def test_adam_grad_clip_equivalent_to_scaled_grads() -> None:
    rng = np.random.default_rng(0)
    grads = {"w": rng.normal(size=(5,))}
    g_norm = float(np.linalg.norm(grads["w"]))

    w0 = rng.normal(size=(5,))
    params1 = {"w": w0.copy()}
    Adam(params1).step({k: v.copy() for k, v in grads.items()}, lr=0.1, grad_clip=g_norm * 0.5)

    params2 = {"w": w0.copy()}
    Adam(params2).step({k: v * 0.5 for k, v in grads.items()}, lr=0.1, grad_clip=1e9)
    np.testing.assert_allclose(params1["w"], params2["w"])


def test_speed_tracker() -> None:
    sp = SpeedTracker(ema=0.9, floor=1.0)
    assert sp.speed() == 1.0
    assert sp.ema_len == 0.0
    sp.update(4.0)
    assert sp.ema_len == 4.0
    assert sp.speed() == 0.25
    sp.update(8.0)
    assert sp.ema_len == pytest.approx(0.9 * 4.0 + 0.1 * 8.0)
    assert sp.speed() == pytest.approx(1.0 / sp.ema_len)


def test_speed_tracker_floor() -> None:
    sp = SpeedTracker(ema=0.5, floor=2.0)
    sp.update(1.0)
    assert sp.speed() == 0.5  # 1 / max(1.0, 2.0)
