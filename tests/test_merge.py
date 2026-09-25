"""权重平均 (enochmodel1.merge) 测试。"""

from __future__ import annotations

import numpy as np
import pytest

from enochmodel1.enoch import TinyTransformer
from enochmodel1.merge import average_models, check_compatible


def make(vocab: int = 13, d_model: int = 16, seed: int = 0) -> TinyTransformer:
    return TinyTransformer(vocab, d_model, 2, 4, 24, seed=seed)


def test_average_is_elementwise_mean() -> None:
    a, b = make(seed=1), make(seed=2)
    avg = average_models([a, b])
    for k in a.params:
        np.testing.assert_allclose(avg.params[k],
                                   (a.params[k] + b.params[k]) / 2, rtol=0, atol=1e-12)


def test_average_keeps_structure_and_does_not_mutate_inputs() -> None:
    a, b = make(seed=1), make(seed=2)
    before = a.params["W_e"].copy()
    avg = average_models([a, b])
    assert avg.d_model == a.d_model and avg.n_layers == a.n_layers
    assert avg.attn_dim == a.attn_dim and avg.d_mlp == a.d_mlp
    np.testing.assert_array_equal(a.params["W_e"], before)   # 输入未被改动
    assert avg.dtype == a.dtype


def test_average_single_model_is_identity() -> None:
    a = make(seed=3)
    avg = average_models([a])
    for k in a.params:
        np.testing.assert_allclose(avg.params[k], a.params[k], atol=1e-12)


def test_shape_mismatch_raises() -> None:
    a = make(d_model=16, seed=1)
    b = make(d_model=32, seed=1)
    with pytest.raises(ValueError):
        check_compatible([a, b])
    with pytest.raises(ValueError):
        average_models([a, b])


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        average_models([])


def test_averaged_model_can_forward_and_generate() -> None:
    a, b = make(seed=4), make(seed=5)
    avg = average_models([a, b])
    X = np.zeros((1, 4), dtype=np.int64)
    logits, _ = avg.forward(X)
    assert logits.shape == (1, 4, 13)
    seqs, _ = avg.generate_batch([0, 1], 2, 4, 0.0, np.random.default_rng(0),
                                 12, avg.max_pos)
    assert len(seqs) == 2
