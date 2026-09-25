"""结构化剪枝测试 (enochmodel1.prune)。

关注三件事:
1. 剪枝是"真的变小": 参数矩阵形状与总量都下降, 不是置零伪剪枝;
2. 剪枝是"精确的": 保留的头/隐藏单元的参数逐位不变, 比例 0 时完全等价;
3. 剪枝后的 checkpoint 能被 ``model_from_config`` 原样重建并加载。
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from enochmodel1.enoch import (
    CharTokenizer,
    TinyTransformer,
    load_checkpoint,
    load_config,
    model_from_config,
    save_checkpoint,
    tokenizer_from_config,
)
from enochmodel1.prune import (
    _keep_indices,
    head_importance,
    mlp_importance,
    prune_model,
)


def make_model(**kw) -> TinyTransformer:
    kw.setdefault("vocab_size", 19)
    kw.setdefault("d_model", 16)
    kw.setdefault("n_layers", 2)
    kw.setdefault("n_heads", 4)
    kw.setdefault("max_pos", 24)
    return TinyTransformer(**kw)


def n_params(model: TinyTransformer) -> int:
    return sum(int(v.size) for v in model.params.values())


def test_importance_shapes() -> None:
    model = make_model()
    assert mlp_importance(model, 0).shape == (model.d_mlp,)
    assert head_importance(model, 0).shape == (model.n_heads,)


def test_keep_indices_always_keeps_one() -> None:
    assert list(_keep_indices(np.arange(4.0), 0.99)) == [3]
    assert sorted(_keep_indices(np.array([3.0, 1.0, 2.0]), 1.0 / 3.0)) == [0, 2]
    assert list(_keep_indices(np.arange(4.0), 0.0)) == [0, 1, 2, 3]


def test_mlp_prune_shrinks_params() -> None:
    model = make_model(seed=0)
    pruned = prune_model(model, mlp_ratio=0.5)
    assert pruned.d_mlp == model.d_mlp // 2
    assert pruned.attn_dim == model.attn_dim
    assert pruned.n_heads == model.n_heads
    assert n_params(pruned) < n_params(model)
    assert pruned.params["W_10"].shape == (model.d_model, pruned.d_mlp)
    assert pruned.params["W_20"].shape == (pruned.d_mlp, model.d_model)
    assert pruned.params["b_10"].shape == (pruned.d_mlp,)
    X = np.random.default_rng(0).integers(0, 19, size=(2, 5)).astype(np.int64)
    logits, _ = pruned.forward(X)
    assert logits.shape == (2, 5, 19)


def test_head_prune_is_exact_on_kept_heads() -> None:
    model = make_model(seed=1)
    dh = model.head_dim
    # 比例 0 = 逐位相同
    same = prune_model(model, 0.0, 0.0)
    for k in model.params:
        np.testing.assert_array_equal(same.params[k], model.params[k])
    # 去掉 1 个头: 保留头对应块逐位相同, 窄出来的维度必须真的没了
    pruned = prune_model(model, 0.0, 0.25)
    assert pruned.n_heads == 3
    assert pruned.attn_dim == 3 * dh
    blocks = (slice(0, dh), slice(dh, 2 * dh), slice(2 * dh, 3 * dh),
              slice(3 * dh, 4 * dh))
    for l in range(model.n_layers):
        assert pruned.params[f"W_q{l}"].shape == (model.d_model, 3 * dh)
        assert pruned.params[f"W_o{l}"].shape == (3 * dh, model.d_model)
        for block in np.split(pruned.params[f"W_q{l}"], 3, axis=1):
            assert any(np.array_equal(block, model.params[f"W_q{l}"][:, s])
                       for s in blocks)


def test_attn_dim_must_divide_heads() -> None:
    with pytest.raises(AssertionError):
        make_model(d_model=16, n_heads=5)   # attn_dim 不能被 n_heads 整除


def test_pruned_checkpoint_roundtrip(tmp_path) -> None:
    tok = CharTokenizer()
    tok.add("中文")
    model = TinyTransformer(tok.vocab_size, 16, 2, 4, 24, seed=0)
    pruned = prune_model(model, mlp_ratio=0.5, head_ratio=0.25)

    out = tmp_path / "pruned"
    save_checkpoint(pruned, tok, argparse.Namespace(d_model=16), str(out))
    cfg = load_config(str(out))
    assert cfg["d_mlp"] == pruned.d_mlp
    assert cfg["attn_dim"] == pruned.attn_dim
    assert cfg["n_heads"] == pruned.n_heads
    assert cfg["vocab"] == list(tok.chars)

    rebuilt = model_from_config(tokenizer_from_config(cfg), cfg)
    assert rebuilt.d_mlp == pruned.d_mlp and rebuilt.attn_dim == pruned.attn_dim
    load_checkpoint(rebuilt, tok, str(out))
    X = np.random.default_rng(0).integers(0, tok.vocab_size, size=(1, 6)).astype(np.int64)
    a, _ = pruned.forward(X)
    b, _ = rebuilt.forward(X)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_prune_then_generate_works() -> None:
    tok = CharTokenizer()
    model = TinyTransformer(tok.vocab_size, 16, 2, 4, 24, seed=0)
    pruned = prune_model(model, mlp_ratio=0.5, head_ratio=0.25)
    prompt = tok.encode("1+2=")
    seqs, lps = pruned.generate_batch(prompt, 2, 6, 0.0,
                                      np.random.default_rng(0), tok.eos_id,
                                      pruned.max_pos)
    assert len(seqs) == 2 and len(lps) == 2
