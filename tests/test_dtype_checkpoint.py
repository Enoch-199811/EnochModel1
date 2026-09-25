"""dtype (float64/float32) 与 checkpoint 存储格式测试。

* float32 档必须能正常前向/反向/训练 (省一半内存与体积的代价是精度);
* checkpoint 用 ``savez_compressed`` 保存, 且 config 里记录 ``d_mlp`` /
  ``attn_dim`` / ``dtype``, 保证剪枝后的窄模型能被三个入口原样重建;
* 旧 checkpoint (没有这些字段) 仍然可以加载。
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from enochmodel1.enoch import (
    Adam,
    CharTokenizer,
    TinyTransformer,
    load_checkpoint,
    load_config,
    model_from_config,
    save_checkpoint,
    token_loss,
    tokenizer_from_config,
)


def make_args(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def test_float32_forward_backward() -> None:
    model = TinyTransformer(15, 16, 1, 4, 16, seed=0, dtype="float32")
    assert model.params["W_e"].dtype == np.float32
    assert model.dtype == np.dtype("float32")
    X = np.random.default_rng(0).integers(0, 15, size=(2, 6)).astype(np.int64)
    logits, cache = model.forward(X)
    assert logits.dtype == np.float32
    model._set_cache(cache)
    targets = np.random.default_rng(1).integers(0, 15, size=(2, 6)).astype(np.int64)
    weights = np.ones((2, 6))
    loss, dlogits = token_loss(logits, targets, weights)
    assert np.isfinite(loss)
    assert dlogits.dtype == np.float32
    grads = model.backward(X, dlogits, 0)
    assert all(np.isfinite(g).all() for g in grads.values())
    Adam(model.params).step(grads, 3e-3, 1.0)
    assert all(np.isfinite(v).all() for v in model.params.values())


def test_float32_training_reduces_loss() -> None:
    from enochmodel1.enoch import lm_step

    tok = CharTokenizer()
    ids = np.array(tok.encode("1+2=3 4+5=9 " * 60), dtype=np.int32)
    model = TinyTransformer(tok.vocab_size, 16, 1, 4, 32, seed=0, dtype="float32")
    opt = Adam(model.params)
    args = make_args(lm_len=24, batch_prompts=4, entropy_beta=0.0, lr=1e-2,
                     grad_clip=1.0)
    rng = np.random.default_rng(0)
    first = lm_step(model, opt, tok, ids, rng, args)
    last = first
    for _ in range(30):
        last = lm_step(model, opt, tok, ids, rng, args)
    assert np.isfinite(last) and last < first


def test_checkpoint_records_shape_and_compresses(tmp_path) -> None:
    tok = CharTokenizer()
    tok.add("中文")
    model = TinyTransformer(tok.vocab_size, 16, 1, 4, 32, seed=0, d_mlp=20,
                            attn_dim=8, dtype="float32")
    out = tmp_path / "ckpt"
    save_checkpoint(model, tok, make_args(d_model=16), str(out))

    cfg = load_config(str(out))
    assert cfg["d_mlp"] == 20 and cfg["attn_dim"] == 8
    assert cfg["dtype"] == "float32"
    assert cfg["vocab"] == list(tok.chars)

    import zipfile

    with zipfile.ZipFile(out / "model.npz") as z:
        methods = {i.compress_type for i in z.infolist()}
    assert methods == {zipfile.ZIP_DEFLATED}   # savez_compressed 生效

    rebuilt = model_from_config(tokenizer_from_config(cfg), cfg)
    assert rebuilt.d_mlp == 20 and rebuilt.attn_dim == 8
    load_checkpoint(rebuilt, tok, str(out))
    X = np.random.default_rng(0).integers(0, tok.vocab_size, size=(1, 5)).astype(np.int64)
    a, _ = model.forward(X)
    b, _ = rebuilt.forward(X)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_checkpoint_dtype_cast_on_load(tmp_path) -> None:
    tok = CharTokenizer()
    model = TinyTransformer(tok.vocab_size, 16, 1, 4, 32, seed=0)
    out = tmp_path / "ckpt"
    save_checkpoint(model, tok, make_args(), str(out))

    small = TinyTransformer(tok.vocab_size, 16, 1, 4, 32, seed=1, dtype="float32")
    load_checkpoint(small, tok, str(out))
    assert all(v.dtype == np.float32 for v in small.params.values())
    np.testing.assert_allclose(small.params["W_e"], model.params["W_e"],
                               rtol=1e-6, atol=1e-6)


def test_legacy_config_without_new_fields(tmp_path) -> None:
    """旧 checkpoint: config 里没有 d_mlp/attn_dim/dtype, 也要能重建。"""
    tok = CharTokenizer()
    out = tmp_path / "legacy"
    out.mkdir()
    model = TinyTransformer(tok.vocab_size, 16, 1, 4, 32, seed=0)
    np.savez_compressed(out / "model.npz", **model.params)
    (out / "config.json").write_text(json.dumps(
        {"d_model": 16, "n_layers": 1, "n_heads": 4, "max_pos": 32,
         "vocab": list(tok.chars)}, ensure_ascii=False), encoding="utf-8")

    cfg = load_config(str(out))
    rebuilt = model_from_config(tokenizer_from_config(cfg), cfg)
    assert rebuilt.d_mlp == 4 * 16 and rebuilt.attn_dim == 16
    assert rebuilt.dtype == np.dtype("float64")
    load_checkpoint(rebuilt, tok, str(out))
    X = np.random.default_rng(0).integers(0, tok.vocab_size, size=(1, 5)).astype(np.int64)
    a, _ = model.forward(X)
    b, _ = rebuilt.forward(X)
    np.testing.assert_allclose(a, b, atol=1e-12)
