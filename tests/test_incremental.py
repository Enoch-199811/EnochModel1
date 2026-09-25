"""增量解码 (KV cache) 与全量前向的等价性测试。

这些用例把"记忆化不能改变行为"这条约束钉死在 CI 里:

* :func:`test_decode_step_matches_full_forward`  —— 单步增量 logits == 全量前向;
* :func:`test_generate_batch_matches_naive`      —— 与"每步重算整段前缀"的朴素
  参考实现逐 token 一致 (含采样, 即 RNG 消耗顺序一致);
* :func:`test_generate_batch_does_not_recompute_prefix` —— 结构上证明 prompt
  只前向一次 (没有重算前缀), 这是"记忆化"本身。
"""

from __future__ import annotations

import numpy as np
import pytest

from enochmodel1.enoch import CharTokenizer, KVCache, TinyTransformer, softmax


def make_model(**kw) -> TinyTransformer:
    kw.setdefault("vocab_size", 17)
    kw.setdefault("d_model", 16)
    kw.setdefault("n_layers", 2)
    kw.setdefault("n_heads", 4)
    kw.setdefault("max_pos", 32)
    return TinyTransformer(**kw)


def naive_generate(model: TinyTransformer, prompt_ids: list[int], group_size: int,
                   max_new: int, temperature: float, rng, eos_id: int,
                   max_pos: int):
    """优化前的实现: 每吐一个 token 都整段重算 (作为等价性基准)。"""
    W = min(len(prompt_ids) + max_new, max_pos)
    seqs = [list(prompt_ids) for _ in range(group_size)]
    done = [False] * group_size
    logprobs: list[list[float]] = [[] for _ in range(group_size)]
    for _ in range(max_new):
        X = np.full((group_size, W), 0, dtype=np.int64)
        for i, s in enumerate(seqs):
            X[i, :len(s)] = s
        logits, _ = model.forward(X)
        for i in range(group_size):
            if done[i]:
                continue
            pos = len(seqs[i]) - 1
            if temperature <= 0.0:
                tok = int(np.argmax(logits[i, pos]))
                logprobs[i].append(0.0)
            else:
                p = softmax(logits[i, pos] / temperature)
                p = p / p.sum()
                tok = int(rng.choice(model.vocab_size, p=p))
                logprobs[i].append(float(np.log(max(p[tok], 1e-12))))
            seqs[i].append(tok)
            if tok == eos_id:
                done[i] = True
        if all(done):
            break
    return seqs, logprobs


def test_decode_step_matches_full_forward() -> None:
    model = make_model(seed=1)
    X = np.random.default_rng(0).integers(0, 17, size=(2, 9)).astype(np.int64)
    logits, _ = model.forward(X)

    kv = KVCache(model, 2, 16)
    model.forward(X[:, :5], kv=kv, kv_offset=0)   # 预填充
    steps = [model.decode_step(X[:, p], np.full(2, p, dtype=np.int64), kv)
             for p in range(5, 9)]
    got = np.stack(steps, axis=1)                 # [B, 4, V]
    np.testing.assert_allclose(got, logits[:, 5:, :], atol=1e-10)


@pytest.mark.parametrize("temperature,group", [(0.0, 1), (0.0, 3), (0.8, 4)])
def test_generate_batch_matches_naive(temperature: float, group: int) -> None:
    model = make_model(seed=3)
    tok = CharTokenizer()
    prompt = tok.encode("1+2=")
    ref_seqs, ref_lp = naive_generate(
        model, prompt, group, 10, temperature, np.random.default_rng(7),
        tok.eos_id, model.max_pos)
    got_seqs, got_lp = model.generate_batch(
        prompt, group, 10, temperature, np.random.default_rng(7),
        tok.eos_id, model.max_pos)
    assert got_seqs == ref_seqs
    for a, b in zip(ref_lp, got_lp):
        np.testing.assert_allclose(a, b, atol=1e-10)


def test_generate_batch_does_not_recompute_prefix(monkeypatch) -> None:
    """结构断言: prompt 只前向一次, 之后每步只走增量解码。"""
    model = make_model(seed=2)
    tok = CharTokenizer()
    prompt = tok.encode("1+2=")
    calls = {"fwd": 0, "step": 0}
    orig_fwd, orig_step = model.forward, model.decode_step

    def fwd(*a, **k):
        calls["fwd"] += 1
        return orig_fwd(*a, **k)

    def step(*a, **k):
        calls["step"] += 1
        return orig_step(*a, **k)

    monkeypatch.setattr(model, "forward", fwd)
    monkeypatch.setattr(model, "decode_step", step)
    model.generate_batch(prompt, 2, 8, 0.0, np.random.default_rng(0),
                         tok.eos_id, model.max_pos)
    assert calls["fwd"] == 1        # 预填充只做一次 -> 前缀没有被重算
    assert 0 < calls["step"] <= 8


def test_generate_batch_caps_at_max_pos() -> None:
    model = make_model(max_pos=8)
    tok = CharTokenizer()
    prompt = tok.encode("1+")
    seqs, _ = model.generate_batch(prompt, 1, 20, 0.0, np.random.default_rng(0),
                                   tok.eos_id, model.max_pos)
    assert len(seqs[0]) <= model.max_pos


def test_kv_cache_store_continuous_and_per_row() -> None:
    model = make_model(seed=0)
    kv = KVCache(model, 2, 6)
    # layers * (K + V) * batch * heads * capacity * head_dim * 8 字节
    assert kv.nbytes() == 2 * 2 * 2 * model.n_heads * 6 * model.head_dim * 8
    kh = np.ones((2, model.n_heads, 3, model.head_dim))
    kv.store(0, 1, kh, kh * 2.0)
    np.testing.assert_array_equal(kv.K[0][:, :, 1:4, :], kh)
    np.testing.assert_array_equal(kv.V[0][:, :, 1:4, :], kh * 2.0)

    kh1 = np.full((2, model.n_heads, 1, model.head_dim), 5.0)
    kv.store(0, np.array([0, 5]), kh1, kh1)
    np.testing.assert_array_equal(kv.K[0][0, :, 0, :], kh1[0, :, 0, :])
    np.testing.assert_array_equal(kv.K[0][1, :, 5, :], kh1[1, :, 0, :])
