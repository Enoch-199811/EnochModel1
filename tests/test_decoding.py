"""解码约束 (top_k / no_repeat_ngram) 测试。

背景：能力评测发现模型有**模板吸引子** —— 不管问什么都续出同一句
（跨回答 10-gram 重叠率 50-62%）。这两个约束是解码层的解药：
* ``top_k`` 砍掉长尾低概率 token（吸引子往往由长尾拼出来）；
* ``no_repeat_ngram`` 直接禁止重复出现过的 n-gram。
关键约束：**默认值必须不改变原有行为**（否则历史结果不可复现）。
"""

from __future__ import annotations

import numpy as np

from enochmodel1.enoch import (
    CharTokenizer,
    TinyTransformer,
    banned_ngram_tokens,
    generate_text,
)


def make_model(**kw) -> TinyTransformer:
    kw.setdefault("vocab_size", 15)
    kw.setdefault("d_model", 16)
    kw.setdefault("n_layers", 1)
    kw.setdefault("n_heads", 4)
    kw.setdefault("max_pos", 32)
    return TinyTransformer(**kw)


def test_banned_ngram_tokens() -> None:
    empty = banned_ngram_tokens([1, 2], 1)
    assert empty.size == 0                       # n<=1 不禁
    assert list(banned_ngram_tokens([1, 2, 1], 2)) == [2]   # "1,2" 已出现 → 禁补 2
    assert banned_ngram_tokens([1, 2, 3], 2).size == 0      # 前缀 (3) 没出现过
    assert list(banned_ngram_tokens([1, 2, 1, 2], 3)) == [1]  # "1,2,1" 出现过


def test_defaults_do_not_change_behavior() -> None:
    """不传约束时必须与旧实现完全一致（同 seed 两次调用逐位相同）。"""
    model = make_model(seed=1)
    tok = CharTokenizer()
    a = generate_text(model, tok, "1+2=", 12, 0.8, np.random.default_rng(0))
    b = generate_text(model, tok, "1+2=", 12, 0.8, np.random.default_rng(0),
                      top_k=0, no_repeat_ngram=0)
    assert a == b


def test_top_k_1_equals_greedy() -> None:
    model = make_model(seed=2)
    tok = CharTokenizer()
    sampled = generate_text(model, tok, "1+2=", 12, 0.8,
                            np.random.default_rng(3), top_k=1)
    greedy = generate_text(model, tok, "1+2=", 12, 0.0, np.random.default_rng(3))
    assert sampled == greedy


def test_no_repeat_ngram_prevents_repetition() -> None:
    """把模型做成"只会吐同一个 token"，再验证约束确实打破复读。"""
    model = make_model(seed=4)
    for k in model.params:
        model.params[k] = np.zeros_like(model.params[k])
    model.params["b_out"][0] = 10.0        # token 0 永远是最大值
    tok = CharTokenizer()
    free = generate_text(model, tok, "1+2=", 10, 0.0, np.random.default_rng(0))
    assert len(set(free)) == 1             # 无约束：复读同一个字
    fixed = generate_text(model, tok, "1+2=", 10, 0.0, np.random.default_rng(0),
                          no_repeat_ngram=2)
    assert len(set(fixed)) > 1, f"约束应打破复读，得到 {fixed!r}"
    pairs = list(zip(fixed, fixed[1:]))
    assert len(pairs) == len(set(pairs)), "禁止重复 2-gram 后不应再出现重复对"


def test_constraints_compose_in_sampling() -> None:
    model = make_model(seed=5)
    tok = CharTokenizer()
    out = generate_text(model, tok, "1+2=", 16, 0.9, np.random.default_rng(1),
                        top_k=4, no_repeat_ngram=3)
    assert isinstance(out, str) and len(out) <= 16
