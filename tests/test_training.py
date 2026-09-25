"""训练步骤 / 采样 / 评估测试。"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pytest

from enochmodel1.enoch import (
    Adam,
    CharTokenizer,
    SpeedTracker,
    TinyTransformer,
    evaluate,
    grow_vocab,
    lm_step,
    make_prompt,
    pretrain_step,
    rl_step,
    sample_group,
    check_gradients,
    show_examples,
)


def make_args(**over) -> argparse.Namespace:
    args = argparse.Namespace(
        task="easy",
        batch_prompts=8,
        group_size=4,
        max_new=24,
        temperature=0.8,
        score_mode="partial",
        entropy_beta=0.01,
        lr=3e-3,
        rl_lr=1e-3,
        grad_clip=1.0,
        kl_beta=0.2,
        lm_len=32,
        max_pos=64,
        verbose_pretrain=False,
    )
    for k, v in over.items():
        setattr(args, k, v)
    return args


def make_small_model() -> tuple[TinyTransformer, CharTokenizer]:
    tok = CharTokenizer()
    model = TinyTransformer(tok.vocab_size, d_model=16, n_layers=1,
                            n_heads=4, max_pos=64, seed=0)
    return model, tok


def test_make_prompt_answers_are_correct() -> None:
    rng = np.random.default_rng(7)
    for task in ("easy", "hard"):
        for _ in range(100):
            prompt, answer = make_prompt(rng, task)
            a, op, b = re.match(r"(\d+)([+\-*])(\d+)", prompt[:-1]).groups()
            expected = str({
                "+": int(a) + int(b),
                "-": int(a) - int(b),
                "*": int(a) * int(b),
            }[op])
            assert answer == expected
            if op == "-":
                assert int(a) >= int(b)


def test_make_prompt_unknown_task() -> None:
    with pytest.raises(ValueError):
        make_prompt(np.random.default_rng(0), "impossible")


def test_pretrain_step_reduces_loss() -> None:
    model, tok = make_small_model()
    opt = Adam(model.params)
    args = make_args()
    rng = np.random.default_rng(0)
    first = pretrain_step(model, opt, tok, rng, args)
    for _ in range(59):
        pretrain_step(model, opt, tok, rng, args)
    last = pretrain_step(model, opt, tok, rng, args)
    assert np.isfinite(first) and np.isfinite(last)
    assert last < first


def test_lm_step_reduces_loss() -> None:
    model, tok = make_small_model()
    opt = Adam(model.params)
    args = make_args(lm_len=32)
    corpus = "0123456789+-*= " * 20
    grow_vocab(model, opt, tok, corpus)
    corpus_ids = tok.encode(corpus)
    rng = np.random.default_rng(0)
    first = lm_step(model, opt, tok, corpus_ids, rng, args)
    for _ in range(59):
        lm_step(model, opt, tok, corpus_ids, rng, args)
    last = lm_step(model, opt, tok, corpus_ids, rng, args)
    assert np.isfinite(first) and np.isfinite(last)
    assert last < first


def test_rl_step_stats() -> None:
    model, tok = make_small_model()
    opt = Adam(model.params)
    speed = SpeedTracker()
    args = make_args(batch_prompts=4, group_size=4, max_new=16, kl_beta=0.0)
    rng = np.random.default_rng(0)
    ref_model = TinyTransformer(tok.vocab_size, 16, 1, 4, 64, seed=0)
    ref_model.params = {k: v.copy() for k, v in model.params.items()}
    stats = rl_step(model, opt, tok, rng, speed, ref_model, args)
    assert set(stats) == {"loss", "mean_score", "mean_len", "speed", "trunc_rate"}
    assert np.isfinite(stats["loss"])
    assert 0.0 <= stats["mean_score"] <= 1.0
    assert stats["mean_len"] >= 0.0
    assert stats["speed"] == pytest.approx(1.0)
    assert speed.ema_len > 0
    stats2 = rl_step(model, opt, tok, rng, speed, ref_model, args)
    assert np.isfinite(stats2["loss"])


def test_sample_group_fields() -> None:
    model, tok = make_small_model()
    args = make_args(group_size=2, max_new=8, temperature=0.0)
    rng = np.random.default_rng(0)
    group = sample_group(model, tok, "1+2=", "3", rng, args)
    assert len(group) == 2
    for s in group:
        assert s["prompt_ids"] == tok.encode("1+2=")
        assert set(s) == {
            "prompt_ids", "resp_ids", "logprobs", "score", "n_tokens",
            "answer", "truncated",
        }
        assert 0.0 <= s["score"] <= 1.0


def test_evaluate_returns_valid_metrics() -> None:
    model, tok = make_small_model()
    args = make_args(max_new=16)
    rng = np.random.default_rng(1)
    ev = evaluate(model, tok, rng, args, n=10)
    assert set(ev) == {"accuracy", "mean_len"}
    assert 0.0 <= ev["accuracy"] <= 1.0
    assert ev["mean_len"] >= 0.0


def test_show_examples_prints(capsys) -> None:
    model, tok = make_small_model()
    args = make_args()
    rng = np.random.default_rng(2)
    show_examples(model, tok, rng, args, n=2)
    assert "示例" in capsys.readouterr().out


def test_check_gradients_passes(capsys) -> None:
    """端到端反向传播数值校验 (失败时 check_gradients 会 sys.exit)。"""
    check_gradients(0)
    assert "[PASS]" in capsys.readouterr().out
