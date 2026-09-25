"""入口模块命令行参数解析。"""

from __future__ import annotations

import sys

from enochmodel1 import chat, main, pretrain


def test_main_parse_args(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["enoch-train", "--steps", "10", "--d-model", "32"])
    args = main.parse_args()
    assert args.steps == 10
    assert args.d_model == 32
    assert args.score_mode == "partial"


def test_pretrain_parse_args(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["enoch-pretrain", "--task", "hard", "--lm-steps", "5"])
    args = pretrain.parse_args()
    assert args.task == "hard"
    assert args.lm_steps == 5
    assert args.interleave is True


def test_chat_parse_args(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["enoch-chat", "--temperature", "0.3", "--memory", "/tmp/m.jsonl"])
    args = chat.parse_args()
    assert args.temperature == 0.3
    assert args.memory == "/tmp/m.jsonl"
