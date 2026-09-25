"""项目根目录路径解析测试。"""

from __future__ import annotations

from pathlib import Path

from enochmodel1._paths import CHECKPOINTS_DIR, DATA_DIR, PROJECT_ROOT, resolve


def test_project_root_points_to_repo() -> None:
    assert PROJECT_ROOT.name == "EnochModel1"
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert CHECKPOINTS_DIR == PROJECT_ROOT / "checkpoints"


def test_resolve_relative_to_project_root() -> None:
    assert resolve("data/corpus.txt") == str(PROJECT_ROOT / "data" / "corpus.txt")
    assert resolve("checkpoints/pretrain") == str(PROJECT_ROOT / "checkpoints" / "pretrain")


def test_resolve_absolute_and_none() -> None:
    assert resolve("/tmp/x") == "/tmp/x"
    assert resolve(None) is None
    assert resolve(str(PROJECT_ROOT / "x")) == str(PROJECT_ROOT / "x")


def test_paths_exist_on_disk() -> None:
    assert Path.cwd().name == PROJECT_ROOT.name
    assert DATA_DIR.exists()
