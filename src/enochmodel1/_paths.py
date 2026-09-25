"""项目路径工具。

源码位于 ``src/enochmodel1/``, 向上两级即为项目根目录。所有默认的
数据 / checkpoint 路径都相对于项目根目录解析, 这样通过 ``uv`` 安装成
控制台命令后, 无论从哪个目录启动都能找到 ``data/`` 和 ``checkpoints/``。
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def resolve(path: str | None) -> str | None:
    """相对路径按项目根目录解析; 绝对路径原样返回; None 原样返回。"""
    if path is None:
        return None
    p = Path(path)
    return str(p) if p.is_absolute() else str(PROJECT_ROOT / p)
