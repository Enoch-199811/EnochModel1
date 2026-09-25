"""
EnochModel1: 训练语料池 (enochmodel1.pool)
====================================

把 `data/corpus/` 的分片拼成一个**按新配比重排**的 token 池:

* 默认 **对话优先** (`dialogue 24M / math 10M / code 6M`), 因为"日常智能"要的是聊天;
* 类别切片按 `manifest.json` 里的字符偏移计算, 因此**不管语料怎么生成、切了几个
  分片**都能取到正确的对话/数学/代码区段 (CI 上重新生成的语料同样适用);
* 字符→id 用"按 Unicode 码点建查找表"完成, 30M 字符只需几秒;
* 输出 ``.npz``: ``ids``(int32) + ``vocab``(字符表); 池子尾部 ``--val-chars``
  个字符留作验证集。

CLI::

    uv run enoch-pool --out /tmp/pool.npz                     # 对话优先默认配比
    uv run enoch-pool --dialogue-chars 25000000 --code-chars 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ._paths import DATA_DIR
from .enoch import CharTokenizer

DEFAULT_CORPUS_DIR = DATA_DIR / "corpus"
DEFAULT_SHARD_CHARS = 64 * 1024 * 1024
# 语料生成顺序 (见 build_corpus.main)
CATEGORY_ORDER = ("code", "chinese", "math", "dialogue")
# 没有 manifest 时的兜底分片下标 (本地 15 分片布局)
FALLBACK_SHARDS = {
    "dialogue": (11, 12, 13, 14),
    "math": (7, 8, 9),
    "code": (0, 1, 2),
}


def codes_of(text: str) -> np.ndarray:
    """文本 -> Unicode 码点数组 (C 级 utf-32 编码, 比逐字符快两个数量级)。"""
    return np.frombuffer(text.encode("utf-32-le"), dtype="<u4")


def _manifest_of(corpus_dir: Path) -> dict:
    path = Path(corpus_dir) / "manifest.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def category_offsets(corpus_dir: Path) -> dict[str, tuple[int, int]]:
    """各类别在"字符流"里的 (起始偏移, 长度); 依赖 manifest 的 actual 统计。"""
    actual = _manifest_of(corpus_dir).get("actual") or {}
    offsets: dict[str, tuple[int, int]] = {}
    pos = 0
    for key in CATEGORY_ORDER:
        n = int(actual.get(key, 0))
        if n > 0:
            offsets[key] = (pos, n)
        pos += n
    return offsets


def read_offset(shards_dir: Path, shard_chars: int, offset: int,
                chars: int) -> np.ndarray:
    """从分片流的第 offset 个字符开始读 chars 个字符 (跨分片自动拼接)。"""
    if chars <= 0:
        return np.empty(0, dtype="<u4")
    parts: list[np.ndarray] = []
    pos, remaining = offset, chars
    cache: dict[int, str] = {}
    while remaining > 0:
        idx = pos // shard_chars
        inner = pos % shard_chars
        path = Path(shards_dir) / f"shard_{idx:05d}.txt"
        if not path.is_file():
            break
        if idx not in cache:
            cache[idx] = path.read_text(encoding="utf-8", errors="ignore")
        chunk = cache[idx][inner:inner + remaining]
        if not chunk:
            break
        parts.append(codes_of(chunk))
        pos += len(chunk)
        remaining -= len(chunk)
    return np.concatenate(parts) if parts else np.empty(0, dtype="<u4")


def _take_fallback(shards_dir: Path, indices: tuple[int, ...],
                   chars: int) -> np.ndarray:
    """旧路径: 按下标取分片片段 (没有 manifest 时用)。"""
    if chars <= 0:
        return np.empty(0, dtype="<u4")
    per = max(chars // len(indices), 1)
    parts = []
    for i in indices:
        path = Path(shards_dir) / f"shard_{i:05d}.txt"
        if path.is_file():
            parts.append(codes_of(
                path.read_text(encoding="utf-8", errors="ignore")[:per]))
    return np.concatenate(parts) if parts else np.empty(0, dtype="<u4")


def take_category(corpus_dir: Path, category: str, chars: int) -> np.ndarray:
    """按 manifest 偏移取某类别的 chars 个字符 (没有 manifest 时下标兜底)。"""
    corpus_dir = Path(corpus_dir)
    shards_dir = corpus_dir / "shards"
    offsets = category_offsets(corpus_dir)
    if category in offsets:
        start, avail = offsets[category]
        shard_chars = int(_manifest_of(corpus_dir).get("shard_chars",
                                                       DEFAULT_SHARD_CHARS))
        return read_offset(shards_dir, shard_chars, start, min(chars, avail))
    if category in FALLBACK_SHARDS:
        return _take_fallback(shards_dir, FALLBACK_SHARDS[category], chars)
    return np.empty(0, dtype="<u4")


def build_lut(tokenizer: CharTokenizer) -> np.ndarray:
    """码点 -> token id 查找表 (未知码点落到 pad)。"""
    lut = np.full(0x110000, tokenizer.pad_id, dtype=np.int32)
    for ch, i in tokenizer.stoi.items():
        lut[ord(ch)] = i
    return lut


def build_pool(corpus_dir: str | Path = DEFAULT_CORPUS_DIR,
               dialogue_chars: int = 24_000_000,
               math_chars: int = 10_000_000,
               code_chars: int = 6_000_000,
               ) -> tuple[np.ndarray, CharTokenizer, dict]:
    """拼池子; 返回 (ids, tokenizer, stats)。"""
    corpus_dir = Path(corpus_dir)
    parts = {
        "dialogue": take_category(corpus_dir, "dialogue", dialogue_chars),
        "math": take_category(corpus_dir, "math", math_chars),
        "code": take_category(corpus_dir, "code", code_chars),
    }
    parts = {k: v for k, v in parts.items() if v.size}
    if not parts:
        raise SystemExit(f"{corpus_dir} 里没有可用语料 (需要 shards/shard_*.txt)")
    all_codes = np.concatenate(list(parts.values()))

    tokenizer = CharTokenizer()
    uniq = np.unique(all_codes)                      # 覆盖池子里出现的所有字符
    tokenizer.add("".join(chr(int(c)) for c in uniq))
    ids = build_lut(tokenizer)[all_codes].astype(np.int32)

    stats = {
        "total_chars": int(ids.size),
        "vocab_size": tokenizer.vocab_size,
        "parts": {k: int(v.size) for k, v in parts.items()},
        "ratios": {k: round(v.size / ids.size, 3) for k, v in parts.items()},
        "corpus_dir": str(corpus_dir),
        "manifest_categories": sorted(category_offsets(corpus_dir)),
    }
    return ids, tokenizer, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="拼装训练语料池 (对话优先)")
    ap.add_argument("--corpus-dir", type=str, default=str(DEFAULT_CORPUS_DIR),
                    help="语料目录 (含 shards/ 与 manifest.json)")
    ap.add_argument("--shards-dir", type=str, default=None,
                    help="兼容旧用法: 直接给分片目录")
    ap.add_argument("--dialogue-chars", type=int, default=24_000_000)
    ap.add_argument("--math-chars", type=int, default=10_000_000)
    ap.add_argument("--code-chars", type=int, default=6_000_000)
    ap.add_argument("--val-chars", type=int, default=200_000,
                    help="池子尾部留作验证集的字符数")
    ap.add_argument("--out", type=str, default="/tmp/enoch_pool.npz")
    ap.add_argument("--stats", type=str, default=None, help="配比统计 JSON 路径")
    args = ap.parse_args()

    corpus_dir = args.corpus_dir
    if args.shards_dir:
        corpus_dir = str(Path(args.shards_dir).parent)
    ids, tokenizer, stats = build_pool(corpus_dir, args.dialogue_chars,
                                       args.math_chars, args.code_chars)
    stats["val_chars"] = min(args.val_chars, ids.size // 10)
    # 40M 字符的池子约 160MB, 用未压缩 npz: 写盘 <1s (压缩要几十秒且没收益)
    np.savez(args.out, ids=ids, vocab=np.array(list(tokenizer.chars)))
    stats["out"] = args.out
    if args.stats:
        Path(args.stats).write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
