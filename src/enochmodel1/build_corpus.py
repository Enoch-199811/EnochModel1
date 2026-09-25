"""生成 10 亿字符级大规模语料 (enochmodel1.build_corpus)。

默认产出 ``data/corpus/`` (分片目录 + manifest.json), 内容构成:

  1. 代码 (~40%): 真实源码 (Go / Node.js / PHP, 从官方 CDN 下载);
  2. 数学 (~30%): 合成四则运算 / 多步表达式 / 应用题 / 解方程
                  (答案由程序计算, 保证正确);
  3. 日常对话 (~30%): 合成多轮中文对话 (问候 / 生活 / 学习 / 问答 /
                  数学辅导 / 知识讲解, 用插槽组合提升多样性)。

如网络允许, 也可用 ``--chinese-chars`` 加入中文维基百科真实正文。

语料以分片 (默认每片 64M 字符) 流式写入, 不会一次性把全部数据放进内存;
``pretrain.py --corpus data/corpus`` 会按分片流式读取训练。

用法:
    uv run enoch-build-corpus                          # 生成 10 亿字符
    uv run enoch-build-corpus --target-chars 100000000 # 只生成 1 亿
    uv run enoch-build-corpus --no-download            # 跳过下载, 用缓存
"""

from __future__ import annotations

import argparse
import bz2
import json
import random
import re
import shutil
import subprocess
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TextIO

from ._paths import DATA_DIR


DEFAULT_OUT_DIR = DATA_DIR / "corpus"
RAW_DIR = DATA_DIR / "raw"
DEFAULT_TARGET_CHARS = 1_000_000_000
DEFAULT_SHARD_CHARS = 64 * 1024 * 1024
DEFAULT_RATIOS = {"code": 0.40, "math": 0.30, "dialogue": 0.30, "chinese": 0.0}

CODE_REPOS = [
    # (名字, 下载地址, 只提取包含该路径前缀的成员; None=全部)
    ("go", "https://go.dev/dl/go1.22.4.src.tar.gz", "go/src/"),
    ("go121", "https://go.dev/dl/go1.21.4.src.tar.gz", "go/src/"),
    ("node", "https://nodejs.org/dist/v20.15.0/node-v20.15.0.tar.gz", None),
    ("php", "https://www.php.net/distributions/php-8.3.8.tar.gz", None),
    ("php82", "https://www.php.net/distributions/php-8.2.24.tar.gz", None),
]

WIKI_URL = "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"

CODE_EXTS = {
    ".py", ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".hxx", ".js", ".mjs",
    ".ts", ".rs", ".go", ".java", ".rb", ".sh", ".bash", ".zsh", ".m", ".mm",
    ".swift", ".cs", ".sql", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".xml", ".md", ".txt", ".rst", ".s", ".asm",
}
SKIP_PARTS = {
    ".git", "node_modules", "vendor", "third_party", "third-party", "site-packages",
    "autom4te.cache", ".idea", ".vscode", "__pycache__", ".venv", "venv",
}
MAX_CODE_FILE_BYTES = 2 * 1024 * 1024


# ---------------------------------------------------------------------------
# 分片写入器
# ---------------------------------------------------------------------------

class ShardWriter:
    """把文本流式写入 out_dir/shards/shard_XXXXX.txt, 按字符数轮转分片。"""

    def __init__(self, out_dir: Path, shard_chars: int = DEFAULT_SHARD_CHARS) -> None:
        self.out_dir = out_dir
        self.shards_dir = out_dir / "shards"
        if self.shards_dir.exists():
            shutil.rmtree(self.shards_dir)
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.shard_chars = shard_chars
        self._fh: TextIO | None = None
        self._idx = 0
        self._shard_written = 0
        self.total_chars = 0
        self.files: list[dict] = []

    def _open_next(self) -> None:
        path = self.shards_dir / f"shard_{self._idx:05d}.txt"
        self._fh = open(path, "w", encoding="utf-8", buffering=1024 * 1024)
        self.files.append({"path": f"shards/{path.name}", "chars": 0})
        self._idx += 1
        self._shard_written = 0

    def write(self, text: str) -> None:
        if self._fh is None:
            self._open_next()
        assert self._fh is not None
        self._fh.write(text)
        n = len(text)
        self._shard_written += n
        self.total_chars += n
        self.files[-1]["chars"] += n
        if self._shard_written >= self.shard_chars:
            self._fh.close()
            self._fh = None

    def finish(self) -> dict:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        while self.files and self.files[-1]["chars"] == 0:
            (self.shards_dir / Path(self.files.pop()["path"]).name).unlink(
                missing_ok=True)
        return {
            "num_shards": len(self.files),
            "files": self.files,
            "total_chars": self.total_chars,
        }


def _progress(label: str, written: int, budget: int, last_pct: int) -> int:
    """每跨过 10% 打印一次进度, 返回新的 last_pct。"""
    if budget <= 0:
        return 100
    pct = int(written * 10 // budget)
    if pct > last_pct:
        print(f"  [{label}] {written / 1e6:.0f}M / {budget / 1e6:.0f}M "
              f"({written / budget * 100:.0f}%)")
        return pct
    return last_pct


def download(url: str, dest: Path, allow_download: bool = True) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  缓存命中: {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return
    if not allow_download:
        raise SystemExit(f"缺少缓存文件 {dest}, 且 --no-download 禁止下载")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  下载 {url}")
    subprocess.run(
        ["curl", "-L", "--fail", "--silent", "--show-error",
         "--retry", "3", "-C", "-", "-o", str(dest), url],
        check=True,
    )
    print(f"  已保存 {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


# ---------------------------------------------------------------------------
# 1. 真实代码 (GitHub 源码包)
# ---------------------------------------------------------------------------

def collect_code(writer: ShardWriter, budget: int, allow_download: bool) -> int:
    written = 0
    for name, url, include in CODE_REPOS:
        if written >= budget:
            break
        tarball = RAW_DIR / f"{name}.tar.gz"
        download(url, tarball, allow_download)
        before = written
        written = _extract_code_tarball(writer, tarball, budget - written,
                                        name, written, include)
        print(f"  [{name}] 写入 {written - before:,} 字符")
    return written


def _extract_code_tarball(writer: ShardWriter, tarball: Path, budget: int,
                          label: str, already: int,
                          include: str | None = None) -> int:
    written = already
    last_pct = 0
    with tarfile.open(tarball, "r:*") as tf:
        for member in tf.getmembers():
            if written - already >= budget:
                break
            if not member.isfile() or member.size == 0:
                continue
            if member.size > MAX_CODE_FILE_BYTES:
                continue
            path = Path(member.name)
            if include is not None and include not in member.name:
                continue
            if path.suffix.lower() not in CODE_EXTS:
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            try:
                fobj = tf.extractfile(member)
            except (KeyError, OSError, tarfile.TarError):
                continue
            if fobj is None:
                continue
            raw = fobj.read()
            fobj.close()
            if b"\x00" in raw[:4096]:
                continue
            text = raw.decode("utf-8", errors="replace")
            block = f"# ==== {label}: {path} ====\n{text}\n"
            writer.write(block)
            written += len(block)
            last_pct = _progress("code", written - already, budget, last_pct)
    return written


# ---------------------------------------------------------------------------
# 2. 中文维基百科正文
# ---------------------------------------------------------------------------

_WIKI_RULES = [
    (re.compile(r"<!--.*?-->", re.S), " "),
    (re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.S), " "),
    (re.compile(r"<[^>]+>"), " "),
    (re.compile(r"\{\{[^{}]*\}\}"), " "),
    (re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]"), r"\1"),
    (re.compile(r"\[(?:https?|ftp)://[^\s\]]+\s*([^\]]*)\]"), r"\1"),
    (re.compile(r"'{2,}"), ""),
    (re.compile(r"={2,}"), " "),
    (re.compile(r"^\s*[|!\-*#:;]+", re.M), " "),
]


def strip_wiki(text: str) -> str:
    """粗略去掉 MediaWiki 标记, 保留正文。"""
    for pat, repl in _WIKI_RULES:
        text = pat.sub(repl, text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def collect_wikipedia(writer: ShardWriter, budget: int,
                      allow_download: bool) -> int:
    dest = RAW_DIR / "zhwiki-pages-articles.xml.bz2"
    download(WIKI_URL, dest, allow_download)
    written = 0
    last_pct = 0
    t0 = time.time()
    with bz2.open(dest, "rb") as raw:
        for _event, elem in ET.iterparse(raw, events=("end",)):
            if not elem.tag.endswith("}text"):
                elem.clear()
                continue
            text = elem.text or ""
            elem.clear()
            cleaned = strip_wiki(text)
            if len(cleaned) < 40:
                continue
            for para in cleaned.split("\n"):
                para = para.strip()
                if len(para) >= 30:
                    writer.write(para + "\n")
                    written += len(para) + 1
            if written >= budget:
                break
            last_pct = _progress("chinese", written, budget, last_pct)
            if written and written % 50_000_000 == 0:
                print(f"  [chinese] 用时 {time.time() - t0:.0f}s, "
                      f"{written / 1e6:.0f}M 字符")
    return written


# ---------------------------------------------------------------------------
# 3. 合成数学
# ---------------------------------------------------------------------------

def _chunk_write(writer: ShardWriter, chunk: list[str], written: int) -> int:
    if chunk:
        text = "".join(chunk)
        writer.write(text)
        written += len(text)
        chunk.clear()
    return written


def generate_math(writer: ShardWriter, budget: int, rng: random.Random) -> int:
    """合成四则运算 / 多步表达式 / 应用题 / 解方程 (答案正确)。"""
    written = 0
    last_pct = 0
    chunk: list[str] = []
    ops = ("+", "-", "*")
    while written < budget:
        r = rng.random()
        if r < 0.45:
            a = rng.randint(0, 999)
            b = rng.randint(0, 999)
            op = rng.choice(ops)
            if op == "-" and a < b:
                a, b = b, a
            ans = a + b if op == "+" else (a - b if op == "-" else a * b)
            chunk.append(f"计算：{a} {op} {b} = {ans}\n")
        elif r < 0.68:
            a = rng.randint(1, 99)
            b = rng.randint(1, 99)
            c = rng.randint(1, 99)
            op1 = rng.choice(ops)
            op2 = rng.choice(ops)
            expr = f"{a} {op1} {b} {op2} {c}"
            ans = eval(expr)
            chunk.append(f"计算：{expr} = {ans}\n")
        elif r < 0.86:
            kind = rng.randint(0, 5)
            if kind == 0:
                a = rng.randint(3, 500)
                b = rng.randint(1, a)
                chunk.append(f"小明有 {a} 个苹果，又买了 {b} 个，"
                             f"现在一共有多少个？答案：{a + b} 个。\n")
            elif kind == 1:
                a = rng.randint(10, 800)
                b = rng.randint(1, a - 1)
                chunk.append(f"书架上有 {a} 本书，借走 {b} 本，"
                             f"还剩多少本？答案：{a - b} 本。\n")
            elif kind == 2:
                a = rng.randint(2, 30)
                b = rng.randint(2, 20)
                chunk.append(f"每盒有 {a} 支铅笔，{b} 盒一共有多少支？"
                             f"答案：{a * b} 支。\n")
            elif kind == 3:
                a = rng.randint(10, 200)
                b = rng.randint(2, 9)
                chunk.append(f"把 {a} 块糖平均分给 {b} 个小朋友，每人分几块，"
                             f"还剩几块？答案：每人 {a // b} 块，余 {a % b} 块。\n")
            elif kind == 4:
                a = rng.randint(30, 120)
                b = rng.randint(2, 8)
                chunk.append(f"汽车每小时行驶 {a} 公里，行驶 {b} 小时，"
                             f"一共行驶多少公里？答案：{a * b} 公里。\n")
            else:
                a = rng.randint(10, 100) * 10
                d = rng.randint(5, 9)
                chunk.append(f"一件商品原价 {a} 元，打 {d} 折出售，"
                             f"现价多少元？答案：{a * d // 10} 元。\n")
        else:
            kind = rng.randint(0, 2)
            if kind == 0:
                b = rng.randint(1, 200)
                x = rng.randint(1, 500)
                chunk.append(f"解方程：x + {b} = {x + b}，x 等于多少？"
                             f"答案：{x}。\n")
            elif kind == 1:
                b = rng.randint(1, 200)
                x = rng.randint(b, 500)
                chunk.append(f"解方程：x - {b} = {x - b}，x 等于多少？"
                             f"答案：{x}。\n")
            else:
                a = rng.randint(2, 9)
                x = rng.randint(1, 200)
                chunk.append(f"解方程：{a} × x = {a * x}，x 等于多少？"
                             f"答案：{x}。\n")
        written = _chunk_write(writer, chunk, written)
        last_pct = _progress("math", written, budget, last_pct)
    written = _chunk_write(writer, chunk, written)
    return written


# ---------------------------------------------------------------------------
# 4. 合成日常对话
# ---------------------------------------------------------------------------

def generate_dialogue(writer: ShardWriter, budget: int,
                      rng: random.Random) -> int:
    """合成多轮对话: 问候 / 生活 / 学习 / 问答 / 数学辅导。"""
    pairs = [
        ("你好", "你好呀，很高兴见到你。"),
        ("你好，今天过得怎么样？", "还不错，就是有点忙，你呢？"),
        ("今天天气怎么样？", "今天晴空万里，适合出门走走。"),
        ("今天下雨了，出门记得带伞。", "好的，谢谢提醒！"),
        ("中午吃什么好呢？", "我煮了番茄鸡蛋面，很好吃，你可以试试。"),
        ("你吃饭了吗？", "刚吃过，谢谢关心。"),
        ("你的爱好是什么？", "我喜欢看书、跑步和听音乐。"),
        ("周末有什么计划？", "打算去爬山，顺便拍点照片。"),
        ("最近在忙什么？", "在学 Python，感觉很有意思。"),
        ("我想学机器学习，从哪里开始好？", "先学 Python 基础，再理解线性代数和概率，最后上手小项目。"),
        ("什么是循环？", "循环就是反复执行同一段代码，比如 for 和 while。"),
        ("你会写代码吗？", "会的，我可以帮你写 Python 程序，也可以帮你排查报错。"),
        ("什么是函数？", "函数用 def 定义，把重复的代码封装起来，方便反复调用。"),
        ("帮我看看这个报错", "把错误信息发给我，我来帮你排查。"),
        ("能给我讲个笑话吗？", "程序员最讨厌两件事：写注释，别人不写注释。"),
        ("为什么天空是蓝色的？", "太阳光在空气中散射，蓝光散射得最厉害，所以天空看起来是蓝色的。"),
        ("你能帮我算一道数学题吗？", "当然可以，请把题目告诉我。"),
        ("谢谢你的帮助！", "不客气，很高兴能帮到你。"),
        ("抱歉，刚才没听清楚。", "没关系，我再说一遍。"),
        ("请问几点钟了？", "现在时间不早了，早点休息。"),
        ("明天见！", "明天见，早点休息！"),
        ("再见！", "好的，回头聊！"),
        ("你会做什么？", "我可以帮你写代码、查资料、解答问题，也可以陪你聊天。"),
        ("什么是列表？", "列表用方括号表示，可以存放任意多个元素，例如 numbers = [1, 2, 3]。"),
        ("什么是字典？", "字典用花括号表示，通过键来取值，例如 person = {\"name\": \"小明\"}。"),
        ("什么是递归？", "递归是函数调用自己，适合解决分治类问题，比如计算阶乘。"),
        ("怎么提高编程水平？", "多写、多读、多调试，把大问题拆成小问题，逐步解决。"),
        ("今天学了什么？", "学了一个新算法，冒泡排序，每轮把相邻的较大元素往后交换。"),
        ("作业多吗？", "作业好多，正在赶进度。"),
        ("有什么好书推荐？", "这本书讲得真清楚，推荐给你。"),
        ("每天背多少单词？", "每天背二十个单词，坚持一个月了。"),
        ("晚安", "晚安，做个好梦！"),
    ]
    mono = [
        "变量是用来保存数据的名字，比如 x = 5 就把数字 5 存进了变量 x。",
        "Python 用缩进表示代码块，同一个缩进级别的语句属于同一个块。",
        "字符串可以用加号拼接，也可以用 format 方法格式化。",
        "异常处理用 try 和 except，让程序在出错时不会直接崩溃。",
        "列表推导式可以一行生成新列表，例如 [x * 2 for x in range(5)]。",
        "二分查找要求数据有序，每次排除一半，效率很高。",
        "栈是后进先出的结构，队列是先进先出的结构。",
        "哈希表通过哈希函数把键映射到位置，查找非常快。",
        "动态规划把大问题拆成重叠的子问题，避免重复计算。",
        "贪心算法每一步都选当前最优，但不一定得到全局最优。",
        "机器学习是让计算机从数据中自动学习规律的方法。",
        "监督学习使用带标签的数据，让模型学会从输入预测输出。",
        "Transformer 使用自注意力机制，能同时看到序列中所有位置。",
        "注意力机制根据查询和键的相似度分配权重。",
        "位置编码告诉模型每个 token 在序列中的位置。",
        "语言模型的任务是预测下一个字符或下一个词。",
        "温度参数控制生成时的随机性，温度越低输出越确定。",
        "交叉熵损失常用于分类问题，衡量两个概率分布的差异。",
        "Adam 优化器结合了动量和自适应学习率，收敛很快。",
        "加法就是把两个数合并在一起，例如 3 加 5 等于 8。",
        "乘法是求几个相同加数的和，例如 4 乘以 3 等于 12。",
        "质数是只有 1 和它本身两个因数的自然数。",
        "圆周率约等于 3.14159，是圆的周长和直径的比值。",
        "分数表示整体的一部分，例如二分之一写作 1/2。",
        "方程是含有未知数的等式，解方程就是求出未知数的值。",
        "平均数是所有数之和除以数的个数。",
        "概率表示事件发生的可能性，范围从 0 到 1。",
        "数据是 AI 的燃料，高质量的数据能显著提升模型效果。",
        "算法是解决问题的步骤，好的算法又快又省资源。",
        "程序 = 数据结构 + 算法，这是计算机科学的一句名言。",
        "写代码要像写文章一样清晰，先想清楚再动手。",
        "遇到问题先搜索、再复现、然后定位、最后修复。",
        "单元测试针对函数的最小行为做验证，保证重构不破坏功能。",
        "缓存把常用数据放在更快的地方，用空间换时间。",
    ]
    followups = [
        "然后呢？", "后来呢？", "原来如此。", "有道理。", "明白了。",
        "真的吗？", "不错不错。", "还有吗？", "学到了。", "挺好的。",
        "太棒了。", "那后来怎么样了？",
    ]
    followup_replies = [
        "嗯嗯，挺好的。", "原来是这样。", "真有意思。", "学到了！",
        "好的，我明白了。", "那我也试试。", "说得对。", "谢谢你告诉我。",
    ]
    topics = ["天气", "学习", "工作", "心情", "美食", "运动", "电影",
              "音乐", "旅行", "读书"]
    adjs = ["不错", "很好", "一般", "有点累", "很棒", "还行", "挺顺利", "有点忙"]
    details = [
        "上午把事情都办完了", "刚开完一个会", "出去走了一圈",
        "在家看了一下午书", "和朋友吃了顿饭", "做了个计划表",
        "练了一会儿琴", "跑了几公里", "拍了些照片", "把房间收拾了一下",
        "听了几首歌", "看了一集纪录片",
    ]
    subjects = ["Python", "机器学习", "英语", "数学", "吉他", "摄影",
                "做饭", "游泳", "写作", "日语"]
    starts = ["基础语法", "常用词汇", "加减法", "和弦", "构图", "切菜",
              "换气", "短句写作", "假名", "线性代数"]
    nexts = ["多写小项目", "多开口说", "多做练习", "多弹几首曲子",
             "多拍多改", "多尝试新菜谱", "多游几圈", "每天写一段",
             "多听多模仿", "多做推导"]
    plans = ["周末去爬山", "晚上看部电影", "明天早起跑步", "下周读完一本书",
             "假期去旅行", "今晚做顿饭", "去逛博物馆", "学一道新菜"]
    written = 0
    last_pct = 0
    chunk: list[str] = []
    while written < budget:
        n_turns = rng.randint(2, 6)
        session: list[str] = []
        for _ in range(n_turns):
            r = rng.random()
            if r < 0.10:
                # 数学辅导: 现场生成并计算, 保证正确
                a = rng.randint(0, 99)
                b = rng.randint(0, 99)
                op = rng.choice(("+", "-", "*"))
                if op == "-" and a < b:
                    a, b = b, a
                ans = a + b if op == "+" else (a - b if op == "-" else a * b)
                session.append(f"你: {a}{op}{b}等于几？\n小诺: 等于 {ans}。")
            elif r < 0.25:
                # 插槽问答: 话题 / 感受 / 学习建议, 组合出大量变体
                q = rng.choice([
                    f"今天{rng.choice(topics)}怎么样？",
                    f"最近{rng.choice(topics)}怎么样？",
                    f"我想学{rng.choice(subjects)}，从哪里开始好？",
                    f"你对{rng.choice(subjects)}有什么建议吗？",
                    "周末有什么安排？",
                    "最近有什么计划？",
                ])
                a = rng.choice([
                    f"{rng.choice(adjs)}，{rng.choice(details)}。",
                    f"可以先从{rng.choice(starts)}开始，然后"
                    f"{rng.choice(nexts)}，慢慢就会熟练。",
                    f"{rng.choice(plans)}。",
                ])
                session.append(f"你: {q}\n小诺: {a}")
            elif r < 0.88:
                q, a = rng.choice(pairs)
                session.append(f"你: {q}\n小诺: {a}")
            else:
                session.append(f"你: 给我讲讲这个知识点\n小诺: {rng.choice(mono)}")
            if rng.random() < 0.22:
                session.append(f"你: {rng.choice(followups)}\n"
                               f"小诺: {rng.choice(followup_replies)}")
        text = "\n".join(session) + "\n\n"
        chunk.append(text)
        written = _chunk_write(writer, chunk, written)
        last_pct = _progress("dialogue", written, budget, last_pct)
    written = _chunk_write(writer, chunk, written)
    return written


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EnochModel1: 生成大规模分片语料 (代码/中文/数学/对话)")
    p.add_argument("--target-chars", type=int, default=DEFAULT_TARGET_CHARS,
                   help="目标字符总数 (默认 10 亿)")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                   help="输出目录 (默认 data/corpus)")
    p.add_argument("--shard-chars", type=int, default=DEFAULT_SHARD_CHARS,
                   help="每个分片的字符数 (默认 64M)")
    for key in DEFAULT_RATIOS:
        p.add_argument(f"--{key}-chars", type=int, default=None,
                       help=f"{key} 类别字符数 (默认按比例 {DEFAULT_RATIOS[key]:.0%})")
    p.add_argument("--no-download", action="store_true",
                   help="跳过网络下载, 只使用已有缓存")
    p.add_argument("--seed", type=int, default=20260829, help="随机种子")
    return p.parse_args(argv)


def category_budgets(args: argparse.Namespace) -> dict[str, int]:
    budgets = {}
    for key, ratio in DEFAULT_RATIOS.items():
        explicit = getattr(args, f"{key}_chars")
        budgets[key] = explicit if explicit is not None else int(args.target_chars * ratio)
    return budgets


def main() -> None:
    args = parse_args()
    budgets = category_budgets(args)
    out_dir = Path(args.out_dir)
    writer = ShardWriter(out_dir, args.shard_chars)
    rng = random.Random(args.seed)

    print("=" * 64)
    print("EnochModel1: 大规模语料构建")
    print(f"目标: {args.target_chars:,} 字符 | 输出: {out_dir}")
    print(f"构成: code={budgets['code']:,} chinese={budgets['chinese']:,} "
          f"math={budgets['math']:,} dialogue={budgets['dialogue']:,}")
    print("=" * 64)

    actual: dict[str, int] = {}
    if budgets["code"] > 0:
        t0 = time.time()
        print("\n[1/4] 真实代码 (GitHub 源码)...")
        actual["code"] = collect_code(writer, budgets["code"], not args.no_download)
        print(f"  完成: {actual['code']:,} 字符, 用时 {time.time() - t0:.0f}s")
    if budgets["chinese"] > 0:
        t0 = time.time()
        print("\n[2/4] 中文维基百科正文...")
        actual["chinese"] = collect_wikipedia(writer, budgets["chinese"],
                                              not args.no_download)
        print(f"  完成: {actual['chinese']:,} 字符, 用时 {time.time() - t0:.0f}s")
    if budgets["math"] > 0:
        t0 = time.time()
        print("\n[3/4] 合成数学...")
        actual["math"] = generate_math(writer, budgets["math"], rng)
        print(f"  完成: {actual['math']:,} 字符, 用时 {time.time() - t0:.0f}s")
    if budgets["dialogue"] > 0:
        t0 = time.time()
        print("\n[4/4] 合成日常对话...")
        actual["dialogue"] = generate_dialogue(writer, budgets["dialogue"], rng)
        print(f"  完成: {actual['dialogue']:,} 字符, 用时 {time.time() - t0:.0f}s")

    info = writer.finish()
    manifest = {
        "target_chars": args.target_chars,
        "total_chars": info["total_chars"],
        "num_shards": info["num_shards"],
        "shard_chars": args.shard_chars,
        "categories": budgets,
        "actual": actual,
        "sources": {
            "code": [url for _, url, _ in CODE_REPOS],
            "chinese": [WIKI_URL],
            "math": "synthetic",
            "dialogue": "synthetic",
        },
        "seed": args.seed,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"语料构建完成: {info['total_chars']:,} 字符, "
          f"{info['num_shards']} 个分片")
    for key in DEFAULT_RATIOS:
        print(f"  {key}: {actual.get(key, 0):,} 字符")
    print(f"分片目录: {writer.shards_dir}")
    print(f"manifest: {out_dir / 'manifest.json'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
