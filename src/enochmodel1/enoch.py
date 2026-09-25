"""
EnochModel1 核心库 (enoch.py)
=============================

被各个入口共用的一份纯 NumPy 实现:

* main.py      预训练 + speed×score 强化学习演示 (RL 收缩实验);
* pretrain.py  预训练入口: 语料无监督 LM + 算术监督训练;
* chat.py      日常对话入口: 手动输入, 可直接纠正/打分并在线训练。

核心组件
--------
CharTokenizer    动态字符级分词器: 默认含数字与运算符, 遇到新字符 (如中文)
                自动扩充词表, 训练与对话都能无缝使用;
TinyTransformer  字符级因果 Transformer, 手写前向/反向传播;
Adam             极简 Adam, 支持词表 / 序列长度动态扩展;
token_loss       加权交叉熵 + 熵正则 + 可选 KL 正则 (预训练与 RL 共用)。

checkpoint 同时保存 vocab, 保证恢复时词表与模型一致。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 1. 基础工具
# ---------------------------------------------------------------------------

EOS_CHAR = "\x03"  # 句子结束符 (打印时显示为 ␃)
BASE_CHARS = "0123456789+-*= "  # 默认词表 (数字 / 运算符 / 空格)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax。"""
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def softmax_inplace(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """原地 softmax: 直接覆盖 ``x`` 并返回它 (数值与 :func:`softmax` 等价)。

    注意力分数在 [B,H,L,L] 量级, 去掉一次 exp 结果与一次除法的临时数组,
    在训练热路径上是实打实的分配/拷贝削减。
    """
    x -= x.max(axis=axis, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=axis, keepdims=True)
    return x


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 log-softmax (比 ``log(softmax(x))`` 更稳, 少一次 clip)。"""
    m = x.max(axis=axis, keepdims=True)
    e = x - m
    np.exp(e, out=e)
    return x - m - np.log(e.sum(axis=axis, keepdims=True))


def atb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``sum_{n} a[n,:] ⊗ b[n,:]``, 即 ``np.einsum("nij,nik->jk", ...)``。

    把 batch 维拍平成一个 dgemm, 比 c_einsum 的通用收缩快约 3 倍
    (c_einsum 曾是反向传播的单项最大开销)。
    """
    return a.reshape(-1, a.shape[-1]).T @ b.reshape(-1, b.shape[-1])


# 记忆化: causal mask 只与 (长度, dtype) 有关, 构造一次后全局复用。
_CAUSAL_MASKS: dict[tuple[int, str], np.ndarray] = {}


def banned_ngram_tokens(tokens: list[int], n: int) -> np.ndarray:
    """重复惩罚：返回**会与已出现 n-gram 重复**的候选 token。

    只 bans "以当前结尾 (n-1) 个 token 为前缀、且该 n-gram 已经出现过"的补全，
    是标准 no-repeat-ngram 采样；``n <= 1`` 时不禁任何 token。
    """
    empty = np.empty(0, dtype=np.int64)
    if n <= 1 or len(tokens) < n - 1:
        return empty
    prefix = tuple(tokens[-(n - 1):])
    banned = {tokens[i + n - 1] for i in range(len(tokens) - n + 1)
              if tuple(tokens[i:i + n - 1]) == prefix}
    return np.array(sorted(banned), dtype=np.int64) if banned else empty


def causal_mask(length: int, dtype: np.dtype) -> np.ndarray:
    """返回 [L, L] 的上三角 -inf 掩码 (只读复用, 调用方不得原地修改)。"""
    key = (length, np.dtype(dtype).str)
    m = _CAUSAL_MASKS.get(key)
    if m is None:
        m = np.full((length, length), -np.inf, dtype=dtype)
        m[np.tril_indices(length)] = 0.0
        _CAUSAL_MASKS[key] = m
    return m


def onehot(targets: np.ndarray, vocab_size: int) -> np.ndarray:
    """[B, L] -> [B, L, V] 的 one-hot 矩阵。"""
    B, L = targets.shape
    oh = np.zeros((B, L, vocab_size), dtype=np.float64)
    oh[np.arange(B)[:, None], np.arange(L)[None, :], targets] = 1.0
    return oh


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU 的 sigmoid 近似: x * σ(1.702x), 便于求导。"""
    return x * (1.0 / (1.0 + np.exp(-1.702 * x)))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    """d gelu(x)/dx。"""
    s = 1.0 / (1.0 + np.exp(-1.702 * x))
    return s + x * 1.702 * s * (1.0 - s)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
               eps: float = 1e-5) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """对最后一维做 LayerNorm, 返回 (y, (xhat, mu, var)) 供反向使用。"""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    return xhat * gamma + beta, (xhat, mu, var)


def layer_norm_backward(dy: np.ndarray, xhat: np.ndarray,
                        gamma: np.ndarray,
                        var: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (dx, dgamma, dbeta)。"""
    s = np.sqrt(var + eps)
    dxhat = dy * gamma
    dgamma = (dy * xhat).sum(axis=(0, 1))
    dbeta = dy.sum(axis=(0, 1))
    dx = (1.0 / s) * (
        dxhat
        - dxhat.mean(axis=-1, keepdims=True)
        - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)
    )
    return dx, dgamma, dbeta


def levenshtein(a: str, b: str) -> int:
    """编辑距离, 用于 partial 分数。"""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                cur[j - 1] + 1,             # 删除
                prev[j] + 1,                # 插入
                prev[j - 1] + (ca != cb),   # 替换
            )
        prev = cur
    return prev[-1]


def score_response(expected: str, response: str, mode: str) -> float:
    """任务正确性 score, 取值 [0, 1]。

    exact:   严格相等才给 1, 否则 0;
    partial: 1 - 归一化编辑距离, 提供更稠密的信号。
    """
    if mode == "exact":
        return 1.0 if expected == response else 0.0
    if mode == "partial":
        d = levenshtein(expected, response)
        return max(0.0, 1.0 - d / max(len(expected), len(response), 1))
    raise ValueError(f"未知 score 模式: {mode}")


# ---------------------------------------------------------------------------
# 2. 字符级分词器 (动态词表)
# ---------------------------------------------------------------------------

class CharTokenizer:
    """极简字符级分词器。

    默认词表为数字/运算符/空格/EOS; 调用 add() 可以把任意新字符
    (比如中文) 追加到词表末尾, 保证已有 id 稳定不变。
    """

    def __init__(self, vocab: Sequence[str] | None = None) -> None:
        if vocab is None:
            self.chars = BASE_CHARS + EOS_CHAR
        else:
            chars = list(vocab)
            for c in BASE_CHARS + EOS_CHAR:  # 确保基础字符齐全
                if c not in chars:
                    chars.append(c)
            self.chars = "".join(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.eos_id = self.stoi[EOS_CHAR]
        self.pad_id = self.stoi[" "]

    def add(self, text: str) -> int:
        """把 text 中出现的新字符加入词表, 返回新增数量。"""
        # 只取"首次出现"的新字符, 避免长文本中同一字符的多次出现
        # 被重复计入词表 (否则词表会被吹到出现次数那么大)。
        new = list(dict.fromkeys(c for c in text if c not in self.stoi))
        if not new:
            return 0
        for c in new:
            self.stoi[c] = self.vocab_size
            self.itos[self.vocab_size] = c
            self.chars += c
            self.vocab_size += 1
        return len(new)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)


# ---------------------------------------------------------------------------
# 3. 数据: 四则运算任务
# ---------------------------------------------------------------------------

def make_prompt(rng: np.random.Generator, task: str = "easy") -> tuple[str, str]:
    """生成一个算术题 (prompt, 标准答案)。

    easy: 个位数加减乘, 答案 1~2 位, 微型模型可学会, 适合演示;
    hard: 多位数加减乘, 答案 1~5 位, 更有挑战 (需要更大模型/更多步数)。
    """
    op = rng.choice(["+", "-", "*"], p=[0.4, 0.3, 0.3])
    if task == "easy":
        a = int(rng.integers(0, 10))
        b = int(rng.integers(0, 10))
    elif task == "hard":
        if op == "+":
            a = int(rng.integers(0, 500))
            b = int(rng.integers(0, 500))
        elif op == "-":
            a = int(rng.integers(0, 500))
            b = int(rng.integers(0, a + 1))
        else:
            a = int(rng.integers(0, 100))
            b = int(rng.integers(0, 100))
    else:
        raise ValueError(f"未知任务: {task}")
    if op == "-" and a < b:
        a, b = b, a  # 保证非负结果
    answer = str({"+": a + b, "-": a - b, "*": a * b}[op])
    return f"{a}{op}{b}=", answer


# ---------------------------------------------------------------------------
# 4. 微型因果 Transformer (纯 NumPy, 手写反向传播)
# ---------------------------------------------------------------------------

class KVCache:
    """增量解码的记忆化缓存。

    每层保存已经算过的 K/V(形状 ``[B, n_heads, capacity, head_dim]``),
    解码下一步时只前向新 token, 直接复用历史 K/V —— 把每步的注意力
    代价从 O(L²) 降到 O(L), 不再重算整段前缀。
    """

    __slots__ = ("K", "V", "batch", "capacity")

    def __init__(self, model: TinyTransformer, batch: int,
                 capacity: int) -> None:
        H, dh, dt = model.n_heads, model.head_dim, model.dtype
        self.K = [np.zeros((batch, H, capacity, dh), dtype=dt)
                  for _ in range(model.n_layers)]
        self.V = [np.zeros((batch, H, capacity, dh), dtype=dt)
                  for _ in range(model.n_layers)]
        self.batch = batch
        self.capacity = capacity

    def store(self, layer: int, positions, kh: np.ndarray,
              vh: np.ndarray) -> None:
        """写入一层的新 K/V。

        ``positions`` 为 int 时表示连续区间 ``[p, p+n)`` (预填充);
        为 ``[B]`` 数组时按行写入各自的位置 (解码时各行长度可能不同)。
        """
        if isinstance(positions, (int, np.integer)):
            p = int(positions)
            n = kh.shape[2]
            self.K[layer][:, :, p:p + n, :] = kh
            self.V[layer][:, :, p:p + n, :] = vh
        else:
            rows = np.arange(self.batch)
            self.K[layer][rows, :, positions, :] = kh[:, :, 0, :]
            self.V[layer][rows, :, positions, :] = vh[:, :, 0, :]

    def nbytes(self) -> int:
        return (sum(a.nbytes for a in self.K) + sum(a.nbytes for a in self.V))


class TinyTransformer:
    """字符级因果 Transformer。

    结构: embedding + 位置编码 -> N 层 (LayerNorm -> MHA -> 残差 ->
    LayerNorm -> MLP -> 残差) -> LayerNorm -> 输出线性层。
    所有参数存放在 self.params 字典中, 便于 checkpoint 与梯度校验。
    """

    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4, max_pos: int = 64, seed: int = 0,
                 d_mlp: int | None = None, attn_dim: int | None = None,
                 dtype: str | np.dtype = np.float64) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_pos = max_pos
        # 注意力内部宽度: 默认 = d_model; 头剪枝后可小于 d_model
        # (k 个保留头 × head_dim), W_q/W_k/W_v 变窄、W_o 行数变少。
        # head_dim 由 attn_dim 决定 (默认 = d_model), 因此只要求 attn_dim
        # 能被 n_heads 整除; 头剪枝后 n_heads 可以不再整除 d_model。
        self.attn_dim = int(attn_dim) if attn_dim is not None else d_model
        assert self.attn_dim % n_heads == 0, "attn_dim 必须能被 n_heads 整除"
        self.head_dim = self.attn_dim // n_heads
        # MLP 隐藏层宽度: 默认 4*d_model; 结构化剪枝可以把它调窄
        self.d_mlp = int(d_mlp) if d_mlp is not None else 4 * d_model
        self.dtype = np.dtype(dtype)

        rng = np.random.default_rng(seed)
        dt = self.dtype
        ad = self.attn_dim
        self.params: dict[str, np.ndarray] = {}
        self.params["W_e"] = rng.normal(0.0, 0.02, (vocab_size, d_model)).astype(dt, copy=False)
        self.params["W_p"] = rng.normal(0.0, 0.02, (max_pos, d_model)).astype(dt, copy=False)
        self.params["W_out"] = rng.normal(0.0, 0.02, (d_model, vocab_size)).astype(dt, copy=False)
        self.params["b_out"] = np.zeros(vocab_size, dtype=dt)

        scale = 1.0 / math.sqrt(d_model)
        dm = self.d_mlp
        for l in range(n_layers):
            self.params[f"W_q{l}"] = rng.normal(0.0, scale, (d_model, ad)).astype(dt, copy=False)
            self.params[f"W_k{l}"] = rng.normal(0.0, scale, (d_model, ad)).astype(dt, copy=False)
            self.params[f"W_v{l}"] = rng.normal(0.0, scale, (d_model, ad)).astype(dt, copy=False)
            self.params[f"W_o{l}"] = rng.normal(0.0, scale, (ad, d_model)).astype(dt, copy=False)
            self.params[f"W_1{l}"] = rng.normal(0.0, scale, (d_model, dm)).astype(dt, copy=False)
            self.params[f"b_1{l}"] = np.zeros(dm, dtype=dt)
            self.params[f"W_2{l}"] = rng.normal(0.0, scale, (dm, d_model)).astype(dt, copy=False)
            self.params[f"b_2{l}"] = np.zeros(d_model, dtype=dt)
            self.params[f"g_ln1{l}"] = np.ones(d_model, dtype=dt)
            self.params[f"b_ln1{l}"] = np.zeros(d_model, dtype=dt)
            self.params[f"g_ln2{l}"] = np.ones(d_model, dtype=dt)
            self.params[f"b_ln2{l}"] = np.zeros(d_model, dtype=dt)
        self.params["g_lnf"] = np.ones(d_model, dtype=dt)
        self.params["b_lnf"] = np.zeros(d_model, dtype=dt)
        self._last_cache: dict | None = None

    def add_vocab(self, n_new: int) -> None:
        """词表扩展: 新字符的词嵌入/输出向量置零 (随后由训练学习)。"""
        if n_new <= 0:
            return
        d = self.d_model
        self.vocab_size += n_new
        self.params["W_e"] = np.vstack(
            [self.params["W_e"], np.zeros((n_new, d))])
        self.params["W_out"] = np.hstack(
            [self.params["W_out"], np.zeros((d, n_new))])
        self.params["b_out"] = np.append(self.params["b_out"], np.zeros(n_new))

    def extend_pos(self, new_max_pos: int) -> None:
        """位置编码扩展到更长序列 (新增部分置零)。"""
        if new_max_pos <= self.max_pos:
            return
        add = new_max_pos - self.max_pos
        self.params["W_p"] = np.vstack(
            [self.params["W_p"], np.zeros((add, self.d_model))])
        self.max_pos = new_max_pos

    # -- 前向 ---------------------------------------------------------------

    def forward(self, X: np.ndarray, kv: KVCache | None = None,
                kv_offset: int = 0) -> tuple[np.ndarray, dict]:
        """X: [B, L] token id (右侧 padding)。返回 (logits, cache)。

        传入 ``kv`` 时, 每层的 K/V 会写入缓存的 ``[kv_offset, kv_offset+L)``
        区间 (预填充阶段用), 供后续 :meth:`decode_step` 增量解码复用。
        """
        B, L = X.shape
        P = self.params
        H = self.n_heads
        dh = self.head_dim
        ad = self.attn_dim
        cache: dict = {"X": X.copy()}

        h = P["W_e"][X] + P["W_p"][:L][None, :, :]  # [B, L, d]

        causal = causal_mask(L, h.dtype)  # 记忆化, 不再每次重建

        for l in range(self.n_layers):
            # 第一个 LayerNorm + 多头注意力
            ln1, (xhat1, mu1, var1) = layer_norm(h, P[f"g_ln1{l}"], P[f"b_ln1{l}"])
            q = ln1 @ P[f"W_q{l}"]   # [B, L, d]
            k = ln1 @ P[f"W_k{l}"]
            v = ln1 @ P[f"W_v{l}"]
            qh = q.reshape(B, L, H, dh).transpose(0, 2, 1, 3)  # [B,H,L,dh]
            kh = k.reshape(B, L, H, dh).transpose(0, 2, 1, 3)
            vh = v.reshape(B, L, H, dh).transpose(0, 2, 1, 3)
            if kv is not None:
                kv.store(l, kv_offset, kh, vh)
            scores = qh @ kh.transpose(0, 1, 3, 2)  # [B,H,L,L]
            scores /= math.sqrt(dh)
            scores += causal[None, None, :, :]      # 原地累加, 省一次分配
            att = softmax_inplace(scores)
            ctx = (att @ vh).transpose(0, 2, 1, 3).reshape(B, L, ad)
            att_out = ctx @ P[f"W_o{l}"]
            att_out += h          # 残差融合 (h 之后不再需要)
            h2 = att_out

            # 第二个 LayerNorm + MLP
            ln2, (xhat2, mu2, var2) = layer_norm(h2, P[f"g_ln2{l}"], P[f"b_ln2{l}"])
            pre = ln2 @ P[f"W_1{l}"] + P[f"b_1{l}"]
            act = gelu(pre)
            mlp_out = act @ P[f"W_2{l}"]
            mlp_out += P[f"b_2{l}"]
            mlp_out += h2         # 残差融合
            h = mlp_out

            cache[f"ln1{l}"] = (ln1, xhat1, mu1, var1)
            cache[f"att{l}"] = (att, ctx)
            cache[f"qkv{l}"] = (q, k, v)   # 反向复用, 省 3 次投影 matmul
            cache[f"ln2{l}"] = (ln2, xhat2, mu2, var2)
            cache[f"mlp{l}"] = (pre, act)

        lnf, (xhatf, muf, varf) = layer_norm(h, P["g_lnf"], P["b_lnf"])
        logits = lnf @ P["W_out"] + P["b_out"]
        cache["lnf"] = (lnf, xhatf, muf, varf)
        return logits, cache

    # -- 增量解码 (记忆化: 复用历史 K/V, 不再重算前缀) ----------------------

    def decode_step(self, tokens: np.ndarray, positions: np.ndarray,
                    kv: KVCache) -> np.ndarray:
        """一次只前向一个新 token: tokens/positions 均为 [B]。

        复用 ``kv`` 里已缓存的历史 K/V, 只把当前 token 过一遍网络,
        每步代价从 O(L²) 降到 O(L)。返回 [B, vocab] 的下一 token logits。
        """
        B = tokens.shape[0]
        P = self.params
        H = self.n_heads
        dh = self.head_dim
        ad = self.attn_dim

        h = (P["W_e"][tokens] + P["W_p"][positions])[:, None, :]  # [B,1,d]
        Lmax = int(positions.max()) + 1
        keep = np.arange(Lmax)[None, :] <= positions[:, None]     # [B,Lmax]
        mask = np.where(keep[:, None, None, :], 0.0, -np.inf).astype(
            h.dtype, copy=False)

        for l in range(self.n_layers):
            ln1, _ = layer_norm(h, P[f"g_ln1{l}"], P[f"b_ln1{l}"])
            q = ln1 @ P[f"W_q{l}"]
            k = ln1 @ P[f"W_k{l}"]
            v = ln1 @ P[f"W_v{l}"]
            qh = q.reshape(B, 1, H, dh).transpose(0, 2, 1, 3)
            kh = k.reshape(B, 1, H, dh).transpose(0, 2, 1, 3)
            vh = v.reshape(B, 1, H, dh).transpose(0, 2, 1, 3)
            kv.store(l, positions, kh, vh)          # 逐行写入 (位置可能不同)
            K = kv.K[l][:, :, :Lmax, :]
            V = kv.V[l][:, :, :Lmax, :]
            scores = qh @ K.transpose(0, 1, 3, 2)
            scores /= math.sqrt(dh)
            scores += mask
            att = softmax_inplace(scores)
            ctx = (att @ V).transpose(0, 2, 1, 3).reshape(B, 1, ad)
            att_out = ctx @ P[f"W_o{l}"]
            att_out += h
            h2 = att_out

            ln2, _ = layer_norm(h2, P[f"g_ln2{l}"], P[f"b_ln2{l}"])
            pre = ln2 @ P[f"W_1{l}"] + P[f"b_1{l}"]
            act = gelu(pre)
            mlp_out = act @ P[f"W_2{l}"]
            mlp_out += P[f"b_2{l}"]
            mlp_out += h2
            h = mlp_out

        lnf, _ = layer_norm(h, P["g_lnf"], P["b_lnf"])
        logits = lnf @ P["W_out"] + P["b_out"]
        return logits[:, 0, :]

    def _set_cache(self, cache: dict) -> None:
        """保存最近一次前向的中间结果, 供 backward 使用。"""
        self._last_cache = cache

    # -- 反向 ---------------------------------------------------------------

    def backward(self, X: np.ndarray, dlogits: np.ndarray,
                 pad_id: int) -> dict[str, np.ndarray]:
        """dlogits: [B, L, V]。返回各参数梯度 (与 self.params 同构)。"""
        B, L = X.shape
        P = self.params
        ad = self.attn_dim
        cache = self._last_cache
        if cache is None:
            raise RuntimeError("backward 前必须先调用 forward 并 _set_cache")
        grads: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in P.items()}

        lnf, xhatf, _muf, varf = cache["lnf"]
        dlnf = dlogits @ P["W_out"].T
        grads["W_out"] = atb(lnf, dlogits)
        grads["b_out"] = dlogits.sum(axis=(0, 1))
        dh, grads["g_lnf"], grads["b_lnf"] = layer_norm_backward(
            dlnf, xhatf, P["g_lnf"], varf)

        for l in reversed(range(self.n_layers)):
            # --- MLP 反向 ---
            pre, act = cache[f"mlp{l}"]
            d_mlp_out = dh  # 残差: h_out = h2 + mlp_out
            dact = d_mlp_out @ P[f"W_2{l}"].T
            grads[f"W_2{l}"] = atb(act, d_mlp_out)
            grads[f"b_2{l}"] = d_mlp_out.sum(axis=(0, 1))
            dpre = dact * gelu_grad(pre)
            grads[f"W_1{l}"] = atb(cache[f"ln2{l}"][0], dpre)
            grads[f"b_1{l}"] = dpre.sum(axis=(0, 1))

            _ln2, xhat2, _mu2, var2 = cache[f"ln2{l}"]
            d_ln2_out = dpre @ P[f"W_1{l}"].T
            dh2, grads[f"g_ln2{l}"], grads[f"b_ln2{l}"] = layer_norm_backward(
                d_ln2_out, xhat2, P[f"g_ln2{l}"], var2)
            dh2 = dh2 + dh  # 残差: h2 = h + att_out

            # --- 注意力反向 ---
            ln1, xhat1, _mu1, var1 = cache[f"ln1{l}"]
            att, ctx = cache[f"att{l}"]
            d_att_out = dh2
            grads[f"W_o{l}"] = atb(ctx, d_att_out)
            dctx = d_att_out @ P[f"W_o{l}"].T
            H = self.n_heads
            dh_dim = self.head_dim
            dctx_h = dctx.reshape(B, L, H, dh_dim).transpose(0, 2, 1, 3)
            # 复用前向缓存的 q/k/v, 省掉 3 次投影 matmul
            q, k, v = cache[f"qkv{l}"]
            qh = q.reshape(B, L, H, dh_dim).transpose(0, 2, 1, 3)
            kh = k.reshape(B, L, H, dh_dim).transpose(0, 2, 1, 3)
            vh = v.reshape(B, L, H, dh_dim).transpose(0, 2, 1, 3)
            datt = dctx_h @ vh.transpose(0, 1, 3, 2)
            dscores = att * (datt - (datt * att).sum(axis=-1, keepdims=True))
            dqh = dscores @ kh / math.sqrt(dh_dim)
            dkh = dscores.transpose(0, 1, 3, 2) @ qh / math.sqrt(dh_dim)
            dvh = att.transpose(0, 1, 3, 2) @ dctx_h
            dq = dqh.transpose(0, 2, 1, 3).reshape(B, L, ad)
            dk = dkh.transpose(0, 2, 1, 3).reshape(B, L, ad)
            dv = dvh.transpose(0, 2, 1, 3).reshape(B, L, ad)
            grads[f"W_q{l}"] = atb(ln1, dq)
            grads[f"W_k{l}"] = atb(ln1, dk)
            grads[f"W_v{l}"] = atb(ln1, dv)
            d_ln1_out = dq @ P[f"W_q{l}"].T + dk @ P[f"W_k{l}"].T + dv @ P[f"W_v{l}"].T
            dh, grads[f"g_ln1{l}"], grads[f"b_ln1{l}"] = layer_norm_backward(
                d_ln1_out, xhat1, P[f"g_ln1{l}"], var1)
            dh = dh + dh2  # 残差: h2 = h + att_out

        # --- embedding 与位置编码反向 ---
        valid = X != pad_id
        np.add.at(grads["W_e"], X[valid], dh[valid])
        grads["W_p"][:L] += dh.sum(axis=0)
        return grads

    # -- 采样 ---------------------------------------------------------------

    def generate_batch(self, prompt_ids: list[int], group_size: int,
                       max_new: int, temperature: float,
                       rng: np.random.Generator,
                       eos_id: int, max_pos: int,
                       top_k: int = 0,
                       no_repeat_ngram: int = 0) -> tuple[list[list[int]], list[list[float]]]:
        """对同一个 prompt 采样 group_size 条回答。

        返回 (回答序列列表[含 eos, 无 padding], 每步 logprob 列表)。

        实现是"预填充 + 增量解码": prompt 只前向一次并写入 :class:`KVCache`,
        之后每步只喂一个新 token。采样顺序 (逐行、每行一次 ``rng.choice``)
        与朴素全量前向完全一致, 因此同 seed 下输出可复现。

        ``top_k > 0`` 只保留概率最高的 k 个 token; ``no_repeat_ngram > 1``
        禁止重复出现过的 n-gram —— 实测这就是"不管问什么都续同一句模板"的解药。
        两者默认 0 = 关闭, 行为与不带参数时**逐位一致**。
        """
        W = min(len(prompt_ids) + max_new, max_pos)
        seqs: list[list[int]] = [list(prompt_ids) for _ in range(group_size)]
        done = [False] * group_size
        logprob_rows: list[list[float]] = [[] for _ in range(group_size)]
        if max_new <= 0 or not prompt_ids or W <= 0:
            return seqs, logprob_rows

        capacity = W
        kv = KVCache(self, group_size, capacity)
        first = np.asarray(prompt_ids[:capacity], dtype=np.int64)
        Xp = np.tile(first, (group_size, 1))
        logits, cache = self.forward(Xp, kv=kv, kv_offset=0)
        self._set_cache(cache)
        row_logits = logits[:, Xp.shape[1] - 1, :]

        for _ in range(max_new):
            for i in range(group_size):
                if done[i]:
                    continue
                if temperature <= 0.0:
                    if no_repeat_ngram > 1 or 0 < top_k < self.vocab_size:
                        # 贪心同样受约束：把被禁/长尾 token 压到 -inf 再取 argmax
                        masked = row_logits[i].copy()
                        if no_repeat_ngram > 1:
                            ban = banned_ngram_tokens(seqs[i], no_repeat_ngram)
                            if ban.size:
                                masked[ban] = -np.inf
                        if 0 < top_k < self.vocab_size:
                            keep = np.argpartition(row_logits[i], -top_k)[-top_k:]
                            drop = np.ones_like(masked, dtype=bool)
                            drop[keep] = False
                            masked[drop] = -np.inf
                        tok = int(np.argmax(masked))
                    else:
                        tok = int(np.argmax(row_logits[i]))
                    logprob_rows[i].append(0.0)
                else:
                    p = softmax(row_logits[i] / temperature)
                    if no_repeat_ngram > 1:
                        ban = banned_ngram_tokens(seqs[i], no_repeat_ngram)
                        if ban.size:
                            p[ban] = 0.0
                    if 0 < top_k < self.vocab_size:
                        # 只留概率最高的 k 个（argpartition 是 O(V)，V 只有几百）
                        cut = np.argpartition(p, -top_k)[:-top_k]
                        p[cut] = 0.0
                    total = float(p.sum())
                    if not np.isfinite(total) or total <= 0.0:
                        p = softmax(row_logits[i] / temperature)   # 兜底：退回未约束分布
                        total = float(p.sum())
                    p = p / total  # 归一化, 避免浮点误差
                    tok = int(rng.choice(self.vocab_size, p=p))
                    logprob_rows[i].append(float(np.log(max(p[tok], 1e-12))))
                if len(seqs[i]) < capacity:
                    seqs[i].append(tok)  # eos 也计入序列, 供 truncation 判断
                else:
                    done[i] = True       # 达到最大长度: 停止增长 (剪掉越界写入)
                if tok == eos_id:
                    done[i] = True
            if all(done):
                break
            tokens = np.fromiter((s[-1] for s in seqs), dtype=np.int64,
                                 count=group_size)
            positions = np.fromiter((len(s) - 1 for s in seqs), dtype=np.int64,
                                    count=group_size)
            row_logits = self.decode_step(tokens, positions, kv)
        return seqs, logprob_rows


def generate_text(model: TinyTransformer, tokenizer: CharTokenizer,
                  seed: str, max_new: int, temperature: float,
                  rng: np.random.Generator, top_k: int = 0,
                  no_repeat_ngram: int = 0) -> str:
    """给一段种子文本, 贪心/采样续写并返回新生成的部分 (不含 eos)。"""
    prompt_ids = tokenizer.encode(seed)
    max_keep = max(model.max_pos - max_new, 1)
    if len(prompt_ids) > max_keep:
        prompt_ids = prompt_ids[-max_keep:]
    seqs, _ = model.generate_batch(
        prompt_ids, 1, max_new, temperature, rng,
        tokenizer.eos_id, model.max_pos,
        top_k=top_k, no_repeat_ngram=no_repeat_ngram)
    resp = seqs[0][len(prompt_ids):]
    return tokenizer.decode([t for t in resp if t != tokenizer.eos_id])


# ---------------------------------------------------------------------------
# 5. 损失 / 优化器 / 速度追踪 / 动态扩展
# ---------------------------------------------------------------------------

def token_loss(logits: np.ndarray, targets: np.ndarray, weights: np.ndarray,
               beta: float = 0.0, ref_logits: np.ndarray | None = None,
               kl_beta: float = 0.0) -> tuple[float, np.ndarray]:
    """带权重 (REINFORCE advantage) 的交叉熵 + 熵正则 + 可选 KL 正则。

    weights: [B, L], 非零位置才计入损失, 值即 per-token 的 advantage。
    ref_logits: 参考策略 (通常是预训练后的模型) 的 logits, 用于 KL 约束,
                防止 RL 阶段策略坍塌 (标准做法, 类似 PPO/GRPO)。
    返回 (loss, dlogits)。
    """
    B, L, _V = logits.shape
    logp = log_softmax(logits, axis=-1)   # 稳定且无需 clip
    probs = np.exp(logp)
    weights = np.asarray(weights, dtype=logp.dtype)
    idx_b = np.arange(B)[:, None]
    idx_l = np.arange(L)[None, :]
    n = max(int((weights != 0).sum()), 1)
    inv_n = 1.0 / n

    ce = -(weights * logp[idx_b, idx_l, targets]).sum() * inv_n

    H = -(probs * logp).sum(axis=-1)  # [B, L]
    masked = (weights != 0)
    ent_term = -(H * masked).sum() * inv_n

    loss = ce + beta * ent_term  # 最小化 -H 即最大化熵

    # dlogits = (probs - onehot(targets)) * scale, 但不再真的构造 onehot
    # (省下 [B, L, V] 的分配与拷贝, 只在目标位置减一次)
    scale = weights * inv_n
    dlogits = probs * scale[..., None]
    dlogits[idx_b, idx_l, targets] -= scale
    if beta != 0.0:
        ent_grad = probs * (logp + H[..., None])
        ent_grad *= masked[..., None]
        dlogits += (beta * inv_n) * ent_grad

    if ref_logits is not None and kl_beta > 0.0:
        logq = log_softmax(ref_logits, axis=-1)
        r = logp - logq  # 逐 token 的 log 比
        kl = (probs * r).sum(axis=-1)  # [B, L]
        loss += kl_beta * (kl * masked).sum() * inv_n
        dkl_dz = probs * (r - (probs * r).sum(axis=-1, keepdims=True))
        dkl_dz *= masked[..., None]
        dlogits += (kl_beta * inv_n) * dkl_dz
    return float(loss), dlogits


class Adam:
    """极简 Adam 优化器。"""

    def __init__(self, params: dict[str, np.ndarray]) -> None:
        self._params = params
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    @property
    def params(self) -> dict[str, np.ndarray]:
        return self._params

    def step(self, grads: dict[str, np.ndarray], lr: float,
             grad_clip: float) -> None:
        self.t += 1
        norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        scale = 1.0 if norm <= grad_clip else grad_clip / max(norm, 1e-12)
        for k, g in grads.items():
            g = g * scale
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            mhat = self.m[k] / (1.0 - 0.9 ** self.t)
            vhat = self.v[k] / (1.0 - 0.999 ** self.t)
            self.params[k] -= lr * mhat / (np.sqrt(vhat) + 1e-8)


class SpeedTracker:
    """追踪"上一步 token 数" (EMA), 输出 speed = 1 / max(ema_len, floor)。

    注意: 用上一批次的 EMA 长度计算本轮 reward, 正是"之前的 token 数";
    模型变短 -> ema_len 下降 -> speed 上升 -> 更强的"短"奖励, 形成正反馈。
    """

    def __init__(self, ema: float = 0.9, floor: float = 1.0) -> None:
        self.ema = ema
        self.floor = floor
        self.len: float | None = None

    def speed(self) -> float:
        if self.len is None:
            return 1.0
        return 1.0 / max(self.len, self.floor)

    def update(self, mean_len: float) -> None:
        if self.len is None:
            self.len = float(mean_len)
        else:
            self.len = self.ema * self.len + (1.0 - self.ema) * mean_len

    @property
    def ema_len(self) -> float:
        return self.len if self.len is not None else 0.0


def grow_vocab(model: TinyTransformer, opt: Adam, tokenizer: CharTokenizer,
               text: str) -> int:
    """把 text 里的新字符加入词表, 模型与优化器同步扩展。"""
    n = tokenizer.add(text)
    if n <= 0:
        return 0
    d = model.d_model
    model.add_vocab(n)
    opt.m["W_e"] = np.vstack([opt.m["W_e"], np.zeros((n, d))])
    opt.v["W_e"] = np.vstack([opt.v["W_e"], np.zeros((n, d))])
    opt.m["W_out"] = np.hstack([opt.m["W_out"], np.zeros((d, n))])
    opt.v["W_out"] = np.hstack([opt.v["W_out"], np.zeros((d, n))])
    opt.m["b_out"] = np.append(opt.m["b_out"], np.zeros(n))
    opt.v["b_out"] = np.append(opt.v["b_out"], np.zeros(n))
    return n


def grow_max_pos(model: TinyTransformer, opt: Adam, new_max_pos: int) -> None:
    """把最大序列长度扩展到 new_max_pos (模型与优化器同步)。"""
    if new_max_pos <= model.max_pos:
        return
    old = model.max_pos
    model.extend_pos(new_max_pos)
    add = new_max_pos - old
    opt.m["W_p"] = np.vstack([opt.m["W_p"], np.zeros((add, model.d_model))])
    opt.v["W_p"] = np.vstack([opt.v["W_p"], np.zeros((add, model.d_model))])


# ---------------------------------------------------------------------------
# 6. 批次构造 (预训练 / LM / RL 共用)
# ---------------------------------------------------------------------------

def _fill_batch(samples: list[tuple[list[int], list[int], float]],
                tokenizer: CharTokenizer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 (prompt_ids, resp_ids, weight) 列表打包成 (X, targets, weights)。

    损失位置 = [len(prompt)-1, len(seq)-2], 即"预测下一个响应 token"的
    所有源位置, 包含 "=" 处预测答案首位, 以及最后一个答案字符处预测 eos。
    """
    L = max(len(p) + len(r) for p, r, _ in samples)
    B = len(samples)
    X = np.full((B, L), tokenizer.pad_id, dtype=np.int64)
    targets = np.zeros_like(X)
    weights = np.zeros((B, L), dtype=np.float64)
    for i, (prompt_ids, resp_ids, w) in enumerate(samples):
        seq = prompt_ids + resp_ids
        X[i, :len(seq)] = seq
        start = len(prompt_ids) - 1
        end = len(seq) - 1  # 不含 (最后一个 token 没有后继)
        for j in range(start, end):
            targets[i, j] = seq[j + 1]
            weights[i, j] = w
    return X, targets, weights


def build_pretrain_batch(batch: list[tuple[str, str]],
                         tokenizer: CharTokenizer,
                         verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """监督预训练: 回答 token (含 eos) 作为下一个 token 预测目标。

    verbose=True 时目标为 answer+answer, 先教模型"啰嗦", 方便 RL 阶段
    展示 speed×score 如何把冗余输出收缩掉。
    """
    samples = []
    for prompt, answer in batch:
        target = answer + answer if verbose else answer
        resp_ids = tokenizer.encode(target) + [tokenizer.eos_id]
        samples.append((tokenizer.encode(prompt), resp_ids, 1.0))
    return _fill_batch(samples, tokenizer)


def build_rl_batch(samples: list[dict],
                   tokenizer: CharTokenizer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RL: 每个样本的 token 都带同一个 advantage 权重。"""
    packed = [(s["prompt_ids"], s["resp_ids"], s["adv"]) for s in samples]
    return _fill_batch(packed, tokenizer)


def build_lm_batch(corpus_ids: list[int], length: int, batch: int,
                   rng: np.random.Generator,
                   tokenizer: CharTokenizer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """无监督语言建模: 从语料随机切窗口, 每个位置预测下一个字符。"""
    L = max(min(length, len(corpus_ids) - 1), 1)
    max_start = max(len(corpus_ids) - L - 1, 0)
    X = np.full((batch, L), tokenizer.pad_id, dtype=np.int64)
    targets = np.full((batch, L), tokenizer.pad_id, dtype=np.int64)
    weights = np.zeros((batch, L), dtype=np.float64)
    for i in range(batch):
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        chunk = corpus_ids[start:start + L + 1]
        n = min(len(chunk) - 1, L)
        X[i, :n] = chunk[:n]
        targets[i, :n] = chunk[1:n + 1]
        weights[i, :n] = 1.0
    return X, targets, weights


# ---------------------------------------------------------------------------
# 7. 采样 + 评分 (RL rollout)
# ---------------------------------------------------------------------------

def sample_group(model: TinyTransformer, tokenizer: CharTokenizer,
                 prompt: str, answer: str, rng: np.random.Generator,
                 args: argparse.Namespace) -> list[dict]:
    """对一个 prompt 采样 group_size 条回答并打分。"""
    prompt_ids = tokenizer.encode(prompt)
    seqs, logprob_rows = model.generate_batch(
        prompt_ids, args.group_size, args.max_new,
        args.temperature, rng, tokenizer.eos_id, args.max_pos)
    group: list[dict] = []
    for seq, lps in zip(seqs, logprob_rows):
        resp = seq[len(prompt_ids):]
        truncated = tokenizer.eos_id not in resp
        ans_ids = [t for t in resp if t != tokenizer.eos_id]
        ans = tokenizer.decode(ans_ids)
        score = 0.0 if truncated else score_response(answer, ans, args.score_mode)
        group.append({
            "prompt_ids": prompt_ids,
            "resp_ids": resp,
            "logprobs": lps,
            "score": score,
            "n_tokens": len(ans_ids),
            "answer": ans,
            "truncated": truncated,
        })
    return group


# ---------------------------------------------------------------------------
# 8. 训练步骤
# ---------------------------------------------------------------------------

def pretrain_step(model: TinyTransformer, opt: Adam, tokenizer: CharTokenizer,
                  rng: np.random.Generator, args: argparse.Namespace) -> float:
    """一个 MLE 预训练步: 学会按标准答案作答。"""
    batch = [make_prompt(rng, args.task) for _ in range(args.batch_prompts * args.group_size)]
    X, targets, weights = build_pretrain_batch(
        batch, tokenizer, verbose=args.verbose_pretrain)
    logits, cache = model.forward(X)
    model._set_cache(cache)
    loss, dlogits = token_loss(logits, targets, weights, beta=args.entropy_beta)
    grads = model.backward(X, dlogits, tokenizer.pad_id)
    opt.step(grads, args.lr, args.grad_clip)
    return loss


def lm_step(model: TinyTransformer, opt: Adam, tokenizer: CharTokenizer,
            corpus_ids: list[int], rng: np.random.Generator,
            args: argparse.Namespace) -> float:
    """一个无监督语言建模步: 从语料随机窗口预测下一个字符。"""
    X, targets, weights = build_lm_batch(
        corpus_ids, args.lm_len, args.batch_prompts, rng, tokenizer)
    logits, cache = model.forward(X)
    model._set_cache(cache)
    loss, dlogits = token_loss(logits, targets, weights, beta=args.entropy_beta)
    grads = model.backward(X, dlogits, tokenizer.pad_id)
    opt.step(grads, args.lr, args.grad_clip)
    return loss


def rl_step(model: TinyTransformer, opt: Adam, tokenizer: CharTokenizer,
            rng: np.random.Generator, speed: SpeedTracker,
            ref_model: TinyTransformer | None,
            args: argparse.Namespace) -> dict:
    """一个 RL 步: reward = speed(上一批长度) × score, REINFORCE + 组内 baseline。"""
    samples: list[dict] = []
    lengths: list[float] = []
    speeds = []
    for _ in range(args.batch_prompts):
        prompt, answer = make_prompt(rng, args.task)
        group = sample_group(model, tokenizer, prompt, answer, rng, args)
        sp = speed.speed()  # 关键: 用"之前"的长度算 speed
        speeds.append(sp)
        for s in group:
            s["reward"] = sp * s["score"]
        mean_r = float(np.mean([s["reward"] for s in group]))
        for s in group:
            s["adv"] = s["reward"] - mean_r
        samples.extend(group)
        lengths.append(float(np.mean([s["n_tokens"] for s in group])))
    speed.update(float(np.mean(lengths)))  # 更新 EMA, 供下一轮使用

    X, targets, weights = build_rl_batch(samples, tokenizer)
    logits, cache = model.forward(X)
    model._set_cache(cache)
    ref_logits = None
    if ref_model is not None:
        ref_logits, _ = ref_model.forward(X)
    loss, dlogits = token_loss(
        logits, targets, weights, beta=args.entropy_beta,
        ref_logits=ref_logits, kl_beta=args.kl_beta)
    grads = model.backward(X, dlogits, tokenizer.pad_id)
    opt.step(grads, args.rl_lr, args.grad_clip)

    return {
        "loss": loss,
        "mean_score": float(np.mean([s["score"] for s in samples])),
        "mean_len": float(np.mean(lengths)),
        "speed": float(np.mean(speeds)),
        "trunc_rate": float(np.mean([s["truncated"] for s in samples])),
    }


# ---------------------------------------------------------------------------
# 9. 评估 / 生成示例
# ---------------------------------------------------------------------------

def evaluate(model: TinyTransformer, tokenizer: CharTokenizer,
             rng: np.random.Generator, args: argparse.Namespace,
             n: int = 30) -> dict:
    """贪心解码评估: 正确率 + 平均长度。"""
    correct = 0
    lengths = []
    for _ in range(n):
        prompt, answer = make_prompt(rng, args.task)
        prompt_ids = tokenizer.encode(prompt)
        seqs, _ = model.generate_batch(
            prompt_ids, 1, args.max_new, 0.0,
            rng, tokenizer.eos_id, args.max_pos)
        resp = seqs[0][len(prompt_ids):]
        ans_ids = [t for t in resp if t != tokenizer.eos_id]
        ans = tokenizer.decode(ans_ids)
        correct += int(ans == answer)
        lengths.append(len(ans_ids))
    return {
        "accuracy": correct / n,
        "mean_len": float(np.mean(lengths)),
    }


def show_examples(model: TinyTransformer, tokenizer: CharTokenizer,
                  rng: np.random.Generator, args: argparse.Namespace,
                  n: int = 5) -> None:
    print("示例 (prompt -> 输出):")
    for _ in range(n):
        prompt, answer = make_prompt(rng, args.task)
        prompt_ids = tokenizer.encode(prompt)
        seqs, _ = model.generate_batch(
            prompt_ids, 1, args.max_new, 0.0,
            rng, tokenizer.eos_id, args.max_pos)
        resp = seqs[0][len(prompt_ids):]
        ans_ids = [t for t in resp if t != tokenizer.eos_id]
        ans = tokenizer.decode(ans_ids)
        mark = "✓" if ans == answer else "✗"
        print(f"  {prompt} -> {ans!r} (标准: {answer!r}) {mark}")


# ---------------------------------------------------------------------------
# 10. checkpoint (含词表)
# ---------------------------------------------------------------------------

def load_config(out_dir: str) -> dict | None:
    """读取 checkpoint 目录下的 config.json, 不存在则返回 None。"""
    path = os.path.join(out_dir, "config.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenizer_from_config(cfg: dict | None) -> CharTokenizer:
    """按 config 重建分词器; 无 vocab 时退回默认算术词表 (兼容旧 checkpoint)。"""
    if cfg and isinstance(cfg.get("vocab"), list):
        return CharTokenizer(vocab=cfg["vocab"])
    return CharTokenizer()


def save_checkpoint(model: TinyTransformer, tokenizer: CharTokenizer,
                    args: argparse.Namespace, out_dir: str) -> None:
    """保存 checkpoint。

    用 ``savez_compressed``: 参数量不变但磁盘占用更小 (降本), 且顺带把
    ``d_mlp`` / ``dtype`` 写进 config, 结构化剪枝后的窄 MLP 模型也能原样恢复。
    """
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "model.npz"), **model.params)
    cfg = dict(vars(args))
    # 模型结构以模型自身为准记录 (调用方传的 args 未必齐全, 剪枝后的窄
    # attn_dim / d_mlp 必须原样落盘, 否则恢复时形状对不上)。
    cfg.update({
        "vocab": list(tokenizer.chars),
        "d_model": model.d_model,
        "n_layers": model.n_layers,
        "n_heads": model.n_heads,
        "max_pos": model.max_pos,
        "d_mlp": model.d_mlp,
        "attn_dim": model.attn_dim,
        "dtype": model.dtype.name,
    })
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_checkpoint(model: TinyTransformer, tokenizer: CharTokenizer,
                    out_dir: str) -> None:
    """把 checkpoint 参数载入 model。

    若 checkpoint 词表比当前模型小 (例如恢复后追加了中文词表), 则只拷贝
    共有的前缀, 新增部分保持当前值 (零初始化); dtype 与当前模型不一致时
    按模型 dtype 转换 (float32/float64 两种存档可互通)。
    """
    data = np.load(os.path.join(out_dir, "model.npz"))
    for k in model.params:
        if k not in data:
            continue
        d = data[k]
        dt = model.params[k].dtype
        if d.shape == model.params[k].shape:
            model.params[k] = d.astype(dt, copy=False)
        elif d.ndim == model.params[k].ndim and all(
                ds <= ms for ds, ms in zip(d.shape, model.params[k].shape)):
            sl = tuple(slice(0, ds) for ds in d.shape)
            model.params[k][sl] = d.astype(dt, copy=False)


MODEL_KEYS = ("d_model", "n_layers", "n_heads", "max_pos", "d_mlp",
              "attn_dim", "dtype")


def model_from_config(tokenizer: CharTokenizer, cfg: dict | None,
                      fallback: dict | None = None,
                      seed: int = 0) -> TinyTransformer:
    """按 config.json 重建模型。

    兼容旧 checkpoint (没有 ``d_mlp`` / ``dtype`` 时退回 4×d_model 与
    float64), 也让剪枝工具产出的窄 MLP / 窄注意力 checkpoint 能被各个入口直接加载。
    """
    kw: dict = dict(fallback or {})
    if cfg:
        for k in MODEL_KEYS:
            if cfg.get(k) is not None:
                kw[k] = cfg[k]
    return TinyTransformer(tokenizer.vocab_size, seed=seed, **kw)


# ---------------------------------------------------------------------------
# 11. 梯度数值校验
# ---------------------------------------------------------------------------

def _loss_for_gradcheck(model: TinyTransformer, X: np.ndarray,
                        targets: np.ndarray, weights: np.ndarray,
                        beta: float) -> float:
    logits, _ = model.forward(X)
    loss, _ = token_loss(logits, targets, weights, beta=beta)
    return loss


def check_gradients(seed: int = 0) -> None:
    """对一个小模型做中心差分梯度校验。"""
    rng = np.random.default_rng(seed + 1)
    model = TinyTransformer(vocab_size=12, d_model=8, n_layers=1,
                            n_heads=2, max_pos=16, seed=seed)
    B, L = 3, 12
    X = rng.integers(0, 11, size=(B, L)).astype(np.int64)
    X[:, -3:] = model.params["W_e"].shape[0] - 1  # 末尾 padding
    targets = rng.integers(0, 12, size=(B, L)).astype(np.int64)
    weights = (rng.random((B, L)) < 0.6).astype(np.float64) * rng.uniform(-1, 1, (B, L))
    weights[:, -3:] = 0.0  # padding 位置不参与损失 (与真实训练一致)
    beta = 0.05

    logits, cache = model.forward(X)
    model._set_cache(cache)
    _, dlogits = token_loss(logits, targets, weights, beta=beta)
    grads = model.backward(X, dlogits, model.params["W_e"].shape[0] - 1)

    eps = 1e-4
    max_rel = 0.0
    max_abs = 0.0
    worst = None
    for key, g in grads.items():
        flat = model.params[key].reshape(-1)
        for _ in range(4):
            idx = int(rng.integers(0, flat.size))
            orig = flat[idx]
            flat[idx] = orig + eps
            lp = _loss_for_gradcheck(model, X, targets, weights, beta)
            flat[idx] = orig - eps
            lm = _loss_for_gradcheck(model, X, targets, weights, beta)
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = float(g.reshape(-1)[idx])
            rel = abs(num - ana) / max(abs(num), abs(ana), 1e-9)
            abs_err = abs(num - ana)
            if rel > max_rel:
                max_rel = rel
                worst = (key, idx, ana, num)
            max_abs = max(max_abs, abs_err)
    if max_rel < 1e-3:
        print(f"[PASS] 梯度校验通过, 最大相对误差 = {max_rel:.2e}, "
              f"最大绝对误差 = {max_abs:.2e}")
    else:
        print(f"[FAIL] 梯度校验失败, 最大相对误差 = {max_rel:.2e}, "
              f"最大绝对误差 = {max_abs:.2e}, 最差参数 = {worst}")
        sys.exit(1)
