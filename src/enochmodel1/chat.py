"""
EnochModel1: 日常对话入口 (enochmodel1.chat)
===================================

交互式对话, 手动输入即可训练:

* 直接输入一句话 -> 模型生成回复, 该轮对话存入记忆文件 (JSONL);
* !c 正确的回答    -> 纠正模型 (MLE 在线更新一步);
* !s 0~10          -> 给上一条回复打分 (RL 在线更新一步, 10=完美);
* !train           -> 把记忆里所有"纠正过/打过分的对话"批量回放训练;
* !save [目录]     -> 保存 checkpoint;
* !q               -> 退出 (自动保存记忆)。

模型是字符级的: 遇到没见过的字会自动扩充词表并同步扩展模型, 因此
中文、标点、表情符号都可以直接输入。

示例
----
    uv run enoch-chat                                  # 全新模型
    uv run enoch-chat --checkpoint checkpoints/pretrain # 接着预训练模型聊
    uv run enoch-chat --temperature 0.2                 # 更保守的回复
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np

from ._paths import CHECKPOINTS_DIR, DATA_DIR, resolve
from .enoch import (
    Adam,
    CharTokenizer,
    TinyTransformer,
    build_pretrain_batch,
    build_rl_batch,
    generate_text,
    grow_max_pos,
    grow_vocab,
    load_checkpoint,
    load_config,
    save_checkpoint,
    token_loss,
    tokenizer_from_config,
)

HELP = """\
命令:
  <直接输入文本>      对话, 模型会回复 (并记入记忆)
  !c <正确的回答>     纠正上一条回复, 在线训练一步 (MLE)
  !s <0~10>           给上一条回复打分, 在线训练一步 (RL, 10=完美)
  !train              把记忆里纠正过/打过分的对话批量回放训练
  !save [目录]        保存 checkpoint (默认: checkpoints/chat)
  !stats              显示词表/记忆/模型信息
  !h / !help          显示本帮助
  !q / !quit          退出 (自动保存记忆)
"""

HELP_DAILY = """\
日常训练模式 (输入语料 -> 模型输出 -> 给评分):
  <直接输入文本>      输入一段语料/题目, 模型续写或回答
  评分 (0~10)         输出后输入分数, 立即 RL 训练一步 (10=完美)
  !c <正确的回答>     不评分, 改为纠正 (MLE 训练一步)
  (回车)              跳过评分, 不训练直接进入下一轮
  !q                  退出日常训练模式 (再按 !q 或 !quit 退出程序)
  !quit               直接退出程序
"""

# 纯算术表达式 (数字 / 运算符 / 空格 / 等号), 用于格式归一化与贪心解码
ARITH_RE = re.compile(r"^[0-9+\-*\s=]+$")


def is_arithmetic(text: str) -> bool:
    """是否看起来是算术表达式 (至少含一个数字)。"""
    return bool(ARITH_RE.match(text)) and any(ch.isdigit() for ch in text)


def normalize_arith(text: str) -> str:
    """把 '1+2' 归一化成训练格式 '1+2='; 已含 '=' 的保持不变。"""
    s = text.strip()
    if is_arithmetic(s) and "=" not in s:
        return s + "="
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EnochModel1: 日常对话 (手动输入, 可在线训练)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="加载的 checkpoint 目录 (默认自动找 checkpoints)")
    p.add_argument("--out-dir", type=str, default=str(CHECKPOINTS_DIR / "chat"),
                   help="!save 默认保存目录")
    p.add_argument("--memory", type=str, default=str(DATA_DIR / "chat_memory.jsonl"),
                   help="对话记忆文件 (JSONL)")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="采样温度 (0=贪心)")
    p.add_argument("--max-new", type=int, default=48,
                   help="回复最大 token 数")
    p.add_argument("--d-model", type=int, default=64, help="模型宽度 (仅新模型)")
    p.add_argument("--n-layers", type=int, default=2, help="层数 (仅新模型)")
    p.add_argument("--n-heads", type=int, default=4, help="注意力头数 (仅新模型)")
    p.add_argument("--max-pos", type=int, default=256, help="最大序列长度")
    p.add_argument("--lr", type=float, default=3e-3, help="在线训练学习率")
    p.add_argument("--entropy-beta", type=float, default=0.02,
                   help="在线训练熵正则强度")
    p.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪范数")
    p.add_argument("--batch-prompts", type=int, default=4,
                   help="!train 每批样本数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    return p.parse_args()


def load_memory(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def save_memory(path: str, entries: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)


def do_mle_step(model: TinyTransformer, opt: Adam, tokenizer: CharTokenizer,
                args: argparse.Namespace, prompt: str, expected: str) -> float:
    """用 (prompt, 期望回复) 做一步 MLE 在线训练。"""
    grow_vocab(model, opt, tokenizer, prompt + expected)
    X, targets, weights = build_pretrain_batch([(prompt, expected)], tokenizer)
    logits, cache = model.forward(X)
    model._set_cache(cache)
    loss, dlogits = token_loss(logits, targets, weights, beta=args.entropy_beta)
    grads = model.backward(X, dlogits, tokenizer.pad_id)
    opt.step(grads, args.lr, args.grad_clip)
    return float(loss)


def do_rl_step(model: TinyTransformer, opt: Adam, tokenizer: CharTokenizer,
               args: argparse.Namespace, prompt: str, response: str,
               score: float) -> float:
    """用用户打分做一步 REINFORCE 在线更新。

    score ∈ [0, 10], advantage = score/10 - 0.5: 高于 5 分加强该回复,
    低于 5 分减弱。
    """
    grow_vocab(model, opt, tokenizer, prompt + response)
    adv = score / 10.0 - 0.5
    samples = [{
        "prompt_ids": tokenizer.encode(prompt),
        "resp_ids": tokenizer.encode(response) + [tokenizer.eos_id],
        "adv": adv,
    }]
    X, targets, weights = build_rl_batch(samples, tokenizer)
    logits, cache = model.forward(X)
    model._set_cache(cache)
    loss, dlogits = token_loss(logits, targets, weights, beta=args.entropy_beta)
    grads = model.backward(X, dlogits, tokenizer.pad_id)
    opt.step(grads, args.lr, args.grad_clip)
    return float(loss)


def train_on_memory(model: TinyTransformer, opt: Adam,
                    tokenizer: CharTokenizer, args: argparse.Namespace,
                    memory: list[dict]) -> dict:
    """把记忆里"纠正过 / 打过分的对话"批量回放训练, 返回统计。"""
    mle = [e for e in memory if e.get("expected")]
    rl = [e for e in memory if e.get("score") is not None]
    mle_losses: list[float] = []
    rl_losses: list[float] = []

    for start in range(0, len(mle), args.batch_prompts):
        pairs = [(e["prompt"], e["expected"]) for e in mle[start:start + args.batch_prompts]]
        for pr, ex in pairs:
            grow_vocab(model, opt, tokenizer, pr + ex)
        X, targets, weights = build_pretrain_batch(pairs, tokenizer)
        logits, cache = model.forward(X)
        model._set_cache(cache)
        loss, dlogits = token_loss(logits, targets, weights, beta=args.entropy_beta)
        grads = model.backward(X, dlogits, tokenizer.pad_id)
        opt.step(grads, args.lr, args.grad_clip)
        mle_losses.append(float(loss))

    for start in range(0, len(rl), args.batch_prompts):
        samples = []
        for e in rl[start:start + args.batch_prompts]:
            grow_vocab(model, opt, tokenizer, e["prompt"] + e["response"])
            samples.append({
                "prompt_ids": tokenizer.encode(e["prompt"]),
                "resp_ids": tokenizer.encode(e["response"]) + [tokenizer.eos_id],
                "adv": e["score"] / 10.0 - 0.5,
            })
        X, targets, weights = build_rl_batch(samples, tokenizer)
        logits, cache = model.forward(X)
        model._set_cache(cache)
        loss, dlogits = token_loss(logits, targets, weights, beta=args.entropy_beta)
        grads = model.backward(X, dlogits, tokenizer.pad_id)
        opt.step(grads, args.lr, args.grad_clip)
        rl_losses.append(float(loss))

    return {
        "mle": len(mle),
        "rl": len(rl),
        "mle_loss": float(np.mean(mle_losses)) if mle_losses else None,
        "rl_loss": float(np.mean(rl_losses)) if rl_losses else None,
    }


def parse_score(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        s = float(text)
    except ValueError:
        return None
    if not (0.0 <= s <= 10.0):
        return None
    return s


def main() -> None:
    args = parse_args()
    args.checkpoint = resolve(args.checkpoint)
    args.out_dir = resolve(args.out_dir) or str(CHECKPOINTS_DIR / "chat")
    args.memory = resolve(args.memory) or str(DATA_DIR / "chat_memory.jsonl")
    rng = np.random.default_rng(args.seed)

    # ---- 找 checkpoint: 显式指定 > 默认 checkpoints 目录 > 全新模型 ----
    ckpt = args.checkpoint
    if ckpt is None and os.path.isfile(os.path.join(
            str(CHECKPOINTS_DIR), "model.npz")):
        ckpt = str(CHECKPOINTS_DIR)

    cfg = load_config(ckpt) if ckpt else None
    tokenizer = tokenizer_from_config(cfg)
    if cfg:
        for key in ("d_model", "n_layers", "n_heads", "max_pos"):
            if key in cfg:
                setattr(args, key, cfg[key])

    model = TinyTransformer(tokenizer.vocab_size, args.d_model, args.n_layers,
                            args.n_heads, args.max_pos, seed=args.seed)
    if ckpt:
        load_checkpoint(model, tokenizer, ckpt)
    opt = Adam(model.params)

    memory = load_memory(args.memory)

    print("=" * 64)
    print("EnochModel1: 日常对话 (输入 !h 查看命令)")
    print(f"模型: d={args.d_model}, layers={args.n_layers}, "
          f"max_pos={args.max_pos}, 词表={tokenizer.vocab_size}")
    print(f"加载: {ckpt or '全新随机模型 (尚未训练)'}")
    print(f"记忆: {args.memory} ({len(memory)} 条)")
    print("=" * 64)

    last: dict | None = None  # 当前轮对话 (供 !c / !s 使用)
    daily = False                # 日常训练模式
    trained_today = 0            # 本次会话里在线训练过的样本数

    while True:
        try:
            label = "训练> " if daily else "你: "
            line = input(label).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("!"):
            cmd, _, rest = line.partition(" ")
            cmd = cmd.lower()
            if cmd in ("!h", "!help"):
                print(HELP_DAILY if daily else HELP)
            elif cmd in ("!q", "!exit") and daily:
                print("退出日常训练模式 (再按 !q 或 !quit 退出程序)")
                if trained_today > 0:
                    save_checkpoint(model, tokenizer, args, args.out_dir)
                    print(f"本次训练 {trained_today} 条, checkpoint 已自动保存到 {args.out_dir}")
                    trained_today = 0
                daily = False
            elif cmd in ("!q", "!quit", "!exit", "!bye"):
                break
            elif cmd in ("!d", "!daily", "!train-mode", "!日常训练"):
                daily = True
                print("进入日常训练模式: 输入语料 -> 模型输出 -> 给评分 (0~10)")
                print("  输入 !h 查看训练模式命令; !q 退出训练模式")
            elif cmd in ("!stats",):
                print(f"词表: {tokenizer.vocab_size} | 记忆: {len(memory)} 条 | "
                      f"模型参数: {sum(int(v.size) for v in model.params.values()):,}")
            elif cmd in ("!c", "!correct"):
                expected = rest.strip()
                if not expected:
                    print("用法: !c 正确的回答")
                    continue
                if last is None:
                    print("还没有可纠正的回复, 先输入一句话试试")
                    continue
                grow_vocab(model, opt, tokenizer, last["prompt"] + expected)
                grow_max_pos(model, opt, len(tokenizer.encode(
                    last["prompt"] + expected)) + 4)
                loss = do_mle_step(model, opt, tokenizer, args,
                                   last["prompt"], expected)
                last["expected"] = expected
                save_memory(args.memory, memory)
                trained_today += 1
                print(f"已按正确回答训练 (MLE loss={loss:.4f}), 记住了这条对话")
            elif cmd in ("!s", "!score"):
                score = parse_score(rest)
                if score is None:
                    print("用法: !s 0~10 (10=完美)")
                    continue
                if last is None:
                    print("还没有可打分的回复, 先输入一句话试试")
                    continue
                loss = do_rl_step(model, opt, tokenizer, args,
                                  last["prompt"], last["response"], score)
                last["score"] = score
                save_memory(args.memory, memory)
                trained_today += 1
                print(f"已按 {score}/10 训练 (RL loss={loss:.4f})")
            elif cmd in ("!t", "!train"):
                stats = train_on_memory(model, opt, tokenizer, args, memory)
                print(f"回放训练完成: MLE {stats['mle']} 条"
                      + (f" (loss={stats['mle_loss']:.4f})" if stats["mle_loss"] else "")
                      + f", RL {stats['rl']} 条"
                      + (f" (loss={stats['rl_loss']:.4f})" if stats["rl_loss"] else ""))
                if stats["mle"] + stats["rl"] == 0:
                    print("  记忆里还没有可训练的数据: 用 !c 纠正或用 !s 打分")
            elif cmd in ("!save",):
                out_dir = rest.strip() or args.out_dir
                save_checkpoint(model, tokenizer, args, out_dir)
                print(f"checkpoint 已保存到 {out_dir}")
            else:
                print(f"未知命令 {cmd!r} (输入 !h 查看帮助)")
            continue

        # ---- 普通对话 ----
        prompt = normalize_arith(line)
        arith = is_arithmetic(prompt)
        grow_vocab(model, opt, tokenizer, prompt)
        grow_max_pos(model, opt, len(tokenizer.encode(prompt)) + args.max_new)
        response = generate_text(model, tokenizer, prompt, args.max_new,
                                 (0.0 if arith else args.temperature), rng).strip()
        if not response:
            # 模型直接输出 EOS (常见于没见过的问题): 先试贪心, 仍空则给提示
            response = generate_text(model, tokenizer, prompt, args.max_new,
                                     0.0, rng).strip()
        if not response:
            response = "这个我还没学会，试试用 !c 教我正确答案。"
        print(f"Enoch: {response}")
        print("  (可直接输入下一句; 想训练这条: !s 0~10 打分 / !c 纠正)")
        entry = {
            "prompt": prompt,
            "response": response,
            "expected": None,
            "score": None,
            "ts": time.time(),
        }
        memory.append(entry)
        save_memory(args.memory, memory)
        last = entry

        # ---- 日常训练模式: 输出后立即要评分 ----
        if not daily:
            continue
        try:
            feedback = input("评分 (0~10, 回车跳过, !c 纠正, !q 退出): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not feedback:
            continue  # 跳过评分, 进入下一轮
        if feedback.startswith("!"):
            fcmd, _, frest = feedback.partition(" ")
            fcmd = fcmd.lower()
            if fcmd in ("!q", "!exit"):
                print("退出日常训练模式 (再按 !q 或 !quit 退出程序)")
                if trained_today > 0:
                    save_checkpoint(model, tokenizer, args, args.out_dir)
                    print(f"本次训练 {trained_today} 条, checkpoint 已自动保存到 {args.out_dir}")
                    trained_today = 0
                daily = False
            elif fcmd in ("!c", "!correct"):
                expected = frest.strip()
                if not expected:
                    print("用法: !c 正确的回答")
                    continue
                grow_vocab(model, opt, tokenizer, last["prompt"] + expected)
                grow_max_pos(model, opt, len(tokenizer.encode(
                    last["prompt"] + expected)) + 4)
                loss = do_mle_step(model, opt, tokenizer, args,
                                   last["prompt"], expected)
                last["expected"] = expected
                save_memory(args.memory, memory)
                trained_today += 1
                print(f"已按正确回答训练 (MLE loss={loss:.4f}), 记住了这条对话")
            else:
                print(f"未知反馈命令 {fcmd!r} (评分 0~10 / !c 纠正 / !q 退出)")
            continue
        score = parse_score(feedback)
        if score is None:
            print("评分需为 0~10 的数字, 已跳过")
            continue
        loss = do_rl_step(model, opt, tokenizer, args,
                          last["prompt"], last["response"], score)
        last["score"] = score
        save_memory(args.memory, memory)
        trained_today += 1
        print(f"已按 {score}/10 训练 (RL loss={loss:.4f})")

    save_memory(args.memory, memory)
    if trained_today > 0:
        save_checkpoint(model, tokenizer, args, args.out_dir)
        print(f"本次训练 {trained_today} 条, checkpoint 已自动保存到 {args.out_dir}")
    print(f"\n再见! 记忆已保存到 {args.memory} ({len(memory)} 条)")


if __name__ == "__main__":
    main()
