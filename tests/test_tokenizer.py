"""动态字符级分词器测试。"""

from __future__ import annotations

import pytest

from enochmodel1.enoch import BASE_CHARS, EOS_CHAR, CharTokenizer


def test_default_vocab() -> None:
    tok = CharTokenizer()
    assert tok.vocab_size == len(BASE_CHARS) + 1
    assert tok.eos_id == tok.stoi[EOS_CHAR]
    assert tok.pad_id == tok.stoi[" "]
    assert tok.decode(tok.encode("12+3= 4")) == "12+3= 4"


def test_add_new_chars() -> None:
    tok = CharTokenizer()
    assert tok.add("你好 world") == 7  # 你 好 w o r l d (空格已有)
    assert tok.vocab_size == len(BASE_CHARS) + 1 + 7
    assert tok.encode("你好") == [tok.stoi["你"], tok.stoi["好"]]
    assert tok.decode(tok.encode("你好")) == "你好"
    assert tok.add("你好") == 0  # 重复添加不增加


def test_add_keeps_existing_ids_stable() -> None:
    tok = CharTokenizer()
    ids_before = tok.encode("123+4=")
    tok.add("中文")
    assert tok.encode("123+4=") == ids_before


def test_custom_vocab_ensures_base_chars() -> None:
    tok = CharTokenizer(vocab=["a", "b"])
    assert "0" in tok.stoi
    assert EOS_CHAR in tok.stoi
    assert tok.decode(tok.encode("ab0+1")) == "ab0+1"


def test_encode_unknown_char_raises() -> None:
    tok = CharTokenizer()
    with pytest.raises(KeyError):
        tok.encode("你好")
