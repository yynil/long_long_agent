from __future__ import annotations

from pathlib import Path

import pytest

from src.training.tokenizer import RWKVByteTokenizer


def write_test_vocabulary(path: Path, *, omit_byte: int | None = None) -> None:
    lines = []
    for value in range(256):
        if value != omit_byte:
            token = bytes([value])
            lines.append(f"{value + 1} {token!r} 1")
    lines.extend(
        [
            "257 b'ab' 2",
            "258 b'abc' 3",
            "259 '你好' 6",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_rwkv_tokenizer_uses_greedy_longest_byte_match(tmp_path: Path) -> None:
    vocabulary = tmp_path / "vocab.txt"
    write_test_vocabulary(vocabulary)
    tokenizer = RWKVByteTokenizer(vocabulary)

    tokens = tokenizer.encode("zabc你好")
    assert tokens == [ord("z") + 1, 258, 259]
    assert tokenizer.decode(tokens) == "zabc你好"
    assert tokenizer.defined_token_count == 259
    assert tokenizer.maximum_token_id == 259
    assert tokenizer.minimum_embedding_size == 260


def test_rwkv_tokenizer_requires_complete_byte_fallback(tmp_path: Path) -> None:
    vocabulary = tmp_path / "bad-vocab.txt"
    write_test_vocabulary(vocabulary, omit_byte=255)

    with pytest.raises(ValueError, match="all 256"):
        RWKVByteTokenizer(vocabulary)
