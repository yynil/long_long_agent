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


def test_trie_matches_legacy_longest_match_for_full_vocabulary_and_mixed_bytes():
    import random
    from pathlib import Path

    from src.training.tokenizer import RWKVByteTokenizer

    vocabulary = Path("external/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt")
    if not vocabulary.exists():
        pytest.skip("pinned vocabulary is not available")
    tokenizer = RWKVByteTokenizer(vocabulary)
    grouped = {}
    for token in tokenizer.token_to_id:
        grouped.setdefault(token[:2], []).append(token)
    for values in grouped.values():
        values.sort(key=len, reverse=True)

    def legacy(source):
        tokens, offset = [], 0
        while offset < len(source):
            token = next(
                (
                    t
                    for t in grouped.get(source[offset : offset + 2], ())
                    if source.startswith(t, offset)
                ),
                source[offset : offset + 1],
            )
            tokens.append(tokenizer.token_to_id[token])
            offset += len(token)
        return tokens

    for token, token_id in tokenizer.token_to_id.items():
        assert tokenizer.encode_bytes(token) == [token_id]
    generator = random.Random(20260905)
    vocabulary_bytes = list(tokenizer.token_to_id)
    examples = [bytes(range(256)), b"", "中文路径/修复.py\n\nUser: verify".encode()]
    examples.extend(b"".join(generator.choices(vocabulary_bytes, k=32)) for _ in range(128))
    examples.extend(generator.randbytes(256) for _ in range(128))
    for example in examples:
        assert tokenizer.encode_bytes(example) == legacy(example)
