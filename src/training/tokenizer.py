"""Safe, dependency-free tokenizer for the pinned RWKV byte vocabulary."""

from __future__ import annotations

import ast
from pathlib import Path


class RWKVByteTokenizer:
    """Greedy longest-match tokenizer compatible with RWKV's G1 vocabulary."""

    def __init__(self, vocabulary_path: str | Path):
        self.vocabulary_path = Path(vocabulary_path)
        self.id_to_token: dict[int, bytes] = {}
        self.token_to_id: dict[bytes, int] = {}
        for line_number, raw_line in enumerate(
            self.vocabulary_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            first_space = raw_line.find(" ")
            last_space = raw_line.rfind(" ")
            if first_space <= 0 or last_space <= first_space:
                raise ValueError(f"invalid vocabulary line {line_number}")
            token_id = int(raw_line[:first_space])
            token_literal = ast.literal_eval(raw_line[first_space + 1 : last_space])
            token = (
                token_literal.encode("utf-8") if isinstance(token_literal, str) else token_literal
            )
            if not isinstance(token, bytes) or not token:
                raise ValueError(f"vocabulary line {line_number} is not a non-empty byte token")
            if len(token) != int(raw_line[last_space + 1 :]):
                raise ValueError(f"vocabulary line {line_number} has an incorrect byte length")
            if token_id in self.id_to_token or token in self.token_to_id:
                raise ValueError(f"duplicate token at vocabulary line {line_number}")
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        missing_bytes = [value for value in range(256) if bytes([value]) not in self.token_to_id]
        if missing_bytes:
            raise ValueError("vocabulary must contain all 256 single-byte tokens")
        self._trie: dict = {}
        for token, token_id in self.token_to_id.items():
            node = self._trie
            for value in token:
                node = node.setdefault(value, {})
            node[-1] = token_id

    @property
    def defined_token_count(self) -> int:
        return len(self.id_to_token)

    @property
    def maximum_token_id(self) -> int:
        return max(self.id_to_token, default=0)

    @property
    def minimum_embedding_size(self) -> int:
        """Smallest embedding that can represent all IDs defined by this file."""
        return self.maximum_token_id + 1

    def encode_bytes(self, source: bytes) -> list[int]:
        tokens: list[int] = []
        offset = 0
        while offset < len(source):
            node = self._trie
            cursor, end, token_id = offset, offset + 1, None
            while cursor < len(source):
                node = node.get(source[cursor])
                if node is None:
                    break
                cursor += 1
                if -1 in node:
                    end, token_id = cursor, node[-1]
            if token_id is None:
                raise ValueError("vocabulary does not cover an input byte")
            tokens.append(token_id)
            offset = end
        return tokens

    def encode(self, source: str) -> list[int]:
        return self.encode_bytes(source.encode("utf-8"))

    def token_bytes(self, token_id: int) -> bytes:
        try:
            return self.id_to_token[token_id]
        except KeyError as error:
            raise ValueError(f"token ID {token_id} is not in the byte vocabulary") from error

    def decode_bytes(self, tokens: list[int] | tuple[int, ...]) -> bytes:
        return b"".join(self.token_bytes(token_id) for token_id in tokens)

    def decode(self, tokens: list[int] | tuple[int, ...]) -> str:
        return self.decode_bytes(tokens).decode("utf-8")
