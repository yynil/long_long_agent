"""Bounded byte-token generation; generated fast state is never a slow-state commit."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from .inference_math import inference_linear
from .rwkv7_stateful import stateful_forward


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 20260905
    max_seconds: float = 120.0

    def __post_init__(self):
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not 0 < self.top_p <= 1 or not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("invalid sampling or time budget")


def sample_token(logits, config: GenerationConfig, generator, maximum_token_id: int) -> int:
    values = logits.float().flatten().clone()
    if not 0 < maximum_token_id < values.numel():
        raise ValueError("invalid vocabulary limit")
    if not torch.isfinite(values[: maximum_token_id + 1]).all():
        raise ValueError("nonfinite generation logits")
    values[maximum_token_id + 1 :] = -torch.inf
    if config.temperature == 0:
        return int(values.argmax())
    probabilities = torch.softmax(values / config.temperature, dim=-1)
    sorted_probs, indices = probabilities.sort(descending=True)
    exclude = sorted_probs.cumsum(-1) - sorted_probs >= config.top_p
    sorted_probs[exclude] = 0
    return int(indices[torch.multinomial(sorted_probs, 1, generator=generator)])


@torch.inference_mode()
def generate(
    network,
    tokenizer,
    prompt_ids,
    config: GenerationConfig,
    *,
    state=None,
    stop_bytes=(b"\n\nUser:",),
):
    if not prompt_ids or len(prompt_ids) > network.args.ctx_len:
        raise ValueError("prompt is empty or exceeds the configured context")
    if any(token not in tokenizer.id_to_token and token != 0 for token in prompt_ids):
        raise ValueError("undefined prompt token")
    device = network.emb.weight.device
    generator = torch.Generator(device=device).manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    # Large aligned prefix uses the audited CUDA kernel; no padding alters state.
    prefix, logits = list(prompt_ids), None
    next_state = state
    for offset in range(0, len(prefix), 128):
        chunk = torch.tensor([prefix[offset : offset + 128]], device=device)
        hidden, next_state = stateful_forward(network, chunk, state=next_state, return_logits=False)
        logits = inference_linear(network.head, hidden[:, -1:])
    output, raw = [], bytearray()
    stop_reason = "token_budget"
    for _ in range(config.max_new_tokens):
        if time.monotonic() - started >= config.max_seconds:
            stop_reason = "time_budget"
            break
        token = sample_token(logits, config, generator, tokenizer.maximum_token_id)
        if token == 0:
            stop_reason = "eod"
            break
        output.append(token)
        raw.extend(tokenizer.token_bytes(token))
        if any(raw.endswith(marker) for marker in stop_bytes):
            stop_reason = "stop_sequence"
            break
        logits, next_state = stateful_forward(
            network, torch.tensor([[token]], device=device), state=next_state
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "token_ids": output,
        "text": bytes(raw).decode("utf-8", errors="replace"),
        "utf8_valid": _valid_utf8(bytes(raw)),
        "stop_reason": stop_reason,
        "model_seconds": time.monotonic() - started,
    }


def _valid_utf8(value: bytes) -> bool:
    try:
        value.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True
